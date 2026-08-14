# train_cifar10_full.py
"""
CIFAR-10 depth=6 / 50-epoch strong-full experiment
for Mini-to-Main Attention.

핵심 원칙
---------
기존 train_cifar10_sanity.py의 모델/손실/학습 구조를 그대로 재사용한다.

그대로 유지:
    - CIFAR-10 train 50,000 / test 10,000
    - img_size=224, patch_size=16, embed_dim=192
    - main_heads=3, mini_heads=1, mini_dim=64
    - budgets=[0,1,2,3]
    - balanced budget exposure
    - Gumbel-ST learned allocation
    - tau geometric annealing: 1.5 -> 0.5
    - CE + 0.01 * direct-mixed diversity loss
    - AdamW(lr=3e-4, wd=0.05)
    - CosineAnnealingLR
    - grad clip=1.0
    - learned/fixed/random inference-time diagnostics

이번 실험에서 변경:
    - depth: 2 -> 6
    - epochs: 10 -> 50

평가 비용 최적화:
    - learned B=0/1/2/3: 매 epoch 전체 test set 평가
    - fixed/random: --diagnostic-every epoch마다 + final epoch
      (default: every 5 epochs)
    - 모델 학습 조건에는 영향을 주지 않는다.

주의:
    현재 attention v1은 모든 Main head를 dense 계산한 뒤 gate를 적용한다.
    따라서 이 실험으로 FLOPs/latency 절감 주장을 하면 안 된다.

    fixed/random은 동일 learned checkpoint의 inference-time ablation이다.
    최종 논문 baseline은 learned/fixed/random 각각 별도 학습이 필요하다.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import train_cifar10_sanity as base
from losses.diversity_loss import HeadDiversityLoss
from models.mini_guided_vit import MiniGuidedViT


# ============================================================
# Parser
# ============================================================

def build_parser():
    # 기존 sanity parser를 그대로 가져와서
    # full experiment에 필요한 default만 변경한다.
    p = base.build_parser()

    p.set_defaults(
        output_dir="./outputs/cifar10_full_d6_e50",
        epochs=50,
        depth=6,
        batch_size=128,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        main_heads=3,
        mini_heads=1,
        mini_dim=64,
        pool_ratio=2,
        mlp_ratio=4.0,
        direct_ratio=0.34,
        alpha_direct=1.0,
        alpha_mixed=0.2,
        lambda_div=0.01,
        lr=3e-4,
        weight_decay=0.05,
        tau_start=1.5,
        tau_end=0.5,
        grad_clip=1.0,
        random_eval_seed=2026,
        seed=42,
        device="auto",
    )

    p.add_argument(
        "--diagnostic-every",
        type=int,
        default=5,
        help=(
            "fixed/random full-test diagnostic interval. "
            "learned budgets are still evaluated every epoch."
        ),
    )

    p.add_argument(
        "--resume",
        type=str,
        default="",
        help=(
            "Path to last.pt for resuming after a Colab/runtime interruption."
        ),
    )

    return p


def validate_args(args):
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0.")

    if args.depth <= 0:
        raise ValueError("--depth must be > 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    if args.main_heads <= 0:
        raise ValueError("--main-heads must be > 0.")

    if args.embed_dim % args.main_heads != 0:
        raise ValueError(
            "--embed-dim must be divisible by --main-heads."
        )

    if args.img_size % args.patch_size != 0:
        raise ValueError(
            "--img-size must be divisible by --patch-size."
        )

    if args.tau_start <= 0 or args.tau_end <= 0:
        raise ValueError(
            "Gumbel temperatures must be positive."
        )

    if args.lambda_div < 0:
        raise ValueError("--lambda-div must be >= 0.")

    if args.diagnostic_every < 0:
        raise ValueError(
            "--diagnostic-every must be >= 0."
        )


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_learned_all(
    model,
    loader,
    device,
    budgets,
    fixed_head_order,
    random_seed,
):
    """
    Every epoch:
        learned B0/B1/B2/B3 full test evaluation.

    base.evaluate_budget()를 그대로 사용하므로
    sanity run과 동일한 accuracy/norm/allocator statistics를 계산한다.
    """
    results = {}

    for budget in budgets:
        results[str(budget)] = (
            base.evaluate_budget(
                model=model,
                loader=loader,
                device=device,
                budget=budget,
                mode="learned",
                fixed_head_order=fixed_head_order,
                random_seed=random_seed + budget,
            )
        )

    return results


@torch.no_grad()
def evaluate_fixed_random(
    model,
    loader,
    device,
    budgets,
    fixed_head_order,
    random_seed,
    learned_results,
):
    """
    Diagnostic epochs only.

    learned result는 이미 계산했으므로 재평가하지 않고,
    fixed/random만 추가한다.
    """
    results = {
        "fixed": {},
        "random": {},
    }

    for mode_idx, mode in enumerate(
        ["fixed", "random"]
    ):
        for budget in budgets:
            b = str(budget)

            if budget == 0:
                copied = dict(
                    learned_results["0"]
                )
                copied["allocation_mode"] = mode
                results[mode][b] = copied
                continue

            results[mode][b] = (
                base.evaluate_budget(
                    model=model,
                    loader=loader,
                    device=device,
                    budget=budget,
                    mode=mode,
                    fixed_head_order=fixed_head_order,
                    random_seed=(
                        random_seed
                        + mode_idx * 100003
                        + budget
                    ),
                )
            )

    return results


def should_run_diagnostic(
    epoch,
    epochs,
    diagnostic_every,
):
    if epoch == epochs:
        return True

    if diagnostic_every <= 0:
        return False

    # 첫 epoch에서 한번 확인하고,
    # 이후 N epoch마다 확인.
    return (
        epoch == 1
        or epoch % diagnostic_every == 0
    )


# ============================================================
# Reporting
# ============================================================

def fmt_head_freq(values):
    return " ".join(
        f"H{i}:{v:5.1f}%"
        for i, v in enumerate(values)
    )


def print_full_epoch_report(
    epoch,
    epochs,
    tau,
    lr,
    train_stats,
    learned_results,
    diagnostic_results,
    elapsed,
):
    print()
    print("=" * 112)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | "
        f"tau={tau:.4f} | "
        f"lr={lr:.3e} | "
        f"time={elapsed:.1f}s"
    )

    print(
        f"Train | "
        f"loss={train_stats['total_loss']:.4f} "
        f"task={train_stats['task_loss']:.4f} "
        f"div={train_stats['div_loss']:.4f} "
        f"acc={train_stats['accuracy']:.2f}% "
        f"allocator_grad="
        f"{train_stats['allocator_grad_norm']:.6e}"
    )

    print(
        "Budget batches:",
        train_stats["budget_batches"],
    )

    for budget in sorted(
        int(x)
        for x in learned_results.keys()
    ):
        b = str(budget)
        learned = learned_results[b]

        print()
        print(f"B={budget}")

        print(
            f"  learned "
            f"acc={learned['accuracy']:6.2f}% "
            f"loss={learned['loss']:.4f} | "
            f"mini={learned['mini_context_norm']:.2f} "
            f"main={learned['main_out_norm']:.2f} "
            f"attn={learned['attn_out_norm']:.2f} "
            f"main/mini="
            f"{learned['main_to_mini_norm_ratio']:.3f}"
        )

        print(
            "  learned active:",
            fmt_head_freq(
                learned[
                    "active_head_freq_pct"
                ]
            ),
        )

        print(
            f"  allocator: "
            f"std={learned['alloc_logits_std']:.4f} "
            f"entropy="
            f"{learned['allocation_entropy_norm']:.4f} "
            f"patterns="
            f"{learned['unique_active_patterns']}"
        )

        if (
            diagnostic_results is not None
            and budget > 0
        ):
            fixed = diagnostic_results[
                "fixed"
            ][b]
            random_r = diagnostic_results[
                "random"
            ][b]

            print(
                f"  fixed   "
                f"acc={fixed['accuracy']:6.2f}% "
                f"loss={fixed['loss']:.4f}"
            )

            print(
                f"  random  "
                f"acc={random_r['accuracy']:6.2f}% "
                f"loss={random_r['loss']:.4f}"
            )

            print(
                f"  delta: "
                f"learned-fixed="
                f"{learned['accuracy'] - fixed['accuracy']:+.2f}pp | "
                f"learned-random="
                f"{learned['accuracy'] - random_r['accuracy']:+.2f}pp"
            )

    monotonic = base.budget_monotonic_score(
        learned_results
    )

    learned_avg = sum(
        r["accuracy"]
        for r in learned_results.values()
    ) / len(learned_results)

    print()
    print(
        f"Learned average budget accuracy: "
        f"{learned_avg:.2f}%"
    )
    print(
        f"Budget monotonic score: "
        f"{monotonic:.2f}"
    )

    if "1" in learned_results:
        if base.detect_head_collapse(
            learned_results["1"]
        ):
            print(
                "WARNING: B=1 head selection "
                "looks collapsed (>90% one head)."
            )

    if diagnostic_results is None:
        print(
            "Diagnostic fixed/random: skipped this epoch"
        )
    else:
        print(
            "Diagnostic fixed/random: full test completed"
        )


# ============================================================
# Checkpoint / resume
# ============================================================

def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    lr_scheduler,
    tau,
    learned_results,
    diagnostic_results,
    learned_avg_acc,
    best_avg_acc,
    args,
):
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": lr_scheduler.state_dict(),
        "tau": tau,
        "eval": {
            "learned": learned_results,
            "diagnostic": diagnostic_results,
        },
        "learned_average_budget_accuracy": (
            learned_avg_acc
        ),
        "best_average_budget_accuracy": (
            best_avg_acc
        ),
        "args": vars(args),
    }

    torch.save(
        checkpoint,
        path,
    )


def load_resume(
    resume_path,
    model,
    optimizer,
    lr_scheduler,
    device,
):
    checkpoint = torch.load(
        resume_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer"]
    )

    lr_scheduler.load_state_dict(
        checkpoint["scheduler"]
    )

    start_epoch = int(
        checkpoint.get("epoch", 0)
    )

    best_avg_acc = float(
        checkpoint.get(
            "best_average_budget_accuracy",
            checkpoint.get(
                "learned_average_budget_accuracy",
                -1.0,
            ),
        )
    )

    return (
        checkpoint,
        start_epoch,
        best_avg_acc,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = build_parser().parse_args()
    validate_args(args)

    # base.train_one_epoch() 내부에서
    # base.args.grad_clip을 사용하므로 동일 args를 연결한다.
    base.args = args

    base.set_seed(args.seed)

    device = base.resolve_device(
        args.device
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    budgets = list(
        range(args.main_heads + 1)
    )

    fixed_head_order = (
        base.parse_fixed_head_order(
            args.fixed_head_order,
            args.main_heads,
        )
    )

    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print()
    print(
        "CIFAR-10 strong/full config:"
    )
    print(
        f"  train=50000 | test=10000 | "
        f"epochs={args.epochs} | "
        f"batch={args.batch_size}"
    )
    print(
        f"  img={args.img_size}, "
        f"patch={args.patch_size}, "
        f"embed={args.embed_dim}, "
        f"depth={args.depth}"
    )
    print(
        f"  main_heads={args.main_heads}, "
        f"mini_heads={args.mini_heads}, "
        f"budgets={budgets}"
    )
    print(
        f"  lambda_div={args.lambda_div} | "
        f"tau={args.tau_start}->{args.tau_end}"
    )
    print(
        f"  fixed order={fixed_head_order}"
    )
    print(
        f"  fixed/random diagnostic every "
        f"{args.diagnostic_every} epochs "
        f"(+ epoch 1 and final)"
    )

    train_loader, test_loader = (
        base.build_loaders(args)
    )

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)}"
    )
    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    model = MiniGuidedViT(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=10,
        embed_dim=args.embed_dim,
        depth=args.depth,
        main_heads=args.main_heads,
        mlp_ratio=args.mlp_ratio,
        mini_heads=args.mini_heads,
        mini_dim=args.mini_dim,
        pool_ratio=args.pool_ratio,
        direct_ratio=args.direct_ratio,
        alpha_direct=args.alpha_direct,
        alpha_mixed=args.alpha_mixed,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        allocator_hidden_dim=128,
        gumbel_tau=args.tau_start,
        use_gumbel=True,
    ).to(device)

    diversity_criterion = (
        HeadDiversityLoss(
            mode="direct_mixed"
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    lr_scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )
    )

    metrics_path = (
        output_dir / "metrics.jsonl"
    )

    start_epoch = 0
    best_avg_acc = -1.0

    if args.resume.strip():
        resume_path = Path(
            args.resume
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                resume_path
            )

        (
            _,
            start_epoch,
            best_avg_acc,
        ) = load_resume(
            resume_path=resume_path,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
        )

        print()
        print(
            f"Resumed from: {resume_path}"
        )
        print(
            f"Completed epochs: {start_epoch}"
        )
        print(
            f"Previous best average accuracy: "
            f"{best_avg_acc:.2f}%"
        )

        if start_epoch >= args.epochs:
            print(
                "Checkpoint already reached requested "
                "number of epochs. Nothing to train."
            )
            return

    # 실험 설정 기록
    config_path = (
        output_dir / "run_config.json"
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            indent=2,
            ensure_ascii=False,
        )

    for epoch_idx in range(
        start_epoch,
        args.epochs,
    ):
        epoch = epoch_idx + 1
        start_time = time.time()

        tau = base.gumbel_temperature(
            epoch_idx=epoch_idx,
            epochs=args.epochs,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
        )

        model.set_gumbel_temperature(
            tau
        )

        # ----------------------------------------------------
        # Train: identical training path to sanity
        # ----------------------------------------------------
        train_stats = base.train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            diversity_criterion=diversity_criterion,
            device=device,
            budgets=budgets,
            lambda_div=args.lambda_div,
        )

        # ----------------------------------------------------
        # Learned B0-B3: every epoch
        # ----------------------------------------------------
        learned_results = (
            evaluate_learned_all(
                model=model,
                loader=test_loader,
                device=device,
                budgets=budgets,
                fixed_head_order=fixed_head_order,
                random_seed=(
                    args.random_eval_seed
                ),
            )
        )

        # ----------------------------------------------------
        # Fixed/random: diagnostic epochs only
        # ----------------------------------------------------
        diagnostic_results = None

        if should_run_diagnostic(
            epoch=epoch,
            epochs=args.epochs,
            diagnostic_every=(
                args.diagnostic_every
            ),
        ):
            diagnostic_results = (
                evaluate_fixed_random(
                    model=model,
                    loader=test_loader,
                    device=device,
                    budgets=budgets,
                    fixed_head_order=(
                        fixed_head_order
                    ),
                    random_seed=(
                        args.random_eval_seed
                    ),
                    learned_results=(
                        learned_results
                    ),
                )
            )

        elapsed = (
            time.time() - start_time
        )

        lr = optimizer.param_groups[0][
            "lr"
        ]

        learned_avg_acc = sum(
            r["accuracy"]
            for r in learned_results.values()
        ) / len(learned_results)

        monotonic_score = (
            base.budget_monotonic_score(
                learned_results
            )
        )

        print_full_epoch_report(
            epoch=epoch,
            epochs=args.epochs,
            tau=tau,
            lr=lr,
            train_stats=train_stats,
            learned_results=(
                learned_results
            ),
            diagnostic_results=(
                diagnostic_results
            ),
            elapsed=elapsed,
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------
        record = {
            "epoch": epoch,
            "tau": tau,
            "lr": lr,
            "elapsed_sec": elapsed,
            "train": train_stats,
            "eval": {
                "learned": (
                    learned_results
                ),
                "diagnostic": (
                    diagnostic_results
                ),
            },
            "learned_average_budget_accuracy": (
                learned_avg_acc
            ),
            "budget_monotonic_score": (
                monotonic_score
            ),
        }

        with metrics_path.open(
            "a",
            encoding="utf-8",
        ) as f:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

        # ----------------------------------------------------
        # Prepare LR state for the next epoch BEFORE saving.
        # This keeps --resume aligned with uninterrupted training.
        # ----------------------------------------------------
        lr_scheduler.step()

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------
        improved = (
            learned_avg_acc
            > best_avg_acc
        )

        if improved:
            best_avg_acc = (
                learned_avg_acc
            )

        save_checkpoint(
            path=output_dir / "last.pt",
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tau=tau,
            learned_results=(
                learned_results
            ),
            diagnostic_results=(
                diagnostic_results
            ),
            learned_avg_acc=(
                learned_avg_acc
            ),
            best_avg_acc=best_avg_acc,
            args=args,
        )

        if improved:
            save_checkpoint(
                path=output_dir / "best.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                tau=tau,
                learned_results=(
                    learned_results
                ),
                diagnostic_results=(
                    diagnostic_results
                ),
                learned_avg_acc=(
                    learned_avg_acc
                ),
                best_avg_acc=best_avg_acc,
                args=args,
            )

            print(
                f"New best checkpoint: "
                f"avg budget acc="
                f"{best_avg_acc:.2f}%"
            )

    print()
    print("=" * 112)
    print(
        "CIFAR-10 depth=6 / 50-epoch "
        "strong-full experiment finished."
    )
    print(
        f"Best learned average budget accuracy: "
        f"{best_avg_acc:.2f}%"
    )
    print(
        f"Metrics: {metrics_path}"
    )
    print(
        f"Last checkpoint: "
        f"{output_dir / 'last.pt'}"
    )
    print(
        f"Best checkpoint: "
        f"{output_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()