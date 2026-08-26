import argparse
from collections import Counter
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--teacher-cache",
        type=str,
        default="/content/drive/MyDrive/mini-to-main-attention/checkpoints/stage2_counterfactual_teacher_cache.pt",
    )
    p.add_argument("--direct-k", type=int, default=2)
    return p.parse_args()


def canonicalize(x):
    return x.sort(dim=-1).values


def exact_match(a, b):
    a = canonicalize(a)
    b = canonicalize(b)
    return (a == b).all(dim=-1).float()


def overlap(a, b):
    matches = a[..., :, None] == b[..., None, :]
    return matches.any(dim=-1).float().mean(dim=-1)


def subset_to_combo_indices(subset, combo_table):
    subset = canonicalize(subset)
    combo_table = canonicalize(combo_table)

    # subset: [B, depth, K]
    # combo_table: [C, K]
    equality = (
        subset[:, :, None, :]
        ==
        combo_table[None, None, :, :]
    ).all(dim=-1)

    if not equality.any(dim=-1).all():
        raise RuntimeError("Subset missing from combination table.")

    return equality.float().argmax(dim=-1)


def selected_regret(selected_pair, subset_losses, combo_table):
    combo_idx = subset_to_combo_indices(
        selected_pair,
        combo_table,
    )

    selected_loss = (
        subset_losses.gather(
            dim=-1,
            index=combo_idx[..., None],
        )
        .squeeze(-1)
    )

    oracle_loss = subset_losses.min(dim=-1).values
    return selected_loss - oracle_loss


def print_pair_frequency(title, pairs, combinations):
    print(f"\n{title}")

    depth = pairs.shape[1]

    for block_idx in range(depth):
        block_pairs = canonicalize(
            pairs[:, block_idx, :]
        ).tolist()

        counter = Counter(
            tuple(int(v) for v in pair)
            for pair in block_pairs
        )

        total = sum(counter.values())

        print(f"\nBlock {block_idx}:")

        for combo in combinations:
            combo = tuple(int(v) for v in combo)
            count = counter.get(combo, 0)

            print(
                f"  {combo}: "
                f"{count:4d} "
                f"({100.0 * count / total:6.2f}%)"
            )


def diagnostic_split(name, data, direct_k):
    teacher_target = data["teacher_target"].float()
    oracle = data["oracle_best_subset"].long()
    subset_losses = data["subset_losses"].float()
    combo_table = data["combination_table"].long()

    teacher_pair = torch.topk(
        teacher_target,
        k=direct_k,
        dim=-1,
    ).indices

    teacher_exact = exact_match(
        teacher_pair,
        oracle,
    )

    teacher_overlap = overlap(
        teacher_pair,
        oracle,
    )

    teacher_regret = selected_regret(
        teacher_pair,
        subset_losses,
        combo_table,
    )

    spread = (
        subset_losses.max(dim=-1).values
        -
        subset_losses.min(dim=-1).values
    )

    print(f"\n================ {name.upper()} ================")
    print(
        f"Teacher Top-K vs Oracle exact: "
        f"{100.0 * teacher_exact.mean().item():.2f}%"
    )
    print(
        f"Teacher Top-K vs Oracle overlap: "
        f"{100.0 * teacher_overlap.mean().item():.2f}%"
    )
    print(
        f"Teacher mean oracle regret: "
        f"{teacher_regret.mean().item():.8e}"
    )
    print(
        f"Teacher median oracle regret: "
        f"{teacher_regret.median().item():.8e}"
    )
    print(
        f"Mean subset spread: "
        f"{spread.mean().item():.8e}"
    )

    return {
        "teacher_target": teacher_target,
        "oracle": oracle,
        "subset_losses": subset_losses,
        "combo_table": combo_table,
        "teacher_pair": teacher_pair,
        "teacher_exact": teacher_exact,
        "teacher_overlap": teacher_overlap,
        "teacher_regret": teacher_regret,
        "spread": spread,
    }


def print_spread_quartiles(result):
    spread = result["spread"].reshape(-1)
    exact = result["teacher_exact"].reshape(-1)
    regret = result["teacher_regret"].reshape(-1)

    q = torch.quantile(
        spread,
        torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0]),
    )

    print("\nTeacher quality by subset-loss-spread quartile")

    for i in range(4):
        lo = q[i]
        hi = q[i + 1]

        if i < 3:
            mask = (spread >= lo) & (spread < hi)
        else:
            mask = (spread >= lo) & (spread <= hi)

        if not mask.any():
            continue

        print(
            f"Q{i + 1} "
            f"[{lo.item():.3e}, {hi.item():.3e}] | "
            f"n={int(mask.sum())} | "
            f"exact={100.0 * exact[mask].mean().item():.2f}% | "
            f"regret={regret[mask].mean().item():.3e}"
        )


def evaluate_global_prior(train_result, val_result, direct_k):
    train_target = train_result["teacher_target"]
    val_target = val_result["teacher_target"]
    val_oracle = val_result["oracle"]
    val_subset_losses = val_result["subset_losses"]
    combo_table = val_result["combo_table"]

    prior_score = train_target.mean(dim=0)  # [depth, H]

    prior_pair_per_block = torch.topk(
        prior_score,
        k=direct_k,
        dim=-1,
    ).indices

    B = val_target.shape[0]

    prior_pair = (
        prior_pair_per_block[None, :, :]
        .expand(B, -1, -1)
        .clone()
    )

    prior_top1 = (
        prior_score.argmax(dim=-1)[None, :]
        .expand(B, -1)
    )

    teacher_top1 = val_target.argmax(dim=-1)

    top1_agreement = (
        prior_top1 == teacher_top1
    ).float().mean()

    prior_exact = exact_match(
        prior_pair,
        val_oracle,
    )

    prior_overlap = overlap(
        prior_pair,
        val_oracle,
    )

    prior_regret = selected_regret(
        prior_pair,
        val_subset_losses,
        combo_table,
    )

    print("\n================ GLOBAL PRIOR BASELINE ================")
    print("Fixed pair learned only from mean train teacher distribution:")
    print(prior_pair_per_block)
    print(
        f"Top-1 teacher agreement: "
        f"{100.0 * top1_agreement.item():.2f}%"
    )
    print(
        f"Pair vs Oracle exact: "
        f"{100.0 * prior_exact.mean().item():.2f}%"
    )
    print(
        f"Pair vs Oracle overlap: "
        f"{100.0 * prior_overlap.mean().item():.2f}%"
    )
    print(
        f"Mean oracle regret: "
        f"{prior_regret.mean().item():.8e}"
    )

    return prior_pair


def main():
    args = parse_args()

    cache = torch.load(
        args.teacher_cache,
        map_location="cpu",
        weights_only=False,
    )

    train = diagnostic_split(
        "utility-train teacher",
        cache["train"],
        args.direct_k,
    )

    val = diagnostic_split(
        "utility-val teacher",
        cache["val"],
        args.direct_k,
    )

    print_spread_quartiles(val)

    evaluate_global_prior(
        train,
        val,
        args.direct_k,
    )

    combinations = val["combo_table"].tolist()

    print_pair_frequency(
        "Oracle pair frequency on utility-val",
        val["oracle"],
        combinations,
    )

    print_pair_frequency(
        "Teacher Top-K pair frequency on utility-val",
        val["teacher_pair"],
        combinations,
    )

    print("\n================ RANDOM REFERENCE ================")

    num_combinations = val["combo_table"].shape[0]
    random_exact = 1.0 / num_combinations

    H = int(
        val["combo_table"].max().item() + 1
    )

    K = args.direct_k
    random_overlap = K / H

    print(
        f"Random exact-pair expectation: "
        f"{100.0 * random_exact:.2f}%"
    )
    print(
        f"Random Top-{K} overlap expectation: "
        f"{100.0 * random_overlap:.2f}%"
    )

    print("\nDiagnostic completed.")


if __name__ == "__main__":
    main()