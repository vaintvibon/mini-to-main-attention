# train_cifar10_short.py
"""
CIFAR-10 short sanity training + allocation diagnostic evaluation.

핵심 목적
---------
1. balanced budget training으로 B=0/1/2/3을 균등 노출한다.
2. Gumbel-ST allocator gradient가 실제 CIFAR-10에서도 유지되는지 본다.
3. 매 epoch 동일 checkpoint를 다음 세 allocation mode로 평가한다.
   - learned: MiniToMainAllocator의 deterministic Top-K
   - fixed:   입력과 무관한 고정 head order
   - random:  입력과 무관한 재현 가능한 random head order
4. budget/mode별 accuracy와 representation norm을 함께 측정한다.
   - mini_context_norm
   - main_out_norm
   - attn_out_norm: TwoLevelMiniMainAttention 최종 output, residual 더하기 전
5. learned가 fixed/random보다 유리한지, 그리고 budget이 늘면서
   Main magnitude가 비정상적으로 커지는지 진단한다.

주의
----
- 이 비교는 "같은 learned-allocation 방식으로 학습된 checkpoint"에서
  evaluation scheduler만 fixed/random으로 바꾸는 inference-time ablation이다.
  논문 최종 baseline으로 쓰려면 fixed/random 방식으로 각각 별도 학습한 실험도 필요하다.
- 현재 attention v1은 모든 Main head를 dense 계산한 뒤 gate를 적용한다.
  따라서 FLOPs/latency 절감 주장은 아직 할 수 없다.
"""

import argparse
import json
import math
import random
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from losses.diversity_loss import HeadDiversityLoss
from models.mini_guided_vit import MiniGuidedViT


# ---------------------------------------------------------------------
# Reproducibility / utility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def limited_loader(loader: DataLoader, max_batches: int):
    """max_batches <= 0이면 loader 전체를 사용."""
    if max_batches <= 0:
        yield from loader
        return

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        yield batch


def global_grad_norm_from_named_parameters(
    named_parameters: Iterable,
    name_contains: Optional[str] = None,
) -> float:
    squared_sum = 0.0
    found_grad = False

    for name, p in named_parameters:
        if name_contains is not None and name_contains not in name:
            continue
        if p.grad is None:
            continue

        grad = p.grad.detach()

        if not torch.isfinite(grad).all():
            raise RuntimeError(
                f"Non-finite gradient detected in parameter: {name}"
            )

        squared_sum += grad.float().pow(2).sum().item()
        found_grad = True

    if not found_grad:
        return 0.0

    return math.sqrt(squared_sum)


def build_balanced_budget_schedule(
    num_batches: int,
    budgets: List[int],
) -> List[int]:
    """
    한 epoch에서 budget 노출 횟수를 최대한 균등하게 한다.

    32 batches, [0,1,2,3]
      -> 각 budget 8회

    10 batches
      -> 각 budget 2~3회

    순서는 매 epoch shuffle한다.
    """
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")

    if not budgets:
        raise ValueError("budgets must not be empty.")

    repeats = math.ceil(num_batches / len(budgets))
    schedule = (budgets * repeats)[:num_batches]
    random.shuffle(schedule)

    return schedule


def gumbel_temperature(
    epoch_idx: int,
    epochs: int,
    tau_start: float,
    tau_end: float,
) -> float:
    """Exponential temperature annealing."""
    if epochs <= 1:
        return float(tau_end)

    if tau_start <= 0.0 or tau_end <= 0.0:
        raise ValueError("Gumbel temperatures must be > 0.")

    ratio = epoch_idx / float(epochs - 1)

    return float(
        tau_start * ((tau_end / tau_start) ** ratio)
    )


def parse_fixed_head_order(
    text: str,
    main_heads: int,
) -> List[int]:
    """
    ""이면 [0,1,...,H-1].
    예: --fixed-head-order 2,0,1
    """
    if text.strip() == "":
        order = list(range(main_heads))
    else:
        order = [
            int(v.strip())
            for v in text.split(",")
            if v.strip() != ""
        ]

    if sorted(order) != list(range(main_heads)):
        raise ValueError(
            "--fixed-head-order must be a permutation of "
            f"0..{main_heads - 1}. Got {order}."
        )

    return order


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------

def build_cifar10_loaders(args):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(
                (args.img_size, args.img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (args.img_size, args.img_size)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            ),
        ]
    )

    train_set = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_set = datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    generator = torch.Generator().manual_seed(
        args.seed
    )

    if (
        args.train_subset > 0
        and args.train_subset < len(train_set)
    ):
        indices = torch.randperm(
            len(train_set),
            generator=generator,
        )[: args.train_subset].tolist()

        train_set = Subset(
            train_set,
            indices,
        )

    if (
        args.test_subset > 0
        and args.test_subset < len(test_set)
    ):
        indices = torch.randperm(
            len(test_set),
            generator=generator,
        )[: args.test_subset].tolist()

        test_set = Subset(
            test_set,
            indices,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

def compute_diversity_loss(
    info_list: List[Dict[str, torch.Tensor]],
    diversity_criterion: HeadDiversityLoss,
) -> torch.Tensor:
    if not info_list:
        raise ValueError("info_list is empty.")

    losses = []

    for info in info_list:
        losses.append(
            diversity_criterion(
                head_out=info["head_out"],
                active_mask=info["active_mask"],
                direct_mask=info["direct_mask"],
                mixed_mask=info["mixed_mask"],
            )
        )

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_epoch(
    model: MiniGuidedViT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    diversity_criterion: HeadDiversityLoss,
    device: torch.device,
    budgets: List[int],
    lambda_div: float,
    max_batches: int,
):
    model.train()

    running_task_loss = 0.0
    running_div_loss = 0.0
    running_total_loss = 0.0
    running_correct = 0
    running_samples = 0

    allocator_grad_sum = 0.0
    allocator_grad_steps = 0

    budget_counter = Counter()

    total_loader_batches = len(loader)

    target_batches = (
        total_loader_batches
        if max_batches <= 0
        else min(max_batches, total_loader_batches)
    )

    budget_schedule = build_balanced_budget_schedule(
        num_batches=target_batches,
        budgets=budgets,
    )

    num_batches = 0

    for batch_idx, (images, targets) in enumerate(
        limited_loader(loader, max_batches)
    ):
        images = images.to(
            device,
            non_blocking=True,
        )
        targets = targets.to(
            device,
            non_blocking=True,
        )

        budget = budget_schedule[batch_idx]
        budget_counter[budget] += 1

        optimizer.zero_grad(set_to_none=True)

        logits, info_list = model(
            images,
            budget=budget,
            return_info=True,
        )

        task_loss = F.cross_entropy(
            logits,
            targets,
        )

        div_loss = compute_diversity_loss(
            info_list=info_list,
            diversity_criterion=diversity_criterion,
        )

        loss = (
            task_loss
            + lambda_div * div_loss
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss. "
                f"budget={budget}, "
                f"task={task_loss.item()}, "
                f"div={div_loss.item()}"
            )

        loss.backward()

        allocator_grad_norm = (
            global_grad_norm_from_named_parameters(
                model.named_parameters(),
                name_contains="allocator",
            )
        )

        # B=0은 allocator decision이 output에 영향을 주지 않으므로
        # allocator grad가 0이어도 정상.
        if budget > 0:
            allocator_grad_sum += allocator_grad_norm
            allocator_grad_steps += 1

            if allocator_grad_norm <= 0.0:
                raise RuntimeError(
                    "Allocator received zero gradient "
                    "on a budget>0 training step."
                )

        optimizer.step()

        batch_size = targets.shape[0]

        running_task_loss += (
            task_loss.item() * batch_size
        )
        running_div_loss += (
            div_loss.item() * batch_size
        )
        running_total_loss += (
            loss.item() * batch_size
        )
        running_correct += (
            logits.argmax(dim=-1)
            == targets
        ).sum().item()

        running_samples += batch_size
        num_batches += 1

    if running_samples == 0:
        raise RuntimeError(
            "No training samples were processed."
        )

    return {
        "task_loss": (
            running_task_loss
            / running_samples
        ),
        "div_loss": (
            running_div_loss
            / running_samples
        ),
        "total_loss": (
            running_total_loss
            / running_samples
        ),
        "train_acc": (
            100.0
            * running_correct
            / running_samples
        ),
        "allocator_grad_norm": (
            allocator_grad_sum
            / allocator_grad_steps
            if allocator_grad_steps > 0
            else 0.0
        ),
        "budget_batches": {
            str(b): int(budget_counter[b])
            for b in budgets
        },
        "num_batches": num_batches,
    }


# ---------------------------------------------------------------------
# Evaluation scheduler override
# ---------------------------------------------------------------------

def _build_override_schedule(
    scheduler,
    alloc_logits: torch.Tensor,
    budget: int,
    order_idx: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    fixed/random baseline용 hard schedule.

    order_idx:
        [B, H], 앞쪽부터 우선 선택.

    eval 전용이므로 ST surrogate는 필요 없다.
    gate == exact hard mask(float).
    """
    if alloc_logits.dim() != 2:
        raise ValueError(
            "alloc_logits must be [B, H]."
        )

    B, H = alloc_logits.shape

    if H != scheduler.main_heads:
        raise ValueError(
            "Head count mismatch in override."
        )

    if budget < 0 or budget > H:
        raise ValueError(
            f"Invalid budget={budget}, H={H}."
        )

    direct_count = (
        scheduler._get_direct_count(budget)
    )

    active_mask = torch.zeros(
        B,
        H,
        dtype=torch.bool,
        device=alloc_logits.device,
    )

    direct_mask = torch.zeros_like(
        active_mask
    )

    if budget > 0:
        active_idx = order_idx[:, :budget]
        active_mask.scatter_(
            1,
            active_idx,
            True,
        )

    if direct_count > 0:
        direct_idx = order_idx[
            :,
            :direct_count,
        ]
        direct_mask.scatter_(
            1,
            direct_idx,
            True,
        )

    mixed_mask = (
        active_mask
        & (~direct_mask)
    )
    inactive_mask = ~active_mask

    if torch.any(
        direct_mask & inactive_mask
    ):
        raise RuntimeError(
            "Override produced invalid "
            "direct/inactive overlap."
        )

    active_gate = active_mask.to(
        dtype=alloc_logits.dtype
    )
    direct_gate = direct_mask.to(
        dtype=alloc_logits.dtype
    )
    mixed_gate = mixed_mask.to(
        dtype=alloc_logits.dtype
    )
    inactive_gate = inactive_mask.to(
        dtype=alloc_logits.dtype
    )

    stats = {
        "budget": torch.tensor(
            float(budget),
            device=alloc_logits.device,
            dtype=alloc_logits.dtype,
        ),
        "direct_count": torch.tensor(
            float(direct_count),
            device=alloc_logits.device,
            dtype=alloc_logits.dtype,
        ),
        "active_count_mean": (
            active_mask.float()
            .sum(dim=1)
            .mean()
            .detach()
        ),
        "direct_count_mean": (
            direct_mask.float()
            .sum(dim=1)
            .mean()
            .detach()
        ),
        "mixed_count_mean": (
            mixed_mask.float()
            .sum(dim=1)
            .mean()
            .detach()
        ),
        "inactive_count_mean": (
            inactive_mask.float()
            .sum(dim=1)
            .mean()
            .detach()
        ),
        "gumbel_tau": torch.tensor(
            float(scheduler.gumbel_tau),
            device=alloc_logits.device,
            dtype=alloc_logits.dtype,
        ),
    }

    # selection_scores는 baseline 선택 ranking을 표현한다.
    # allocator statistics는 별도 alloc_logits에서 계속 계산한다.
    rank_scores = torch.empty_like(
        alloc_logits
    )

    rank_value = torch.arange(
        H,
        0,
        -1,
        dtype=alloc_logits.dtype,
        device=alloc_logits.device,
    ).unsqueeze(0).expand(B, -1)

    rank_scores.scatter_(
        1,
        order_idx,
        rank_value,
    )

    return {
        "active_mask": active_mask,
        "direct_mask": direct_mask,
        "mixed_mask": mixed_mask,
        "inactive_mask": inactive_mask,
        "active_gate": active_gate,
        "direct_gate": direct_gate,
        "mixed_gate": mixed_gate,
        "inactive_gate": inactive_gate,
        "selection_scores": rank_scores,
        "gumbel_noise": torch.zeros_like(
            alloc_logits
        ),
        "stats": stats,
    }


@contextmanager
def temporary_allocation_mode(
    model: MiniGuidedViT,
    mode: str,
    fixed_head_order: Sequence[int],
    random_seed: int,
):
    """
    evaluation 동안 각 block scheduler만 임시 교체한다.

    learned:
        원래 scheduler 사용.

    fixed:
        모든 sample에서 fixed_head_order 사용.
        예: [0,1,2]
          B=1 -> H0
          B=2 -> H0,H1

    random:
        sample/block별 random permutation.
        Mini/allocator output은 선택에 사용하지 않는다.
        random_seed로 재현 가능.

    context 종료 후 원래 scheduler.forward를 복원한다.
    """
    if mode == "learned":
        yield
        return

    if mode not in {"fixed", "random"}:
        raise ValueError(
            f"Unknown allocation mode: {mode}"
        )

    original_forwards = []

    for block_idx, block in enumerate(
        model.blocks
    ):
        scheduler = block.attn.scheduler

        original_forwards.append(
            (scheduler, scheduler.forward)
        )

        if mode == "random":
            # CPU generator 사용 후 index를 target device로 이동.
            # CPU/CUDA 환경에서 동일하게 재현 가능하다.
            generator = (
                torch.Generator(device="cpu")
                .manual_seed(
                    random_seed
                    + block_idx * 100003
                )
            )
        else:
            generator = None

        fixed_order_tensor = torch.tensor(
            list(fixed_head_order),
            dtype=torch.long,
        )

        def override_forward(
            self_scheduler,
            alloc_logits,
            budget,
            *,
            _mode=mode,
            _generator=generator,
            _fixed_order=fixed_order_tensor,
        ):
            B, H = alloc_logits.shape

            if _mode == "fixed":
                order_idx = (
                    _fixed_order
                    .to(
                        device=alloc_logits.device
                    )
                    .unsqueeze(0)
                    .expand(B, -1)
                )
            else:
                # 각 sample마다 independent random ranking.
                random_scores_cpu = torch.rand(
                    B,
                    H,
                    generator=_generator,
                    device="cpu",
                )

                order_idx = (
                    random_scores_cpu
                    .argsort(
                        dim=-1,
                        descending=True,
                    )
                    .to(
                        device=alloc_logits.device
                    )
                )

            return _build_override_schedule(
                scheduler=self_scheduler,
                alloc_logits=alloc_logits,
                budget=budget,
                order_idx=order_idx,
            )

        scheduler.forward = MethodType(
            override_forward,
            scheduler,
        )

    try:
        yield
    finally:
        for scheduler, original_forward in (
            original_forwards
        ):
            scheduler.forward = (
                original_forward
            )


# ---------------------------------------------------------------------
# Evaluation statistics
# ---------------------------------------------------------------------

class AllocationStats:
    def __init__(self, main_heads: int):
        self.main_heads = main_heads

        self.active_count = torch.zeros(
            main_heads,
            dtype=torch.float64,
        )
        self.direct_count = torch.zeros(
            main_heads,
            dtype=torch.float64,
        )
        self.mixed_count = torch.zeros(
            main_heads,
            dtype=torch.float64,
        )

        self.total_block_samples = 0

        self.logit_sum = 0.0
        self.logit_sq_sum = 0.0
        self.logit_numel = 0

        self.entropy_sum = 0.0
        self.entropy_count = 0

        self.active_patterns = Counter()
        self.per_block_active = {}

    @torch.no_grad()
    def update(
        self,
        info_list: List[
            Dict[str, torch.Tensor]
        ],
    ):
        for block_idx, info in enumerate(
            info_list
        ):
            active = (
                info["active_mask"]
                .detach()
                .cpu()
            )
            direct = (
                info["direct_mask"]
                .detach()
                .cpu()
            )
            mixed = (
                info["mixed_mask"]
                .detach()
                .cpu()
            )
            logits = (
                info["alloc_logits"]
                .detach()
                .float()
                .cpu()
            )

            batch_size = active.shape[0]

            self.active_count += (
                active.float()
                .sum(dim=0)
                .double()
            )
            self.direct_count += (
                direct.float()
                .sum(dim=0)
                .double()
            )
            self.mixed_count += (
                mixed.float()
                .sum(dim=0)
                .double()
            )

            self.total_block_samples += (
                batch_size
            )

            if block_idx not in (
                self.per_block_active
            ):
                self.per_block_active[
                    block_idx
                ] = torch.zeros(
                    self.main_heads,
                    dtype=torch.float64,
                )

            self.per_block_active[
                block_idx
            ] += (
                active.float()
                .sum(dim=0)
                .double()
            )

            self.logit_sum += (
                logits.sum().item()
            )
            self.logit_sq_sum += (
                (logits ** 2)
                .sum()
                .item()
            )
            self.logit_numel += logits.numel()

            probs = logits.softmax(dim=-1)

            entropy = -(
                probs
                * (probs + 1e-8).log()
            ).sum(dim=-1)

            if self.main_heads > 1:
                entropy = (
                    entropy
                    / math.log(self.main_heads)
                )

            self.entropy_sum += (
                entropy.sum().item()
            )
            self.entropy_count += (
                entropy.numel()
            )

            for row in active.tolist():
                key = "".join(
                    "1" if bool(v) else "0"
                    for v in row
                )
                self.active_patterns[key] += 1

    def summary(self):
        denom = max(
            1,
            self.total_block_samples,
        )

        active_freq = (
            100.0
            * self.active_count
            / denom
        ).tolist()

        direct_freq = (
            100.0
            * self.direct_count
            / denom
        ).tolist()

        mixed_freq = (
            100.0
            * self.mixed_count
            / denom
        ).tolist()

        if self.logit_numel > 0:
            mean = (
                self.logit_sum
                / self.logit_numel
            )

            var = max(
                0.0,
                (
                    self.logit_sq_sum
                    / self.logit_numel
                    - mean * mean
                ),
            )

            std = math.sqrt(var)
        else:
            mean = 0.0
            std = 0.0

        entropy = (
            self.entropy_sum
            / self.entropy_count
            if self.entropy_count > 0
            else 0.0
        )

        per_block_active_freq = {}

        num_blocks = max(
            1,
            len(self.per_block_active),
        )

        block_denom = max(
            1.0,
            self.total_block_samples
            / num_blocks,
        )

        for block_idx, counts in sorted(
            self.per_block_active.items()
        ):
            per_block_active_freq[
                str(block_idx)
            ] = (
                100.0
                * counts
                / block_denom
            ).tolist()

        return {
            "active_head_freq_pct": active_freq,
            "direct_head_freq_pct": direct_freq,
            "mixed_head_freq_pct": mixed_freq,
            "alloc_logits_mean": mean,
            "alloc_logits_std": std,
            "allocation_entropy_norm": entropy,
            "unique_active_patterns": len(
                self.active_patterns
            ),
            "active_pattern_top5": (
                self.active_patterns
                .most_common(5)
            ),
            "per_block_active_head_freq_pct": (
                per_block_active_freq
            ),
        }


class RepresentationNormStats:
    """
    각 tensor를 sample별 Frobenius L2 norm으로 계산한 뒤
    모든 block/sample에 대해 평균한다.

    예:
        x: [B, N, D]
        norm_b = ||x_b||_F

    출력:
        mini_context_norm
        main_out_norm
        attn_out_norm

    attn_out:
        TwoLevelMiniMainAttention의 최종 output.
        즉 block residual x + attn_out 이전의 attention branch 출력.
    """

    def __init__(self):
        self.mini_sum = 0.0
        self.main_sum = 0.0
        self.info_count = 0

        self.attn_sum = 0.0
        self.attn_count = 0

    @staticmethod
    def _sample_frobenius_norm(
        x: torch.Tensor,
    ) -> torch.Tensor:
        return (
            x.detach()
            .float()
            .flatten(start_dim=1)
            .norm(p=2, dim=1)
        )

    @torch.no_grad()
    def update_info(
        self,
        info_list: List[
            Dict[str, torch.Tensor]
        ],
    ):
        for info in info_list:
            mini_norm = (
                self._sample_frobenius_norm(
                    info["mini_context"]
                )
            )

            main_norm = (
                self._sample_frobenius_norm(
                    info["main_out"]
                )
            )

            self.mini_sum += (
                mini_norm.sum().item()
            )
            self.main_sum += (
                main_norm.sum().item()
            )

            self.info_count += (
                mini_norm.numel()
            )

    @torch.no_grad()
    def update_attn_output(
        self,
        attn_out: torch.Tensor,
    ):
        norms = self._sample_frobenius_norm(
            attn_out
        )

        self.attn_sum += norms.sum().item()
        self.attn_count += norms.numel()

    def summary(self):
        mini = (
            self.mini_sum
            / self.info_count
            if self.info_count > 0
            else 0.0
        )

        main = (
            self.main_sum
            / self.info_count
            if self.info_count > 0
            else 0.0
        )

        attn = (
            self.attn_sum
            / self.attn_count
            if self.attn_count > 0
            else 0.0
        )

        return {
            "mini_context_norm": mini,
            "main_out_norm": main,
            "attn_out_norm": attn,
            "main_to_mini_norm_ratio": (
                main / mini
                if mini > 0.0
                else 0.0
            ),
            "attn_to_mini_norm_ratio": (
                attn / mini
                if mini > 0.0
                else 0.0
            ),
        }


class AttentionOutputHookGroup:
    """
    각 block.attn의 최종 output norm을 수집한다.
    models/*.py를 수정하지 않기 위해 evaluation 시 hook 사용.
    """

    def __init__(
        self,
        model: MiniGuidedViT,
        norm_stats: RepresentationNormStats,
    ):
        self.handles = []

        for block in model.blocks:
            handle = (
                block.attn.register_forward_hook(
                    self._make_hook(norm_stats)
                )
            )
            self.handles.append(handle)

    @staticmethod
    def _make_hook(
        norm_stats: RepresentationNormStats,
    ):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                attn_out = output[0]
            else:
                attn_out = output

            norm_stats.update_attn_output(
                attn_out
            )

        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        self.remove()


@torch.no_grad()
def evaluate_budget(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
    budget: int,
    max_batches: int,
    allocation_mode: str,
    fixed_head_order: Sequence[int],
    random_seed: int,
):
    """
    learned/fixed/random 중 하나의 allocation mode로
    특정 budget을 평가한다.
    """
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    allocation_stats = AllocationStats(
        main_heads=model.main_heads
    )
    norm_stats = RepresentationNormStats()

    with temporary_allocation_mode(
        model=model,
        mode=allocation_mode,
        fixed_head_order=fixed_head_order,
        random_seed=random_seed,
    ):
        with AttentionOutputHookGroup(
            model=model,
            norm_stats=norm_stats,
        ):
            for images, targets in limited_loader(
                loader,
                max_batches,
            ):
                images = images.to(
                    device,
                    non_blocking=True,
                )
                targets = targets.to(
                    device,
                    non_blocking=True,
                )

                logits, info_list = model(
                    images,
                    budget=budget,
                    return_info=True,
                )

                loss = F.cross_entropy(
                    logits,
                    targets,
                    reduction="sum",
                )

                correct += (
                    logits.argmax(dim=-1)
                    == targets
                ).sum().item()

                total += targets.shape[0]
                loss_sum += loss.item()

                allocation_stats.update(
                    info_list
                )
                norm_stats.update_info(
                    info_list
                )

    if total == 0:
        raise RuntimeError(
            "No evaluation samples were processed."
        )

    result = {
        "allocation_mode": allocation_mode,
        "budget": int(budget),
        "loss": loss_sum / total,
        "accuracy": (
            100.0 * correct / total
        ),
    }

    result.update(
        allocation_stats.summary()
    )
    result.update(
        norm_stats.summary()
    )

    return result


@torch.no_grad()
def evaluate_all_modes(
    model: MiniGuidedViT,
    loader: DataLoader,
    device: torch.device,
    budgets: List[int],
    max_batches: int,
    fixed_head_order: Sequence[int],
    random_eval_seed: int,
):
    """
    결과 구조:
    {
      "learned": {"0": ..., "1": ..., ...},
      "fixed":   {...},
      "random":  {...}
    }

    B=0은 scheduler 선택이 output에 영향을 주지 않으므로
    learned 결과를 fixed/random에도 복사해 evaluation 비용을 줄인다.
    """
    results = {
        "learned": {},
        "fixed": {},
        "random": {},
    }

    # learned
    for budget in budgets:
        results["learned"][str(budget)] = (
            evaluate_budget(
                model=model,
                loader=loader,
                device=device,
                budget=budget,
                max_batches=max_batches,
                allocation_mode="learned",
                fixed_head_order=fixed_head_order,
                random_seed=(
                    random_eval_seed
                    + budget * 1009
                ),
            )
        )

    # fixed/random
    for mode_idx, mode in enumerate(
        ["fixed", "random"]
    ):
        for budget in budgets:
            if budget == 0:
                copied = dict(
                    results["learned"]["0"]
                )
                copied["allocation_mode"] = mode
                copied["copied_from_learned"] = True

                results[mode]["0"] = copied
                continue

            results[mode][str(budget)] = (
                evaluate_budget(
                    model=model,
                    loader=loader,
                    device=device,
                    budget=budget,
                    max_batches=max_batches,
                    allocation_mode=mode,
                    fixed_head_order=fixed_head_order,
                    random_seed=(
                        random_eval_seed
                        + mode_idx * 1000003
                        + budget * 1009
                    ),
                )
            )

    return results


# ---------------------------------------------------------------------
# Printing / checkpoint
# ---------------------------------------------------------------------

def fmt_head_freq(values):
    return " ".join(
        f"H{i}:{v:5.1f}%"
        for i, v in enumerate(values)
    )


def print_mode_budget_line(
    mode: str,
    result: Dict,
):
    print(
        f"    {mode:<7} | "
        f"acc={result['accuracy']:5.2f}% "
        f"loss={result['loss']:.4f} | "
        f"mini={result['mini_context_norm']:.3f} "
        f"main={result['main_out_norm']:.3f} "
        f"attn={result['attn_out_norm']:.3f} | "
        f"main/mini="
        f"{result['main_to_mini_norm_ratio']:.3f}"
    )


def print_epoch_report(
    epoch: int,
    epochs: int,
    tau: float,
    lr: float,
    train_stats: Dict,
    eval_stats: Dict,
    elapsed_sec: float,
):
    print()
    print("=" * 104)

    print(
        f"Epoch {epoch:02d}/{epochs:02d} | "
        f"tau={tau:.4f} | "
        f"lr={lr:.3e} | "
        f"time={elapsed_sec:.1f}s"
    )

    print(
        f"Train | "
        f"total={train_stats['total_loss']:.4f} "
        f"task={train_stats['task_loss']:.4f} "
        f"div={train_stats['div_loss']:.4f} "
        f"acc={train_stats['train_acc']:.2f}% "
        f"allocator_grad="
        f"{train_stats['allocator_grad_norm']:.6e}"
    )

    print(
        "Train budget batches: "
        f"{train_stats['budget_batches']}"
    )

    budgets = list(
        eval_stats["learned"].keys()
    )

    for budget_str in budgets:
        print()
        print(
            f"B={budget_str}"
        )

        for mode in [
            "learned",
            "fixed",
            "random",
        ]:
            print_mode_budget_line(
                mode,
                eval_stats[mode][budget_str],
            )

        learned = (
            eval_stats["learned"][
                budget_str
            ]
        )

        print(
            "      learned active : "
            + fmt_head_freq(
                learned[
                    "active_head_freq_pct"
                ]
            )
        )

        print(
            "      learned direct : "
            + fmt_head_freq(
                learned[
                    "direct_head_freq_pct"
                ]
            )
        )

        print(
            "      learned mixed  : "
            + fmt_head_freq(
                learned[
                    "mixed_head_freq_pct"
                ]
            )
        )

        print(
            "      allocator      : "
            f"logit_std="
            f"{learned['alloc_logits_std']:.4f} "
            f"entropy="
            f"{learned['allocation_entropy_norm']:.4f} "
            f"patterns="
            f"{learned['unique_active_patterns']}"
        )

        if int(budget_str) > 0:
            learned_acc = (
                eval_stats["learned"][
                    budget_str
                ]["accuracy"]
            )
            fixed_acc = (
                eval_stats["fixed"][
                    budget_str
                ]["accuracy"]
            )
            random_acc = (
                eval_stats["random"][
                    budget_str
                ]["accuracy"]
            )

            print(
                "      delta          : "
                f"learned-fixed="
                f"{learned_acc - fixed_acc:+.2f}pp "
                f"learned-random="
                f"{learned_acc - random_acc:+.2f}pp"
            )


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    lr_scheduler,
    epoch,
    tau,
    eval_stats,
    args,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": (
                lr_scheduler.state_dict()
            ),
            "gumbel_tau": tau,
            "eval_stats": eval_stats,
            "args": vars(args),
        },
        path,
    )


# ---------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------

def build_argparser():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/cifar10_short_compare",
    )
    parser.add_argument(
        "--train-subset",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--test-subset",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    # run length
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=0,
        help="0이면 train subset 전체.",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="0이면 test subset 전체.",
    )

    # model
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--main-heads",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--mini-heads",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--mini-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--pool-ratio",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--mlp-ratio",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--direct-ratio",
        type=float,
        default=0.34,
    )
    parser.add_argument(
        "--alpha-direct",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--alpha-mixed",
        type=float,
        default=0.2,
    )

    # optimization
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--lambda-div",
        type=float,
        default=0.01,
    )

    # Gumbel-ST
    parser.add_argument(
        "--tau-start",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--tau-end",
        type=float,
        default=0.5,
    )

    # diagnostic baselines
    parser.add_argument(
        "--fixed-head-order",
        type=str,
        default="",
        help=(
            "fixed baseline head priority. "
            "Example: '0,1,2'. "
            "Empty -> natural order."
        ),
    )
    parser.add_argument(
        "--random-eval-seed",
        type=int,
        default=2026,
    )

    # misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', 'cuda', 'cuda:0'",
    )

    return parser


def validate_args(args):
    if args.main_heads <= 0:
        raise ValueError(
            "--main-heads must be positive."
        )

    if (
        args.embed_dim
        % args.main_heads
        != 0
    ):
        raise ValueError(
            "--embed-dim must be divisible "
            "by --main-heads."
        )

    if (
        args.img_size
        % args.patch_size
        != 0
    ):
        raise ValueError(
            "--img-size must be divisible "
            "by --patch-size."
        )

    patch_side = (
        args.img_size
        // args.patch_size
    )

    if patch_side < args.pool_ratio:
        raise ValueError(
            "Patch grid is smaller "
            "than pool_ratio."
        )

    if args.lambda_div < 0:
        raise ValueError(
            "--lambda-div must be >= 0."
        )

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be > 0."
        )


def main():
    args = (
        build_argparser()
        .parse_args()
    )

    validate_args(args)
    set_seed(args.seed)

    device = resolve_device(
        args.device
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fixed_head_order = (
        parse_fixed_head_order(
            args.fixed_head_order,
            args.main_heads,
        )
    )

    print("Device:", device)
    print("PyTorch:", torch.__version__)
    print()

    print("Model config:")
    print(
        f"  img={args.img_size}, "
        f"patch={args.patch_size}, "
        f"embed={args.embed_dim}, "
        f"depth={args.depth}"
    )
    print(
        f"  main_heads={args.main_heads}, "
        f"mini_heads={args.mini_heads}, "
        f"mini_dim={args.mini_dim}, "
        f"pool_ratio={args.pool_ratio}"
    )

    budgets = list(
        range(args.main_heads + 1)
    )

    print("Budgets:", budgets)
    print(
        "Budget sampling: "
        "balanced per epoch "
        "(order shuffled)"
    )
    print(
        "Eval allocation modes: "
        "learned / fixed / random"
    )
    print(
        "Fixed head order:",
        fixed_head_order,
    )
    print(
        "Random eval seed:",
        args.random_eval_seed,
    )

    train_loader, test_loader = (
        build_cifar10_loaders(args)
    )

    print(
        f"Train samples: "
        f"{len(train_loader.dataset)} | "
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
            T_max=max(1, args.epochs),
        )
    )

    metrics_path = (
        output_dir
        / "metrics_compare.jsonl"
    )

    best_learned_avg_acc = -1.0

    for epoch_idx in range(
        args.epochs
    ):
        epoch = epoch_idx + 1
        start_time = time.time()

        tau = gumbel_temperature(
            epoch_idx=epoch_idx,
            epochs=args.epochs,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
        )

        model.set_gumbel_temperature(
            tau
        )

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            diversity_criterion=(
                diversity_criterion
            ),
            device=device,
            budgets=budgets,
            lambda_div=args.lambda_div,
            max_batches=(
                args.max_train_batches
            ),
        )

        eval_stats = evaluate_all_modes(
            model=model,
            loader=test_loader,
            device=device,
            budgets=budgets,
            max_batches=(
                args.max_eval_batches
            ),
            fixed_head_order=(
                fixed_head_order
            ),
            random_eval_seed=(
                args.random_eval_seed
            ),
        )

        lr = optimizer.param_groups[0][
            "lr"
        ]

        elapsed = (
            time.time()
            - start_time
        )

        print_epoch_report(
            epoch=epoch,
            epochs=args.epochs,
            tau=tau,
            lr=lr,
            train_stats=train_stats,
            eval_stats=eval_stats,
            elapsed_sec=elapsed,
        )

        learned_avg_acc = (
            sum(
                result["accuracy"]
                for result
                in eval_stats[
                    "learned"
                ].values()
            )
            / len(
                eval_stats["learned"]
            )
        )

        fixed_avg_acc = (
            sum(
                result["accuracy"]
                for result
                in eval_stats[
                    "fixed"
                ].values()
            )
            / len(
                eval_stats["fixed"]
            )
        )

        random_avg_acc = (
            sum(
                result["accuracy"]
                for result
                in eval_stats[
                    "random"
                ].values()
            )
            / len(
                eval_stats["random"]
            )
        )

        record = {
            "epoch": epoch,
            "tau": tau,
            "lr": lr,
            "elapsed_sec": elapsed,
            "train": train_stats,
            "eval": eval_stats,
            "average_budget_accuracy": {
                "learned": learned_avg_acc,
                "fixed": fixed_avg_acc,
                "random": random_avg_acc,
            },
            "fixed_head_order": (
                fixed_head_order
            ),
            "random_eval_seed": (
                args.random_eval_seed
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

        save_checkpoint(
            path=(
                output_dir
                / "last.pt"
            ),
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            tau=tau,
            eval_stats=eval_stats,
            args=args,
        )

        if (
            learned_avg_acc
            > best_learned_avg_acc
        ):
            best_learned_avg_acc = (
                learned_avg_acc
            )

            save_checkpoint(
                path=(
                    output_dir
                    / "best_learned_avg.pt"
                ),
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                tau=tau,
                eval_stats=eval_stats,
                args=args,
            )

        lr_scheduler.step()

    print()
    print("=" * 104)
    print(
        "Short diagnostic training finished."
    )
    print(
        "Best learned average budget "
        f"accuracy: "
        f"{best_learned_avg_acc:.2f}%"
    )
    print(
        "Metrics:",
        metrics_path,
    )
    print(
        "Checkpoint:",
        output_dir
        / "best_learned_avg.pt",
    )
    print()
    print(
        "Interpretation: "
        "learned > fixed/random인지와 "
        "budget 증가 시 main_out norm이 "
        "비정상적으로 커지는지를 함께 확인."
    )


if __name__ == "__main__":
    main()