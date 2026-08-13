import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class _HardForwardSoftBackward(torch.autograd.Function):
    """
    Forward에서는 hard gate를 *정확히* 반환하고,
    backward에서는 soft surrogate 쪽으로 upstream gradient를 전달한다.

    이렇게 하면 hard + soft - soft.detach()에서 생길 수 있는
    미세한 floating-point residue 없이 inactive head가 정확히 0을 유지한다.
    """

    @staticmethod
    def forward(ctx, hard: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # hard에는 gradient를 보내지 않고, soft surrogate에 동일한 upstream gradient를 전달.
        return None, grad_output



class TwoLevelHeadScheduler(nn.Module):
    """
    Two-level head scheduler with Gumbel + Straight-Through relaxed Top-K.

    핵심 목표
    ---------
    1. Forward에서는 정확히 budget개의 Main head만 hard-selected 된 것처럼 동작한다.
    2. Backward에서는 relaxed Top-K surrogate를 통해 alloc_logits까지 gradient가 흐른다.
    3. direct head는 active head의 subset이라는 invariant를 유지한다.
    4. evaluation에서는 Gumbel noise 없이 deterministic hard Top-K를 사용한다.

    출력
    ----
    hard mask (bool):
        active_mask, direct_mask, mixed_mask, inactive_mask
        - 디버깅 / 로깅 / diversity pair 선택에 사용

    differentiable gate (float):
        active_gate, direct_gate, mixed_gate
        - 실제 attention 계산에 사용해야 allocator까지 gradient가 연결됨
        - forward value는 hard 0/1과 동일
        - backward gradient는 relaxed surrogate에서 옴
    """

    def __init__(
        self,
        main_heads: int,
        direct_ratio: float = 0.34,
        gumbel_tau: float = 1.0,
        use_gumbel: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()

        if main_heads <= 0:
            raise ValueError(f"main_heads must be positive. Got {main_heads}.")

        if not (0.0 <= direct_ratio <= 1.0):
            raise ValueError(
                f"direct_ratio must be in [0, 1]. Got {direct_ratio}."
            )

        if gumbel_tau <= 0.0:
            raise ValueError(f"gumbel_tau must be > 0. Got {gumbel_tau}.")

        self.main_heads = main_heads
        self.direct_ratio = direct_ratio
        self.gumbel_tau = gumbel_tau
        self.use_gumbel = use_gumbel
        self.eps = eps

    def set_temperature(self, tau: float) -> None:
        """학습 중 Gumbel temperature annealing에 사용."""
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0. Got {tau}.")
        self.gumbel_tau = float(tau)

    def _get_direct_count(self, budget: int) -> int:
        if budget == 0:
            return 0

        direct_count = int(round(budget * self.direct_ratio))
        direct_count = max(1, direct_count)
        direct_count = min(direct_count, budget)
        return direct_count

    def _sample_gumbel_like(self, x: torch.Tensor) -> torch.Tensor:
        """Sample standard Gumbel noise with same shape/device/dtype as x."""
        u = torch.rand_like(x).clamp_(self.eps, 1.0 - self.eps)
        return -torch.log(-torch.log(u))

    def _relaxed_topk(
        self,
        scores: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        """
        Differentiable relaxed k-hot approximation.

        동일한 scores에서 k번 연속적으로 relaxed one-hot을 뽑아 합산한다.
        k=R과 k=K를 같은 scores로 계산하면, R<=K일 때 direct surrogate가
        active surrogate의 앞부분에 해당하므로 mixed = active - direct가 자연스럽다.

        Args:
            scores: [B, H]
            k: number of relaxed selections

        Returns:
            relaxed_khot: [B, H], 각 row의 합은 근사적으로 k
        """
        if k == 0:
            return torch.zeros_like(scores)

        # 반복 선택 시 이미 선택된 항목의 확률을 억제하기 위한 누적값
        relaxed = torch.zeros_like(scores)

        for _ in range(k):
            # log(1 - relaxed)는 이미 많이 선택된 위치를 다음 iteration에서 억제한다.
            remaining = (1.0 - relaxed).clamp_min(self.eps)
            adjusted_scores = scores + torch.log(remaining)
            one_hot_soft = F.softmax(adjusted_scores / self.gumbel_tau, dim=-1)
            relaxed = relaxed + one_hot_soft

        return relaxed

    def _straight_through_topk(
        self,
        scores: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            hard_mask_bool: [B, H] bool
            hard_mask_float: [B, H] float
            st_gate: [B, H] float
                forward == hard_mask_float
                backward == relaxed_topk gradient
        """
        B, H = scores.shape

        if k == 0:
            hard_bool = torch.zeros(B, H, dtype=torch.bool, device=scores.device)
            hard_float = torch.zeros_like(scores)
            st_gate = hard_float
            return hard_bool, hard_float, st_gate

        topk_idx = torch.topk(
            scores,
            k=k,
            dim=-1,
            largest=True,
            sorted=True,
        ).indices

        hard_bool = torch.zeros(B, H, dtype=torch.bool, device=scores.device)
        hard_bool.scatter_(1, topk_idx, True)
        hard_float = hard_bool.to(dtype=scores.dtype)

        # Evaluation에서는 gradient가 필요 없으므로 완전한 hard gate를 반환한다.
        if not self.training:
            return hard_bool, hard_float, hard_float

        relaxed = self._relaxed_topk(scores, k=k)

        # Straight-through estimator:
        # forward: exact hard_float (0/1)
        # backward: relaxed surrogate의 gradient
        st_gate = _HardForwardSoftBackward.apply(hard_float, relaxed)

        return hard_bool, hard_float, st_gate

    def forward(
        self,
        alloc_logits: torch.Tensor,
        budget: int,
    ) -> Dict[str, torch.Tensor]:
        if alloc_logits.dim() != 2:
            raise ValueError(
                f"Expected alloc_logits shape [B, H], but got {alloc_logits.shape}."
            )

        B, H = alloc_logits.shape

        if H != self.main_heads:
            raise ValueError(
                f"Expected main_heads={self.main_heads}, but got H={H}."
            )

        if not isinstance(budget, int):
            raise TypeError(f"budget must be int. Got {type(budget)}.")

        if budget < 0:
            raise ValueError(f"budget must be >= 0. Got {budget}.")

        if budget > self.main_heads:
            raise ValueError(
                f"budget cannot exceed main_heads. "
                f"Got budget={budget}, main_heads={self.main_heads}."
            )

        direct_count = self._get_direct_count(budget)

        # 같은 perturbed score를 active/direct 모두 공유해야 direct ⊂ active가 유지된다.
        if self.training and self.use_gumbel and budget > 0:
            gumbel_noise = self._sample_gumbel_like(alloc_logits)
            selection_scores = alloc_logits + gumbel_noise
        else:
            gumbel_noise = torch.zeros_like(alloc_logits)
            selection_scores = alloc_logits

        # budget == H이면 active set은 고정(all heads)이므로
        # active 선택 자체에 surrogate gradient를 만들 이유가 없다.
        if budget == self.main_heads:
            active_mask = torch.ones(B, H, dtype=torch.bool, device=alloc_logits.device)
            active_hard = torch.ones_like(alloc_logits)
            active_gate = active_hard
        else:
            active_mask, active_hard, active_gate = self._straight_through_topk(
                selection_scores,
                k=budget,
            )

        # direct_count == budget이면 direct set == active set이므로 재선택하지 않는다.
        if direct_count == budget:
            direct_mask = active_mask.clone()
            direct_hard = active_hard
            direct_gate = active_gate
        else:
            direct_mask, direct_hard, direct_gate = self._straight_through_topk(
                selection_scores,
                k=direct_count,
            )

        # Hard masks: exact combinatorial semantics
        mixed_mask = active_mask & (~direct_mask)
        inactive_mask = ~active_mask

        # Surrogate gates:
        # 동일한 relaxed selection sequence를 쓰므로 active-direct는 mixed surrogate가 된다.
        mixed_gate = active_gate - direct_gate

        # inactive는 실제 forward 계산에 사용하지 않으므로 hard mask만 유지한다.
        inactive_gate = 1.0 - active_gate

        # Safety invariants
        if torch.any(direct_mask & inactive_mask):
            raise RuntimeError("Invalid masks: direct_mask overlaps with inactive_mask.")

        expected_active = torch.full(
            (B,), float(budget), device=alloc_logits.device, dtype=alloc_logits.dtype
        )
        hard_active_count = active_hard.sum(dim=1)
        if not torch.allclose(hard_active_count, expected_active):
            raise RuntimeError("Hard active gate does not satisfy exact budget.")

        expected_direct = torch.full(
            (B,), float(direct_count), device=alloc_logits.device, dtype=alloc_logits.dtype
        )
        hard_direct_count = direct_hard.sum(dim=1)
        if not torch.allclose(hard_direct_count, expected_direct):
            raise RuntimeError("Hard direct gate does not satisfy direct_count.")

        stats = {
            "budget": torch.tensor(
                float(budget), device=alloc_logits.device, dtype=alloc_logits.dtype
            ),
            "direct_count": torch.tensor(
                float(direct_count), device=alloc_logits.device, dtype=alloc_logits.dtype
            ),
            "active_count_mean": active_mask.float().sum(dim=1).mean().detach(),
            "direct_count_mean": direct_mask.float().sum(dim=1).mean().detach(),
            "mixed_count_mean": mixed_mask.float().sum(dim=1).mean().detach(),
            "inactive_count_mean": inactive_mask.float().sum(dim=1).mean().detach(),
            "gumbel_tau": torch.tensor(
                float(self.gumbel_tau),
                device=alloc_logits.device,
                dtype=alloc_logits.dtype,
            ),
        }

        return {
            # exact hard masks for logging / invariants / diversity pair construction
            "active_mask": active_mask,
            "direct_mask": direct_mask,
            "mixed_mask": mixed_mask,
            "inactive_mask": inactive_mask,

            # differentiable straight-through gates for actual computation
            "active_gate": active_gate,
            "direct_gate": direct_gate,
            "mixed_gate": mixed_gate,
            "inactive_gate": inactive_gate,

            "selection_scores": selection_scores,
            "gumbel_noise": gumbel_noise,
            "stats": stats,
        }
