# models/two_level_scheduler.py

import math
from typing import Dict

import torch
import torch.nn as nn


class TwoLevelHeadScheduler(nn.Module):
    """
    Two-level head scheduler.

    역할:
        MiniToMainAllocator가 만든 alloc_logits를 보고,
        주어진 Main head budget 안에서 head slot을 다음 세 종류로 나눈다.

        1. direct-bound head
            - Mini가 확신한 relation을 강하게 확장할 head
        2. mixed head
            - Mini context를 약하게 참고하면서 추가 relation을 학습할 head
        3. inactive head
            - 현재 budget에서는 사용하지 않을 head

    입력:
        alloc_logits: [B, H]
            B = batch size
            H = total main heads

        budget: int
            이번 forward에서 사용할 active main head 개수.
            예: DeiT-tiny scale이면 budget in [0, 1, 2, 3]

    출력:
        masks: Dict[str, Tensor]
            active_mask:   [B, H], bool
            direct_mask:   [B, H], bool
            mixed_mask:    [B, H], bool
            inactive_mask: [B, H], bool

        stats: Dict[str, Tensor]
            디버깅용 통계값
    """

    def __init__(
        self,
        main_heads: int,
        direct_ratio: float = 0.34,
    ):
        super().__init__()

        if main_heads <= 0:
            raise ValueError(f"main_heads must be positive. Got {main_heads}.")

        if not (0.0 <= direct_ratio <= 1.0):
            raise ValueError(
                f"direct_ratio must be in [0, 1]. Got {direct_ratio}."
            )

        self.main_heads = main_heads
        self.direct_ratio = direct_ratio

    def _get_direct_count(self, budget: int) -> int:
        """
        active head 중 몇 개를 direct-bound로 둘지 결정한다.

        예:
            main_heads=3, direct_ratio=0.34
            budget=0 -> direct=0
            budget=1 -> direct=1
            budget=2 -> direct=1
            budget=3 -> direct=1

            main_heads=12, direct_ratio=0.33
            budget=3  -> direct=1
            budget=6  -> direct=2
            budget=12 -> direct=4
        """
        if budget == 0:
            return 0

        direct_count = int(round(budget * self.direct_ratio))

        # budget이 0이 아니면 direct head를 최소 1개는 둔다.
        direct_count = max(1, direct_count)

        # direct head 수는 active head 수를 넘으면 안 된다.
        direct_count = min(direct_count, budget)

        return direct_count

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

        device = alloc_logits.device

        active_mask = torch.zeros(B, H, dtype=torch.bool, device=device)
        direct_mask = torch.zeros(B, H, dtype=torch.bool, device=device)

        if budget > 0:
            # active head 선택
            active_idx = torch.topk(
                alloc_logits,
                k=budget,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices  # [B, budget]

            active_mask.scatter_(dim=1, index=active_idx, value=True)

            # direct-bound head 선택
            direct_count = self._get_direct_count(budget)

            direct_idx = torch.topk(
                alloc_logits,
                k=direct_count,
                dim=-1,
                largest=True,
                sorted=True,
            ).indices  # [B, direct_count]

            direct_mask.scatter_(dim=1, index=direct_idx, value=True)

        mixed_mask = active_mask & (~direct_mask)
        inactive_mask = ~active_mask

        # 안전성 검사
        # direct는 반드시 active의 subset이어야 한다.
        if torch.any(direct_mask & inactive_mask):
            raise RuntimeError("Invalid masks: direct_mask overlaps with inactive_mask.")

        stats = {
            "budget": torch.tensor(float(budget), device=device),
            "active_count_mean": active_mask.float().sum(dim=1).mean().detach(),
            "direct_count_mean": direct_mask.float().sum(dim=1).mean().detach(),
            "mixed_count_mean": mixed_mask.float().sum(dim=1).mean().detach(),
            "inactive_count_mean": inactive_mask.float().sum(dim=1).mean().detach(),
        }

        return {
            "active_mask": active_mask,
            "direct_mask": direct_mask,
            "mixed_mask": mixed_mask,
            "inactive_mask": inactive_mask,
            "stats": stats,
        }