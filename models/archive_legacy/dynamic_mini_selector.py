# models/dynamic_mini_selector.py

from typing import Dict

import torch
import torch.nn as nn


class DynamicMiniSelector(nn.Module):
    """
    Input-dependent Mini Head Selector.

    목적
    ----
    Utility Predictor가 생성한 Mini Head별 utility score를 사용해서
    입력마다 Direct로 보존할 Mini Head를 선택한다.

    현재 v1에서는:
        - utility가 높은 Top-K Mini Head -> Direct
        - 나머지 Mini Head -> Remaining / Mix 대상

    Input
    -----
    utility_logits:
        [B, Hm]

        예:
            sample 0:
                [0.8, 0.2, 0.7, 0.1]

            sample 1:
                [0.1, 0.9, 0.2, 0.8]


    Output
    ------
    direct_mask:
        [B, Hm] bool

    remaining_mask:
        [B, Hm] bool

    direct_indices:
        [B, direct_k]

    direct_scores:
        [B, direct_k]


    중요
    ----
    이 selector는 Main Head를 선택하는 모듈이 아니다.

    여기서는 오직:

        "어떤 Mini Head 정보를 Direct로 보존할 것인가?"

    만 결정한다.
    """

    def __init__(
        self,
        mini_heads: int,
        direct_k: int,
    ):
        super().__init__()

        if mini_heads <= 0:
            raise ValueError(
                f"mini_heads must be > 0, got {mini_heads}"
            )

        if direct_k < 0:
            raise ValueError(
                f"direct_k must be >= 0, got {direct_k}"
            )

        if direct_k > mini_heads:
            raise ValueError(
                "direct_k cannot exceed mini_heads. "
                f"direct_k={direct_k}, "
                f"mini_heads={mini_heads}"
            )

        self.mini_heads = mini_heads
        self.direct_k = direct_k

    def forward(
        self,
        utility_logits: torch.Tensor,
        return_info: bool = False,
    ):
        """
        Args
        ----
        utility_logits:
            [B, Hm]

        Returns
        -------
        direct_mask:
            [B, Hm]

        remaining_mask:
            [B, Hm]

        return_info=True이면 추가 정보도 반환.
        """

        # =========================================================
        # 1. 입력 shape 확인
        # =========================================================

        if utility_logits.dim() != 2:
            raise ValueError(
                "Expected utility_logits shape [B, Hm], "
                f"got {utility_logits.shape}"
            )

        B, Hm = utility_logits.shape

        if Hm != self.mini_heads:
            raise ValueError(
                f"Expected mini_heads={self.mini_heads}, "
                f"got {Hm}"
            )

        # =========================================================
        # 2. direct_k == 0
        #
        # 모든 Mini가 Mix 대상으로 간다.
        # =========================================================

        if self.direct_k == 0:

            direct_mask = torch.zeros(
                B,
                Hm,
                dtype=torch.bool,
                device=utility_logits.device,
            )

            remaining_mask = torch.ones(
                B,
                Hm,
                dtype=torch.bool,
                device=utility_logits.device,
            )

            direct_indices = torch.empty(
                B,
                0,
                dtype=torch.long,
                device=utility_logits.device,
            )

            direct_scores = torch.empty(
                B,
                0,
                dtype=utility_logits.dtype,
                device=utility_logits.device,
            )

        # =========================================================
        # 3. Top-K Mini Head 선택
        # =========================================================

        else:

            topk = torch.topk(
                utility_logits,
                k=self.direct_k,
                dim=-1,
                largest=True,
                sorted=True,
            )

            # [B, direct_k]
            direct_indices = topk.indices

            # [B, direct_k]
            direct_scores = topk.values

            # =====================================================
            # Direct mask
            #
            # 예:
            #
            # selected = [0, 2]
            #
            # ->
            #
            # [True, False, True, False]
            # =====================================================

            direct_mask = torch.zeros(
                B,
                Hm,
                dtype=torch.bool,
                device=utility_logits.device,
            )

            direct_mask.scatter_(
                dim=1,
                index=direct_indices,
                value=True,
            )

            # =====================================================
            # Direct가 아닌 Mini는 모두 Mix 후보
            # =====================================================

            remaining_mask = ~direct_mask

        # =========================================================
        # 4. Invariant 확인
        # =========================================================

        direct_count = (
            direct_mask
            .sum(dim=-1)
        )

        expected_count = torch.full(
            (B,),
            self.direct_k,
            dtype=direct_count.dtype,
            device=direct_count.device,
        )

        if not torch.equal(
            direct_count,
            expected_count,
        ):
            raise RuntimeError(
                "Direct mask does not satisfy direct_k."
            )

        if torch.any(
            direct_mask & remaining_mask
        ):
            raise RuntimeError(
                "direct_mask and remaining_mask overlap."
            )

        if not torch.all(
            direct_mask | remaining_mask
        ):
            raise RuntimeError(
                "Some Mini Heads belong to neither "
                "Direct nor Remaining."
            )

        # =========================================================
        # 5. 반환
        # =========================================================

        if return_info:

            info: Dict[str, torch.Tensor] = {

                "direct_indices":
                    direct_indices,

                "direct_scores":
                    direct_scores,

                "direct_count":
                    direct_count,

                "remaining_count":
                    remaining_mask
                    .sum(dim=-1),

            }

            return (
                direct_mask,
                remaining_mask,
                info,
            )

        return (
            direct_mask,
            remaining_mask,
        )