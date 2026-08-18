# models/mini_mixer.py

from typing import Dict

import torch
import torch.nn as nn


class MiniMixer(nn.Module):
    """
    Dynamic Remaining-Mini Mixer.

    목적
    ----
    Direct로 선택되지 않은 Mini Head들의 정보를 버리지 않고
    입력별 utility를 기반으로 weighted summary를 만든다.

    Input
    -----
    mini_contexts:
        [B, Hm, N, Dh]

    utility_logits:
        [B, Hm]

    remaining_mask:
        [B, Hm] bool

        True:
            Direct로 선택되지 않았으며 Mix에 참여하는 Mini Head

        False:
            Direct로 이미 보존된 Mini Head


    Output
    ------
    mixed_context:
        [B, N, Dh]

        Remaining Mini Head들의 weighted sum.

    mix_weights:
        [B, Hm]

        Direct Head의 weight는 정확히 0.
        Remaining Head들의 weight 합은 1.
    """

    def __init__(
        self,
        mini_heads: int,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()

        if mini_heads <= 0:
            raise ValueError(
                f"mini_heads must be > 0, got {mini_heads}"
            )

        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {temperature}"
            )

        self.mini_heads = mini_heads
        self.temperature = temperature
        self.eps = eps

    def set_temperature(
        self,
        temperature: float,
    ) -> None:
        """
        필요하면 학습 중 Mix temperature를 변경한다.

        temperature가 작을수록:
            특정 Remaining Mini Head에 더 집중

        temperature가 클수록:
            Remaining Mini들을 더 균등하게 Mix
        """

        if temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0, got {temperature}"
            )

        self.temperature = float(temperature)

    def forward(
        self,
        mini_contexts: torch.Tensor,
        utility_logits: torch.Tensor,
        remaining_mask: torch.Tensor,
        return_info: bool = False,
    ):
        """
        Returns
        -------
        mixed_context:
            [B, N, Dh]

        return_info=True:
            mixed_context,
            info
        """

        # =========================================================
        # 1. Shape 검사
        # =========================================================

        if mini_contexts.dim() != 4:
            raise ValueError(
                "Expected mini_contexts shape [B, Hm, N, Dh], "
                f"got {mini_contexts.shape}"
            )

        if utility_logits.dim() != 2:
            raise ValueError(
                "Expected utility_logits shape [B, Hm], "
                f"got {utility_logits.shape}"
            )

        if remaining_mask.dim() != 2:
            raise ValueError(
                "Expected remaining_mask shape [B, Hm], "
                f"got {remaining_mask.shape}"
            )

        B, Hm, N, Dh = mini_contexts.shape

        if Hm != self.mini_heads:
            raise ValueError(
                f"Expected mini_heads={self.mini_heads}, "
                f"got {Hm}"
            )

        if utility_logits.shape != (B, Hm):
            raise ValueError(
                "utility_logits shape mismatch. "
                f"Expected {(B, Hm)}, "
                f"got {utility_logits.shape}"
            )

        if remaining_mask.shape != (B, Hm):
            raise ValueError(
                "remaining_mask shape mismatch. "
                f"Expected {(B, Hm)}, "
                f"got {remaining_mask.shape}"
            )

        if remaining_mask.dtype != torch.bool:
            raise TypeError(
                "remaining_mask must be torch.bool."
            )

        # =========================================================
        # 2. 각 sample의 Remaining Mini Head 개수
        # =========================================================

        remaining_count = (
            remaining_mask
            .sum(dim=-1)
        )
        # [B]

        # =========================================================
        # 3. Remaining Head에 대해서만 softmax
        # =========================================================

        scaled_logits = (
            utility_logits
            / self.temperature
        )

        # Direct Head는 softmax에 들어가지 못하도록
        # 매우 작은 값(-inf)으로 masking한다.
        masked_logits = scaled_logits.masked_fill(
            ~remaining_mask,
            float("-inf"),
        )

        # =========================================================
        # 4. 모든 Head가 Direct인 예외 처리
        #
        # direct_k == mini_heads이면
        # Remaining Head가 하나도 없다.
        #
        # 이 경우 mixed_context는 zero로 둔다.
        # =========================================================

        has_remaining = (
            remaining_count > 0
        )
        # [B]

        # softmax를 바로 하면 모든 값이 -inf인 row에서
        # NaN이 발생하므로 안전한 값을 넣는다.
        safe_logits = torch.where(
            has_remaining[:, None],
            masked_logits,
            torch.zeros_like(masked_logits),
        )

        mix_weights = torch.softmax(
            safe_logits,
            dim=-1,
        )

        # =========================================================
        # 5. Direct Head weight를 정확히 0으로 만든다.
        # =========================================================

        mix_weights = (
            mix_weights
            * remaining_mask.to(
                dtype=utility_logits.dtype
            )
        )

        # =========================================================
        # 6. 다시 normalization
        #
        # numerical safety.
        #
        # Remaining이 있다면 sum=1.
        # Remaining이 없다면 전부 0.
        # =========================================================

        weight_sum = mix_weights.sum(
            dim=-1,
            keepdim=True,
        )

        mix_weights = torch.where(
            has_remaining[:, None],
            mix_weights
            / (
                weight_sum
                + self.eps
            ),
            torch.zeros_like(mix_weights),
        )

        # =========================================================
        # 7. Mini Context weighted mixing
        #
        # mini_contexts:
        # [B, Hm, N, Dh]
        #
        # mix_weights:
        # [B, Hm]
        #
        # ->
        #
        # [B, Hm, 1, 1]
        # =========================================================

        weighted_contexts = (
            mini_contexts
            * mix_weights[
                :,
                :,
                None,
                None,
            ]
        )

        # =========================================================
        # Head dimension을 합친다.
        #
        # [B, Hm, N, Dh]
        #
        # ->
        #
        # [B, N, Dh]
        # =========================================================

        mixed_context = (
            weighted_contexts
            .sum(dim=1)
        )

        # =========================================================
        # 8. Logging / Debug info
        # =========================================================

        if return_info:

            info: Dict[str, torch.Tensor] = {

                "mix_weights":
                    mix_weights,

                "remaining_count":
                    remaining_count,

                "mix_weight_sum":
                    mix_weights
                    .sum(dim=-1),

                "has_remaining":
                    has_remaining,

            }

            return (
                mixed_context,
                info,
            )

        return mixed_context