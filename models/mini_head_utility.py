# models/mini_head_utility.py

from typing import Dict

import torch
import torch.nn as nn


class MiniHeadUtility(nn.Module):
    """
    Input-dependent Mini Head Utility Predictor.

    목적
    ----
    각 Mini Head가 현재 입력에서 얼마나 유용할지를
    forward 정보만 사용하여 예측한다.

    Predictor 입력:
        1. Mini Head CLS context
        2. Mini Head patch mean context
        3. Attention entropy
        4. Max-attention confidence

    최종적으로:
        utility_logits: [B, Hm]

    를 출력한다.


    중요
    ----
    utility_logits는 그 자체로 정답이 아니다.

    이후 학습 단계에서 실제 task loss로 계산한
    First-Order Taylor head importance를 supervision target으로 사용하여

        predicted utility
            ≈
        Taylor importance

    가 되도록 학습할 예정이다.

    따라서 inference에서는 backward 없이
    이 Predictor만으로 Head importance를 추정할 수 있다.
    """

    def __init__(
        self,
        mini_head_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        has_cls_token: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()

        if mini_head_dim <= 0:
            raise ValueError(
                f"mini_head_dim must be > 0, got {mini_head_dim}"
            )

        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be > 0, got {hidden_dim}"
            )

        self.mini_head_dim = mini_head_dim
        self.hidden_dim = hidden_dim
        self.has_cls_token = has_cls_token
        self.eps = eps

        # =========================================================
        # Predictor input
        #
        # CLS context        : Dh
        # Patch mean context : Dh
        # Attention entropy  : 1
        # Max confidence     : 1
        #
        # total:
        # 2*Dh + 2
        # =========================================================

        feature_dim = (
            mini_head_dim * 2
            + 2
        )

        # 모든 Mini Head가 동일한 scorer를 공유한다.
        #
        # Head 번호 자체가 아니라
        # 현재 입력에서 Head가 만든 representation과
        # attention behavior를 평가하기 위함이다.

        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim),

            nn.Linear(
                feature_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def _summarize_context(
        self,
        mini_contexts: torch.Tensor,
    ):
        """
        Mini Head별 context representation을 요약한다.

        Input
        -----
        mini_contexts:
            [B, Hm, N, Dh]

        Returns
        -------
        cls_summary:
            [B, Hm, Dh]

        patch_summary:
            [B, Hm, Dh]
        """

        if mini_contexts.dim() != 4:
            raise ValueError(
                "Expected mini_contexts shape [B, Hm, N, Dh], "
                f"got {mini_contexts.shape}"
            )

        B, Hm, N, Dh = mini_contexts.shape

        if Dh != self.mini_head_dim:
            raise ValueError(
                f"Expected mini_head_dim={self.mini_head_dim}, "
                f"got Dh={Dh}"
            )

        if self.has_cls_token:

            if N < 2:
                raise ValueError(
                    "has_cls_token=True requires "
                    "CLS + at least one patch token."
                )

            # -----------------------------------------------------
            # CLS representation
            # -----------------------------------------------------

            cls_summary = mini_contexts[:, :, 0, :]
            # [B, Hm, Dh]

            # -----------------------------------------------------
            # Patch representation
            # -----------------------------------------------------

            patch_summary = (
                mini_contexts[:, :, 1:, :]
                .mean(dim=2)
            )
            # [B, Hm, Dh]

        else:

            # CLS가 없는 경우
            # mean + max pooling 사용

            cls_summary = (
                mini_contexts
                .max(dim=2)
                .values
            )

            patch_summary = (
                mini_contexts
                .mean(dim=2)
            )

        return (
            cls_summary,
            patch_summary,
        )

    def _attention_statistics(
        self,
        mini_attn: torch.Tensor,
    ):
        """
        Mini Head별 attention statistics를 계산한다.

        Input
        -----
        mini_attn:
            [B, Hm, N, M]

        Returns
        -------
        normalized_entropy:
            [B, Hm]

        max_confidence:
            [B, Hm]
        """

        if mini_attn.dim() != 4:
            raise ValueError(
                "Expected mini_attn shape [B, Hm, N, M], "
                f"got {mini_attn.shape}"
            )

        B, Hm, N, M = mini_attn.shape

        # =========================================================
        # 1. Attention Entropy
        # =========================================================

        # token별 entropy
        #
        # [B, Hm, N, M]
        #
        # ->
        #
        # [B, Hm, N]

        entropy = -(
            mini_attn
            * (
                mini_attn + self.eps
            ).log()
        ).sum(dim=-1)

        # ---------------------------------------------------------
        # log(M)으로 normalization
        #
        # entropy ≈ 0
        # -> 매우 집중
        #
        # entropy ≈ 1
        # -> 거의 uniform
        # ---------------------------------------------------------

        max_entropy = torch.log(
            torch.tensor(
                float(M),
                device=mini_attn.device,
                dtype=mini_attn.dtype,
            )
        )

        normalized_entropy = (
            entropy
            / (
                max_entropy
                + self.eps
            )
        )

        # query token들에 대해 평균
        #
        # [B, Hm, N]
        # ->
        # [B, Hm]

        normalized_entropy = (
            normalized_entropy
            .mean(dim=2)
        )

        # =========================================================
        # 2. Max Attention Confidence
        # =========================================================

        # query별 가장 높은 attention probability
        #
        # [B, Hm, N]

        max_prob = (
            mini_attn
            .max(dim=-1)
            .values
        )

        # token 평균
        #
        # [B, Hm]

        max_confidence = (
            max_prob
            .mean(dim=2)
        )

        return (
            normalized_entropy,
            max_confidence,
        )

    def forward(
        self,
        mini_contexts: torch.Tensor,
        mini_attn: torch.Tensor,
        return_info: bool = False,
    ):
        """
        Inputs
        ------
        mini_contexts:
            [B, Hm, N, Dh]

        mini_attn:
            [B, Hm, N, M]


        Returns
        -------
        utility_logits:
            [B, Hm]

        return_info=True:
            utility_logits,
            info
        """

        # =========================================================
        # Shape consistency
        # =========================================================

        if (
            mini_contexts.shape[0]
            != mini_attn.shape[0]
        ):
            raise ValueError(
                "Batch size mismatch between "
                "mini_contexts and mini_attn."
            )

        if (
            mini_contexts.shape[1]
            != mini_attn.shape[1]
        ):
            raise ValueError(
                "Mini head count mismatch between "
                "mini_contexts and mini_attn."
            )

        # =========================================================
        # 1. Representation features
        # =========================================================

        (
            cls_summary,
            patch_summary,
        ) = self._summarize_context(
            mini_contexts
        )

        # =========================================================
        # 2. Attention behavior features
        # =========================================================

        (
            attention_entropy,
            max_confidence,
        ) = self._attention_statistics(
            mini_attn
        )

        # [B,Hm]
        #
        # ->
        #
        # [B,Hm,1]

        attention_entropy_feature = (
            attention_entropy
            .unsqueeze(-1)
        )

        max_confidence_feature = (
            max_confidence
            .unsqueeze(-1)
        )

        # =========================================================
        # 3. Utility feature 생성
        # =========================================================

        # CLS                  [B,Hm,Dh]
        # Patch Mean           [B,Hm,Dh]
        # Entropy              [B,Hm,1]
        # Max Confidence       [B,Hm,1]
        #
        # ->
        #
        # [B,Hm,2Dh+2]

        utility_features = torch.cat(
            [
                cls_summary,
                patch_summary,
                attention_entropy_feature,
                max_confidence_feature,
            ],
            dim=-1,
        )

        # =========================================================
        # 4. Utility prediction
        # =========================================================

        # [B,Hm,2Dh+2]
        #
        # ->
        #
        # [B,Hm,1]

        utility_logits = self.scorer(
            utility_features
        )

        # ->
        #
        # [B,Hm]

        utility_logits = (
            utility_logits
            .squeeze(-1)
        )

        # =========================================================
        # 5. Probability
        #
        # Top-K selection에는 logits를 사용한다.
        #
        # probability는 logging / 분석 등에 사용한다.
        # =========================================================

        utility_probs = torch.softmax(
            utility_logits,
            dim=-1,
        )

        if return_info:

            info: Dict[str, torch.Tensor] = {

                "cls_summary":
                    cls_summary,

                "patch_summary":
                    patch_summary,

                "attention_entropy":
                    attention_entropy,

                "max_confidence":
                    max_confidence,

                "utility_features":
                    utility_features,

                "utility_probs":
                    utility_probs,

                "utility_mean":
                    utility_logits
                    .mean()
                    .detach(),

                "utility_std":
                    utility_logits
                    .std()
                    .detach(),
            }

            return (
                utility_logits,
                info,
            )

        return utility_logits