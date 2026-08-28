import math
from typing import Dict, Any

import torch
import torch.nn as nn


class SetAwareMiniHeadUtility(nn.Module):
    """
    Set-aware per-Mini-head utility predictor.

    기존 local-only predictor의 문제:
        각 Mini Head를 거의 독립적으로 점수화하므로
        "이 Head가 다른 Head들과 비교해서 얼마나 유용한가"를
        표현하기 어렵다.

    이 모듈은 최종 출력은 여전히 per-head utility [B,H]로 유지하면서,
    각 Head를 전체 Mini Head set의 문맥 안에서 점수화한다.

    구조
    ----
    1) Head-local feature
       - CLS context
       - patch mean context
       - normalized attention entropy
       - max attention confidence

    2) Shared local encoder

    3) Set context
       - encoded head mean
       - encoded head max
       - head - set mean (relative feature)

    4) Shared scorer
       -> utility logit per head

    중요한 성질
    -----------
    - Head 순서를 바꾸면 score 순서도 같이 바뀌는 permutation-equivariant 구조.
    - 별도의 learned head-ID embedding을 사용하지 않는다.
      즉 static head prior를 외우는 것보다 input-dependent relative utility를
      학습하도록 유도한다.
    """

    def __init__(
        self,
        mini_head_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        has_cls_token: bool = True,
        eps: float = 1e-6,
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

        # CLS [Dh] + patch mean [Dh] + entropy [1] + confidence [1]
        self.local_feature_dim = (
            2 * mini_head_dim + 2
        )

        self.local_encoder = nn.Sequential(
            nn.LayerNorm(
                self.local_feature_dim
            ),
            nn.Linear(
                self.local_feature_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        # local + set_mean + set_max + relative(local - mean)
        score_input_dim = (
            4 * hidden_dim
        )

        self.scorer = nn.Sequential(
            nn.LayerNorm(
                score_input_dim
            ),
            nn.Linear(
                score_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def _attention_statistics(
        self,
        mini_attn: torch.Tensor,
    ):
        """
        mini_attn:
            [B,H,N,M]

        returns:
            normalized_entropy [B,H]
            max_confidence     [B,H]
        """

        if mini_attn.dim() != 4:
            raise ValueError(
                "mini_attn must have shape [B,H,N,M], "
                f"got {mini_attn.shape}"
            )

        _, _, _, M = (
            mini_attn.shape
        )

        probabilities = (
            mini_attn.clamp_min(
                self.eps
            )
        )

        entropy = -(
            probabilities
            *
            probabilities.log()
        ).sum(
            dim=-1
        )

        # [B,H,N] -> [B,H]
        entropy = entropy.mean(
            dim=-1
        )

        max_entropy = math.log(
            float(M)
        )

        normalized_entropy = (
            entropy
            /
            max(
                max_entropy,
                self.eps,
            )
        )

        max_confidence = (
            mini_attn.max(
                dim=-1
            ).values
            .mean(
                dim=-1
            )
        )

        return (
            normalized_entropy,
            max_confidence,
        )

    def _build_local_features(
        self,
        mini_contexts: torch.Tensor,
        mini_attn: torch.Tensor,
    ):
        """
        mini_contexts:
            [B,H,N,Dh]

        mini_attn:
            [B,H,N,M]

        returns:
            local_features [B,H,F]
            statistics
        """

        if mini_contexts.dim() != 4:
            raise ValueError(
                "mini_contexts must have shape [B,H,N,Dh], "
                f"got {mini_contexts.shape}"
            )

        B, H, N, Dh = (
            mini_contexts.shape
        )

        if Dh != self.mini_head_dim:
            raise ValueError(
                f"Expected mini_head_dim={self.mini_head_dim}, "
                f"got {Dh}"
            )

        if mini_attn.shape[:3] != (
            B,
            H,
            N,
        ):
            raise ValueError(
                "mini_contexts / mini_attn shape mismatch. "
                f"contexts={mini_contexts.shape}, "
                f"attn={mini_attn.shape}"
            )

        if self.has_cls_token:
            cls_context = (
                mini_contexts[
                    :,
                    :,
                    0,
                    :,
                ]
            )

            if N > 1:
                patch_mean = (
                    mini_contexts[
                        :,
                        :,
                        1:,
                        :,
                    ]
                    .mean(
                        dim=2
                    )
                )
            else:
                patch_mean = (
                    cls_context
                )
        else:
            # CLS가 없다면 두 summary 모두 전체 token 통계로 둔다.
            cls_context = (
                mini_contexts.mean(
                    dim=2
                )
            )

            patch_mean = (
                cls_context
            )

        (
            attention_entropy,
            max_confidence,
        ) = self._attention_statistics(
            mini_attn
        )

        local_features = torch.cat(
            [
                cls_context,
                patch_mean,
                attention_entropy[
                    ...,
                    None,
                ],
                max_confidence[
                    ...,
                    None,
                ],
            ],
            dim=-1,
        )

        return (
            local_features,
            cls_context,
            patch_mean,
            attention_entropy,
            max_confidence,
        )

    def forward(
        self,
        mini_contexts: torch.Tensor,
        mini_attn: torch.Tensor,
        return_info: bool = False,
    ):
        (
            local_features,
            cls_context,
            patch_mean,
            attention_entropy,
            max_confidence,
        ) = self._build_local_features(
            mini_contexts,
            mini_attn,
        )

        # ---------------------------------------------------------
        # Shared local encoder
        # ---------------------------------------------------------

        local_encoded = (
            self.local_encoder(
                local_features
            )
        )

        # [B,H,C]
        set_mean = (
            local_encoded.mean(
                dim=1,
                keepdim=True,
            )
        )

        set_max = (
            local_encoded.max(
                dim=1,
                keepdim=True,
            ).values
        )

        relative = (
            local_encoded
            -
            set_mean
        )

        H = local_encoded.shape[1]

        set_mean_expanded = (
            set_mean.expand(
                -1,
                H,
                -1,
            )
        )

        set_max_expanded = (
            set_max.expand(
                -1,
                H,
                -1,
            )
        )

        score_features = torch.cat(
            [
                local_encoded,
                set_mean_expanded,
                set_max_expanded,
                relative,
            ],
            dim=-1,
        )

        utility_logits = (
            self.scorer(
                score_features
            )
            .squeeze(-1)
        )

        utility_probs = torch.softmax(
            utility_logits,
            dim=-1,
        )

        if return_info:
            info: Dict[str, Any] = {
                "utility_probs":
                    utility_probs,

                "attention_entropy":
                    attention_entropy,

                "max_confidence":
                    max_confidence,

                "cls_summary":
                    cls_context,

                "patch_mean":
                    patch_mean,

                # Compatibility/debug aliases
                "features":
                    local_features,

                "local_features":
                    local_features,

                "local_encoded":
                    local_encoded,

                "set_mean":
                    set_mean.squeeze(1),

                "set_max":
                    set_max.squeeze(1),

                "relative_features":
                    relative,

                "score_features":
                    score_features,
            }

            return (
                utility_logits,
                info,
            )

        return utility_logits
