# models/mini_to_main_allocator.py

from typing import Dict, Tuple

import torch
import torch.nn as nn


class MiniImportance(nn.Module):
    """
    Mini-attention score에서 relation confidence를 계산한다.

    입력:
        attn: [B, Hm, N, M]
            Mini attention score.
            Hm = mini_heads
            N = query token 수
            M = pooled key/value token 수

    출력:
        importance_feat: [B, 2]
            sample별 Mini confidence feature.
            [:, 0] = max-attention confidence
            [:, 1] = entropy-based confidence

        stats:
            디버깅용 통계값 dictionary
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, attn: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if attn.dim() != 4:
            raise ValueError(
                f"Expected attn shape [B, Hm, N, M], but got {attn.shape}."
            )

        B, Hm, N, M = attn.shape

        # 1. Max attention confidence
        # 각 query token이 pooled token 중 하나에 얼마나 강하게 집중했는지
        # [B, Hm, N]
        max_prob = attn.max(dim=-1).values

        # sample별 평균 confidence
        # [B, 1]
        max_conf = max_prob.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)

        # 2. Entropy-based confidence
        # attention이 퍼져 있으면 entropy 높음 → confidence 낮음
        # attention이 뾰족하면 entropy 낮음 → confidence 높음
        entropy = -(attn * (attn + self.eps).log()).sum(dim=-1)  # [B, Hm, N]

        # entropy normalization: log(M)
        max_entropy = torch.log(
            torch.tensor(float(M), device=attn.device, dtype=attn.dtype)
        )

        normalized_entropy = entropy / (max_entropy + self.eps)  # [B, Hm, N]

        # confidence = 1 - normalized entropy
        entropy_conf = 1.0 - normalized_entropy.mean(dim=(1, 2), keepdim=False)
        entropy_conf = entropy_conf.unsqueeze(-1)  # [B, 1]

        # [B, 2]
        importance_feat = torch.cat([max_conf, entropy_conf], dim=-1)

        stats = {
            "max_conf_mean": max_conf.mean().detach(),
            "entropy_conf_mean": entropy_conf.mean().detach(),
            "attn_max_mean": max_prob.mean().detach(),
            "attn_entropy_mean": entropy.mean().detach(),
        }

        return importance_feat, stats


class MiniToMainAllocator(nn.Module):
    """
    Mini shared context와 Mini confidence를 이용해
    Main head slot별 allocation logits를 생성한다.

    중요:
        alloc_logits_h는 h번째 Main head 자체의 중요도가 아니다.
        "Mini가 찾은 relation을 h번째 Main head slot에 배정할 가치"다.

    입력:
        c_m: [B, N, D]
            MiniAttention이 만든 shared context.

        importance_feat: [B, importance_dim]
            MiniImportance가 만든 confidence feature.

    출력:
        alloc_logits: [B, main_heads]
            Main head slot별 allocation score.
    """

    def __init__(
        self,
        dim: int,
        main_heads: int,
        importance_dim: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        use_cls_token: bool = True,
    ):
        super().__init__()

        if main_heads <= 0:
            raise ValueError(f"main_heads must be positive. Got {main_heads}.")

        self.dim = dim
        self.main_heads = main_heads
        self.importance_dim = importance_dim
        self.hidden_dim = hidden_dim
        self.use_cls_token = use_cls_token

        input_dim = dim + importance_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, main_heads),
        )

    def forward(
        self,
        c_m: torch.Tensor,
        importance_feat: torch.Tensor,
    ) -> torch.Tensor:
        if c_m.dim() != 3:
            raise ValueError(f"Expected c_m shape [B, N, D], but got {c_m.shape}.")

        if importance_feat.dim() != 2:
            raise ValueError(
                f"Expected importance_feat shape [B, importance_dim], "
                f"but got {importance_feat.shape}."
            )

        B, N, D = c_m.shape

        if D != self.dim:
            raise ValueError(f"Expected c_m dim={self.dim}, but got {D}.")

        if importance_feat.shape[0] != B:
            raise ValueError(
                f"Batch size mismatch: c_m batch={B}, "
                f"importance_feat batch={importance_feat.shape[0]}."
            )

        if importance_feat.shape[1] != self.importance_dim:
            raise ValueError(
                f"Expected importance_dim={self.importance_dim}, "
                f"but got {importance_feat.shape[1]}."
            )

        if self.use_cls_token:
            # CLS token의 Mini shared context 사용
            context_summary = c_m[:, 0, :]  # [B, D]
        else:
            # CLS token이 없는 구조라면 전체 token 평균 사용
            context_summary = c_m.mean(dim=1)  # [B, D]

        z = torch.cat([context_summary, importance_feat], dim=-1)

        alloc_logits = self.mlp(z)  # [B, main_heads]

        return alloc_logits