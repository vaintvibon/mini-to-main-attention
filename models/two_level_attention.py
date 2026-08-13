from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.mini_attention import MiniAttention
from models.mini_to_main_allocator import MiniImportance, MiniToMainAllocator
from models.two_level_scheduler import TwoLevelHeadScheduler


class TwoLevelMiniMainAttention(nn.Module):
    """
    v1 Two-Level Mini-to-Main Attention + differentiable Gumbel-ST scheduling.

    중요:
        - 여전히 모든 Main head Q/K/V + attention을 계산한 뒤 gate를 적용한다.
          따라서 이 파일은 구조/학습 검증용 v1이며, FLOPs 절감 주장은 불가능하다.
        - 하지만 scheduler의 active/direct/mixed gate는 Straight-Through estimator이므로
          classification loss gradient가 allocator까지 전달된다.
    """

    def __init__(
        self,
        dim: int,
        main_heads: int,
        mini_heads: int = 1,
        mini_dim: int = 64,
        pool_ratio: int = 2,
        direct_ratio: float = 0.34,
        alpha_direct: float = 1.0,
        alpha_mixed: float = 0.2,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        allocator_hidden_dim: int = 128,
        has_cls_token: bool = True,
        gumbel_tau: float = 1.0,
        use_gumbel: bool = True,
    ):
        super().__init__()

        if dim % main_heads != 0:
            raise ValueError(
                f"dim must be divisible by main_heads. "
                f"Got dim={dim}, main_heads={main_heads}."
            )

        self.dim = dim
        self.main_heads = main_heads
        self.head_dim = dim // main_heads
        self.scale = self.head_dim ** -0.5

        self.alpha_direct = alpha_direct
        self.alpha_mixed = alpha_mixed
        self.has_cls_token = has_cls_token

        self.mini_attn = MiniAttention(
            dim=dim,
            mini_heads=mini_heads,
            mini_dim=mini_dim,
            pool_ratio=pool_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            has_cls_token=has_cls_token,
        )

        self.mini_importance = MiniImportance()

        self.allocator = MiniToMainAllocator(
            dim=dim,
            main_heads=main_heads,
            importance_dim=2,
            hidden_dim=allocator_hidden_dim,
            dropout=proj_drop,
            use_cls_token=has_cls_token,
        )

        self.scheduler = TwoLevelHeadScheduler(
            main_heads=main_heads,
            direct_ratio=direct_ratio,
            gumbel_tau=gumbel_tau,
            use_gumbel=use_gumbel,
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.seed_proj = nn.Linear(dim, dim, bias=True)

        self.main_attn_drop = nn.Dropout(attn_drop)
        self.main_proj = nn.Linear(dim, dim)
        self.main_proj_drop = nn.Dropout(proj_drop)

        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def set_gumbel_temperature(self, tau: float) -> None:
        self.scheduler.set_temperature(tau)

    def forward(
        self,
        x: torch.Tensor,
        budget: int,
        patch_hw: Optional[Tuple[int, int]] = None,
        return_info: bool = False,
    ):
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, N, D], but got {x.shape}.")

        B, N, D = x.shape

        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, but got input dim={D}.")

        # 1. Mini-attention
        c_m, mini_attn_score = self.mini_attn(
            x,
            patch_hw=patch_hw,
            return_attn=True,
        )

        # 2. Mini importance
        importance_feat, importance_stats = self.mini_importance(mini_attn_score)

        # 3. Mini-to-Main allocation logits
        alloc_logits = self.allocator(c_m, importance_feat)

        # 4. Scheduler
        schedule_out = self.scheduler(alloc_logits, budget=budget)

        # hard masks: logging / diversity pairing / exact semantic checks
        active_mask = schedule_out["active_mask"]
        direct_mask = schedule_out["direct_mask"]
        mixed_mask = schedule_out["mixed_mask"]
        inactive_mask = schedule_out["inactive_mask"]

        # differentiable gates: MUST be used in actual forward computation
        active_gate = schedule_out["active_gate"]
        direct_gate = schedule_out["direct_gate"]
        mixed_gate = schedule_out["mixed_gate"]

        # 5. Budget 0: Mini-only path
        # Allocation decision itself is irrelevant when no Main head can be used.
        if budget == 0:
            zero_main_out = torch.zeros_like(c_m)
            zero_head_out = x.new_zeros(B, self.main_heads, N, self.head_dim)

            out = self.out_proj(c_m)
            out = self.out_drop(out)

            if return_info:
                info = {
                    "mini_context": c_m,
                    "mini_attn_score": mini_attn_score,
                    "importance_feat": importance_feat,
                    "importance_stats": importance_stats,
                    "alloc_logits": alloc_logits,
                    "active_mask": active_mask,
                    "direct_mask": direct_mask,
                    "mixed_mask": mixed_mask,
                    "inactive_mask": inactive_mask,
                    "active_gate": active_gate,
                    "direct_gate": direct_gate,
                    "mixed_gate": mixed_gate,
                    "selection_scores": schedule_out["selection_scores"],
                    "head_out": zero_head_out,
                    "main_out": zero_main_out,
                    "scheduler_stats": schedule_out["stats"],
                }
                return out, info

            return out

        # 6. Main QKV (v1: all heads are still densely computed)
        qkv = self.qkv(x)  # [B, N, 3D]

        qkv = qkv.reshape(
            B,
            N,
            3,
            self.main_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)

        q = qkv[0]  # [B, H, N, Dh]
        k = qkv[1]  # [B, H, N, Dh]
        v = qkv[2]  # [B, H, N, Dh]

        # 7. Q-seeding from Mini context
        seed = self.seed_proj(c_m)  # [B, N, D]

        seed = seed.reshape(
            B,
            N,
            self.main_heads,
            self.head_dim,
        ).permute(0, 2, 1, 3)  # [B, H, N, Dh]

        # IMPORTANT:
        # bool mask가 아니라 ST gate를 사용해야 allocator까지 gradient가 연결된다.
        alpha = (
            direct_gate * self.alpha_direct
            + mixed_gate * self.alpha_mixed
        )  # [B, H]

        alpha = alpha[:, :, None, None]  # [B, H, 1, 1]
        q = q + alpha * seed

        # 8. Main attention
        main_attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        main_attn = main_attn.softmax(dim=-1)
        main_attn = self.main_attn_drop(main_attn)

        head_out = main_attn @ v  # [B, H, N, Dh]

        # diversity loss는 hard direct/mixed pair를 사용하되,
        # head representation 자체에는 gradient가 유지되어야 하므로 detach하지 않는다.
        head_out_for_loss = head_out

        # 9. Active ST gate 적용
        # forward에서는 exact 0/1 hard mask와 동일,
        # backward에서는 relaxed Top-K gradient가 allocator로 흐른다.
        head_out = head_out * active_gate[:, :, None, None]

        # [B, H, N, Dh] -> [B, N, D]
        main_out = head_out.transpose(1, 2).reshape(B, N, D)

        main_out = self.main_proj(main_out)
        main_out = self.main_proj_drop(main_out)

        # 10. Mini + Main 결합
        out = c_m + main_out
        out = self.out_proj(out)
        out = self.out_drop(out)

        if return_info:
            info = {
                "mini_context": c_m,
                "mini_attn_score": mini_attn_score,
                "importance_feat": importance_feat,
                "importance_stats": importance_stats,
                "alloc_logits": alloc_logits,
                "active_mask": active_mask,
                "direct_mask": direct_mask,
                "mixed_mask": mixed_mask,
                "inactive_mask": inactive_mask,
                "active_gate": active_gate,
                "direct_gate": direct_gate,
                "mixed_gate": mixed_gate,
                "selection_scores": schedule_out["selection_scores"],
                "main_attn": main_attn,
                "head_out": head_out_for_loss,
                "main_out": main_out,
                "scheduler_stats": schedule_out["stats"],
            }
            return out, info

        return out
