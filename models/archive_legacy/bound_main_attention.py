from typing import Dict

import torch
import torch.nn as nn


class BoundMainAttention(nn.Module):
    """
    Main Attention guided by dynamically bound Mini information.

    Q_main_h = Q_base_h + alpha_h * Seed_h

    Inputs
    ------
    x:
        [B, N, D]

    main_seeds:
        [B, Hmain, N, Dmain]
    """

    def __init__(
        self,
        dim: int,
        main_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        normalize_seed: bool = True,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError(f"dim must be > 0, got {dim}")

        if main_heads <= 0:
            raise ValueError(f"main_heads must be > 0, got {main_heads}")

        if dim % main_heads != 0:
            raise ValueError(
                "dim must be divisible by main_heads. "
                f"dim={dim}, main_heads={main_heads}"
            )

        self.dim = dim
        self.main_heads = main_heads
        self.main_head_dim = dim // main_heads
        self.scale = self.main_head_dim ** -0.5

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias,
        )

        self.seed_norm = (
            nn.LayerNorm(self.main_head_dim)
            if normalize_seed
            else nn.Identity()
        )

        self.seed_scale = nn.Parameter(
            torch.ones(main_heads)
        )

        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(
            dim,
            dim,
        )

        self.proj_drop = nn.Dropout(
            proj_drop
        )

    def forward(
        self,
        x: torch.Tensor,
        main_seeds: torch.Tensor,
        return_info: bool = False,
    ):
        if x.dim() != 3:
            raise ValueError(
                "Expected x [B,N,D], "
                f"got {x.shape}"
            )

        if main_seeds.dim() != 4:
            raise ValueError(
                "Expected main_seeds [B,Hmain,N,Dmain], "
                f"got {main_seeds.shape}"
            )

        B, N, D = x.shape

        if D != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {D}"
            )

        expected_seed_shape = (
            B,
            self.main_heads,
            N,
            self.main_head_dim,
        )

        if tuple(main_seeds.shape) != expected_seed_shape:
            raise ValueError(
                "main_seeds shape mismatch. "
                f"Expected {expected_seed_shape}, "
                f"got {tuple(main_seeds.shape)}"
            )

        qkv = (
            self.qkv(x)
            .reshape(
                B,
                N,
                3,
                self.main_heads,
                self.main_head_dim,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        q_base = qkv[0]
        k = qkv[1]
        v = qkv[2]

        seed = self.seed_norm(
            main_seeds
        )

        seed_scale = self.seed_scale[
            None,
            :,
            None,
            None,
        ]

        q = (
            q_base
            + seed_scale * seed
        )

        main_attn = (
            q
            @ k.transpose(-2, -1)
        ) * self.scale

        main_attn = torch.softmax(
            main_attn,
            dim=-1,
        )

        main_attn = self.attn_drop(
            main_attn
        )

        head_out = (
            main_attn @ v
        )

        merged = (
            head_out
            .transpose(1, 2)
            .reshape(
                B,
                N,
                self.dim,
            )
        )

        out = self.proj(
            merged
        )

        out = self.proj_drop(
            out
        )

        if return_info:
            info: Dict[str, torch.Tensor] = {
                "q_base": q_base,
                "normalized_seed": seed,
                "seed_scale": self.seed_scale,
                "seeded_q": q,
                "main_attn": main_attn,
                "head_out": head_out,
            }

            return out, info

        return out
