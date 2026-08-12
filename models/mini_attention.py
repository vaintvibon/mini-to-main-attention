# models/mini_attention.py

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MiniAttention(nn.Module):
    """
    Mini-attention module for Mini-to-Main Two-Level Head Scheduler.

    역할:
        - 입력 token X에서 Q는 전체 token 기준으로 만든다.
        - K/V는 patch token을 spatial pooling한 뒤 만든다.
        - 따라서 attention score는 N x M 형태가 된다.
        - 출력 C_m은 다시 원래 embedding dimension으로 projection한다.

    Input:
        x: [B, N, D]
           ViT 계열에서는 보통 N = 1 + num_patches.
           예: 224x224 image, patch_size=16이면 N = 1 + 14*14 = 197.

    Output:
        c_m:  [B, N, D]
              Mini-attention이 만든 shared context.
        attn: [B, Hm, N, M]
              Mini attention score.
              M = 1 + pooled_patch_tokens if has_cls_token=True.
    """

    def __init__(
        self,
        dim: int,
        mini_heads: int = 1,
        mini_dim: Optional[int] = None,
        pool_ratio: int = 2,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        has_cls_token: bool = True,
    ):
        super().__init__()

        if mini_dim is None:
            mini_dim = dim

        if mini_dim % mini_heads != 0:
            raise ValueError(
                f"mini_dim must be divisible by mini_heads. "
                f"Got mini_dim={mini_dim}, mini_heads={mini_heads}."
            )

        if pool_ratio < 1:
            raise ValueError(f"pool_ratio must be >= 1. Got {pool_ratio}.")

        self.dim = dim
        self.mini_dim = mini_dim
        self.mini_heads = mini_heads
        self.head_dim = mini_dim // mini_heads
        self.scale = self.head_dim ** -0.5
        self.pool_ratio = pool_ratio
        self.has_cls_token = has_cls_token

        self.q = nn.Linear(dim, mini_dim, bias=qkv_bias)
        self.k = nn.Linear(dim, mini_dim, bias=qkv_bias)
        self.v = nn.Linear(dim, mini_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(mini_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _pool_tokens(
        self,
        x: torch.Tensor,
        patch_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Patch token을 spatial average pooling한다.

        Args:
            x: [B, N, D]
            patch_hw:
                patch token의 spatial grid 크기.
                예: 14x14 patch라면 patch_hw=(14, 14).
                None이면 patch token 수가 정사각형이라고 가정하고 자동 계산한다.

        Returns:
            pooled_x: [B, M, D]
                has_cls_token=True이면 CLS token은 pooling하지 않고 그대로 앞에 붙인다.
        """
        B, N, D = x.shape

        if self.has_cls_token:
            cls_token = x[:, :1, :]      # [B, 1, D]
            patch_tokens = x[:, 1:, :]   # [B, N-1, D]
        else:
            cls_token = None
            patch_tokens = x             # [B, N, D]

        num_patch_tokens = patch_tokens.shape[1]

        if patch_hw is None:
            side = int(math.sqrt(num_patch_tokens))
            if side * side != num_patch_tokens:
                raise ValueError(
                    "patch_hw is None, but the number of patch tokens is not square. "
                    f"num_patch_tokens={num_patch_tokens}. "
                    "Pass patch_hw=(H, W) explicitly."
                )
            H, W = side, side
        else:
            H, W = patch_hw
            if H * W != num_patch_tokens:
                raise ValueError(
                    f"patch_hw does not match number of patch tokens. "
                    f"patch_hw={patch_hw}, H*W={H*W}, "
                    f"num_patch_tokens={num_patch_tokens}."
                )

        # [B, Np, D] -> [B, D, H, W]
        patch_map = patch_tokens.transpose(1, 2).reshape(B, D, H, W)

        if self.pool_ratio == 1:
            pooled_map = patch_map
        else:
            pooled_map = F.avg_pool2d(
                patch_map,
                kernel_size=self.pool_ratio,
                stride=self.pool_ratio,
                ceil_mode=False,
            )

        # [B, D, Hp, Wp] -> [B, Mp, D]
        pooled_tokens = pooled_map.flatten(2).transpose(1, 2)

        if cls_token is not None:
            pooled_x = torch.cat([cls_token, pooled_tokens], dim=1)
        else:
            pooled_x = pooled_tokens

        return pooled_x

    def forward(
        self,
        x: torch.Tensor,
        patch_hw: Optional[Tuple[int, int]] = None,
        return_attn: bool = True,
    ):
        """
        Args:
            x: [B, N, D]
            patch_hw: optional tuple (H, W)
            return_attn: True이면 (c_m, attn), False이면 c_m만 반환

        Returns:
            c_m: [B, N, D]
            attn: [B, mini_heads, N, M] if return_attn=True
        """
        B, N, D = x.shape

        if D != self.dim:
            raise ValueError(f"Expected dim={self.dim}, but got input dim={D}.")

        pooled_x = self._pool_tokens(x, patch_hw=patch_hw)  # [B, M, D]
        M = pooled_x.shape[1]

        q = self.q(x)         # [B, N, mini_dim]
        k = self.k(pooled_x)  # [B, M, mini_dim]
        v = self.v(pooled_x)  # [B, M, mini_dim]

        # [B, N, mini_dim] -> [B, Hm, N, head_dim]
        q = q.reshape(B, N, self.mini_heads, self.head_dim).permute(0, 2, 1, 3)

        # [B, M, mini_dim] -> [B, Hm, M, head_dim]
        k = k.reshape(B, M, self.mini_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.mini_heads, self.head_dim).permute(0, 2, 1, 3)

        # [B, Hm, N, M]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # [B, Hm, N, head_dim]
        c_m = attn @ v

        # [B, Hm, N, head_dim] -> [B, N, mini_dim]
        c_m = c_m.transpose(1, 2).reshape(B, N, self.mini_dim)

        # [B, N, mini_dim] -> [B, N, D]
        c_m = self.proj(c_m)
        c_m = self.proj_drop(c_m)

        if return_attn:
            return c_m, attn

        return c_m