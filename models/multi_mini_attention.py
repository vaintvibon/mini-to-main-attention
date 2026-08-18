import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiMiniAttention(nn.Module):
    """
    Multi-Head Mini Attention for Dynamic Mini-to-Main Routing.

    목적
    ----
    여러 개의 저비용 Mini Head가 입력을 먼저 분석한다.

    중요한 설계 원칙:
        Mini Head별 정보를 절대 여기서 합치지 않는다.

    각 Mini Head의 context는 이후

        1. Mini Head Utility 평가
        2. Direct Mini 선택
        3. Remaining Mini Mix
        4. Mini -> Main Dynamic Binding

    에 사용된다.


    Input
    -----
    x:
        [B, N, D]

        B : batch size
        N : token 수
        D : ViT embedding dimension


    Output
    ------
    mini_contexts:
        [B, Hm, N, Dh]

        Hm : Mini Head 수
        Dh : Mini Head dimension

        mini_contexts[:, h]
        = h번째 Mini Head가 추출한 정보


    mini_attn:
        [B, Hm, N, M]

        M : pooling 이후 K/V token 수
    """

    def __init__(
        self,
        dim: int,
        mini_heads: int,
        mini_head_dim: int,
        pool_ratio: int = 2,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        has_cls_token: bool = True,
    ):
        super().__init__()

        if mini_heads <= 0:
            raise ValueError(
                f"mini_heads must be > 0, got {mini_heads}"
            )

        if mini_head_dim <= 0:
            raise ValueError(
                f"mini_head_dim must be > 0, got {mini_head_dim}"
            )

        if pool_ratio < 1:
            raise ValueError(
                f"pool_ratio must be >= 1, got {pool_ratio}"
            )

        self.dim = dim

        self.mini_heads = mini_heads
        self.mini_head_dim = mini_head_dim

        self.mini_dim = (
            mini_heads * mini_head_dim
        )

        self.pool_ratio = pool_ratio
        self.has_cls_token = has_cls_token

        self.scale = mini_head_dim ** -0.5

        # ---------------------------------------------------------
        # Mini Q/K/V
        #
        # 하나의 projection을 사용하지만,
        # 출력 channel을 Mini Head별 subspace로 분리한다.
        #
        # 예:
        # mini_heads = 4
        # mini_head_dim = 16
        #
        # mini_dim = 64
        # ---------------------------------------------------------

        self.q_proj = nn.Linear(
            dim,
            self.mini_dim,
            bias=qkv_bias,
        )

        self.k_proj = nn.Linear(
            dim,
            self.mini_dim,
            bias=qkv_bias,
        )

        self.v_proj = nn.Linear(
            dim,
            self.mini_dim,
            bias=qkv_bias,
        )

        self.attn_drop = nn.Dropout(
            attn_drop
        )

    def _pool_tokens(
        self,
        x: torch.Tensor,
        patch_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        K/V 계산에 사용할 patch token을 spatial pooling한다.

        CLS token은 pooling하지 않는다.


        Input
        -----
        x:
            [B, N, D]


        Output
        ------
        pooled_x:
            [B, M, D]
        """

        B, N, D = x.shape

        # ---------------------------------------------------------
        # CLS / Patch 분리
        # ---------------------------------------------------------

        if self.has_cls_token:

            cls_token = x[:, :1, :]
            patch_tokens = x[:, 1:, :]

        else:

            cls_token = None
            patch_tokens = x

        num_patch_tokens = patch_tokens.shape[1]

        # ---------------------------------------------------------
        # Patch grid 확인
        # ---------------------------------------------------------

        if patch_hw is None:

            side = int(
                math.sqrt(num_patch_tokens)
            )

            if side * side != num_patch_tokens:
                raise ValueError(
                    "Number of patch tokens is not square. "
                    f"num_patch_tokens={num_patch_tokens}. "
                    "Pass patch_hw=(H, W)."
                )

            H = side
            W = side

        else:

            H, W = patch_hw

            if H * W != num_patch_tokens:
                raise ValueError(
                    "patch_hw does not match patch tokens. "
                    f"patch_hw={patch_hw}, "
                    f"H*W={H * W}, "
                    f"num_patch_tokens={num_patch_tokens}"
                )

        # ---------------------------------------------------------
        # [B, Npatch, D]
        #
        # ->
        #
        # [B, D, H, W]
        # ---------------------------------------------------------

        patch_map = (
            patch_tokens
            .transpose(1, 2)
            .reshape(
                B,
                D,
                H,
                W,
            )
        )

        # ---------------------------------------------------------
        # Spatial pooling
        # ---------------------------------------------------------

        if self.pool_ratio == 1:

            pooled_map = patch_map

        else:

            pooled_map = F.avg_pool2d(
                patch_map,
                kernel_size=self.pool_ratio,
                stride=self.pool_ratio,
            )

        # ---------------------------------------------------------
        # [B, D, Hp, Wp]
        #
        # ->
        #
        # [B, Mpatch, D]
        # ---------------------------------------------------------

        pooled_tokens = (
            pooled_map
            .flatten(2)
            .transpose(1, 2)
        )

        # ---------------------------------------------------------
        # CLS token 복원
        # ---------------------------------------------------------

        if cls_token is not None:

            pooled_tokens = torch.cat(
                [
                    cls_token,
                    pooled_tokens,
                ],
                dim=1,
            )

        return pooled_tokens

    def forward(
        self,
        x: torch.Tensor,
        patch_hw: Optional[Tuple[int, int]] = None,
    ):
        """
        Forward.


        Returns
        -------
        mini_contexts:
            [B, Hm, N, Dh]

        mini_attn:
            [B, Hm, N, M]
        """

        if x.dim() != 3:

            raise ValueError(
                "Expected x shape [B, N, D], "
                f"got {x.shape}"
            )

        B, N, D = x.shape

        if D != self.dim:

            raise ValueError(
                f"Expected dim={self.dim}, "
                f"got D={D}"
            )

        # =========================================================
        # 1. K/V용 token pooling
        # =========================================================

        pooled_x = self._pool_tokens(
            x,
            patch_hw=patch_hw,
        )

        M = pooled_x.shape[1]

        # =========================================================
        # 2. Q/K/V projection
        # =========================================================

        # Q:
        # [B, N, mini_dim]

        q = self.q_proj(x)

        # K/V:
        # [B, M, mini_dim]

        k = self.k_proj(pooled_x)
        v = self.v_proj(pooled_x)

        # =========================================================
        # 3. Mini Head별로 분리
        # =========================================================

        # ---------------------------------------------------------
        # Q
        #
        # [B, N, Hm * Dh]
        #
        # ->
        #
        # [B, Hm, N, Dh]
        # ---------------------------------------------------------

        q = (
            q.reshape(
                B,
                N,
                self.mini_heads,
                self.mini_head_dim,
            )
            .permute(
                0,
                2,
                1,
                3,
            )
        )

        # ---------------------------------------------------------
        # K
        #
        # [B, M, Hm * Dh]
        #
        # ->
        #
        # [B, Hm, M, Dh]
        # ---------------------------------------------------------

        k = (
            k.reshape(
                B,
                M,
                self.mini_heads,
                self.mini_head_dim,
            )
            .permute(
                0,
                2,
                1,
                3,
            )
        )

        # ---------------------------------------------------------
        # V
        # ---------------------------------------------------------

        v = (
            v.reshape(
                B,
                M,
                self.mini_heads,
                self.mini_head_dim,
            )
            .permute(
                0,
                2,
                1,
                3,
            )
        )

        # =========================================================
        # 4. Mini Attention
        # =========================================================

        # q:
        # [B, Hm, N, Dh]
        #
        # k^T:
        # [B, Hm, Dh, M]
        #
        # ->
        #
        # [B, Hm, N, M]

        mini_attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        mini_attn = mini_attn.softmax(
            dim=-1
        )

        mini_attn = self.attn_drop(
            mini_attn
        )

        # =========================================================
        # 5. Mini Head별 context 계산
        # =========================================================

        # mini_attn:
        # [B, Hm, N, M]
        #
        # v:
        # [B, Hm, M, Dh]
        #
        # ->
        #
        # mini_contexts:
        # [B, Hm, N, Dh]

        mini_contexts = (
            mini_attn @ v
        )

        # =========================================================
        # 중요
        # =========================================================
        #
        # 여기서 Head를 concat하지 않는다.
        #
        # projection도 하지 않는다.
        #
        # Head별 identity를 그대로 유지한다.
        #
        # 다음 모듈인 MiniHeadUtility가
        # 이 representation들을 독립적으로 평가한다.
        # =========================================================

        return (
            mini_contexts,
            mini_attn,
        )