# models/bound_main_attention.py

from typing import Dict

import torch
import torch.nn as nn


class BoundMainAttention(nn.Module):
    """
    Main Attention guided by dynamically bound Mini information.

    핵심 구조
    ---------
    Binder가 생성한 Main Head별 seed를
    해당 Main Head의 Query에 직접 주입한다.

        Q_main_h
            =
        Q_base_h
            +
        alpha_h * Seed_h

    여기서 Seed_h는 입력마다 달라진다.

    예:
        Sample A:
            Main H0 <- Direct Mini H1
            Main H1 <- Direct Mini H2
            Main H2 <- Mixed Mini

        Sample B:
            Main H0 <- Direct Mini H2
            Main H1 <- Direct Mini H1
            Main H2 <- Mixed Mini

    따라서 같은 Main Head라도 입력에 따라
    서로 다른 Mini information에 의해 guided될 수 있다.


    Inputs
    ------
    x:
        [B, N, D]

    main_seeds:
        [B, Hmain, N, Dmain]


    Outputs
    -------
    out:
        [B, N, D]

    return_info=True일 경우:
        out, info
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
            raise ValueError(
                f"dim must be > 0, got {dim}"
            )

        if main_heads <= 0:
            raise ValueError(
                f"main_heads must be > 0, got {main_heads}"
            )

        if dim % main_heads != 0:
            raise ValueError(
                "dim must be divisible by main_heads. "
                f"dim={dim}, main_heads={main_heads}"
            )

        self.dim = dim
        self.main_heads = main_heads

        self.main_head_dim = (
            dim // main_heads
        )

        self.scale = (
            self.main_head_dim ** -0.5
        )

        self.normalize_seed = normalize_seed

        # =========================================================
        # Standard Main QKV
        # =========================================================

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias,
        )

        # =========================================================
        # Mini-derived seed normalization
        #
        # Binder output과 Main Q의 scale 차이가
        # 너무 커지는 것을 방지한다.
        # =========================================================

        if normalize_seed:

            self.seed_norm = nn.LayerNorm(
                self.main_head_dim
            )

        else:

            self.seed_norm = nn.Identity()

        # =========================================================
        # Head별 learnable seed strength
        #
        # 초기값 = 1
        #
        # Main H0/H1/H2가 Mini information을
        # 얼마나 사용할지도 학습할 수 있다.
        #
        # shape:
        # [Hmain]
        # =========================================================

        self.seed_scale = nn.Parameter(
            torch.ones(
                main_heads
            )
        )

        self.attn_drop = nn.Dropout(
            attn_drop
        )

        # =========================================================
        # Main output projection
        # =========================================================

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
        """
        Forward.

        Inputs
        ------
        x:
            [B,N,D]

        main_seeds:
            [B,Hmain,N,Dmain]


        Returns
        -------
        out:
            [B,N,D]
        """

        # =========================================================
        # 1. Input 검사
        # =========================================================

        if x.dim() != 3:

            raise ValueError(
                "Expected x [B,N,D], "
                f"got {x.shape}"
            )

        if main_seeds.dim() != 4:

            raise ValueError(
                "Expected main_seeds "
                "[B,Hmain,N,Dmain], "
                f"got {main_seeds.shape}"
            )

        B, N, D = x.shape

        if D != self.dim:

            raise ValueError(
                f"Expected dim={self.dim}, "
                f"got {D}"
            )

        expected_seed_shape = (
            B,
            self.main_heads,
            N,
            self.main_head_dim,
        )

        if main_seeds.shape != expected_seed_shape:

            raise ValueError(
                "main_seeds shape mismatch. "
                f"Expected {expected_seed_shape}, "
                f"got {main_seeds.shape}"
            )

        # =========================================================
        # 2. Main Q/K/V
        # =========================================================

        # [B,N,D]
        #
        # ->
        #
        # [B,N,3D]

        qkv = self.qkv(
            x
        )

        # ---------------------------------------------------------
        # [B,N,3D]
        #
        # ->
        #
        # [B,N,3,Hmain,Dmain]
        #
        # ->
        #
        # [3,B,Hmain,N,Dmain]
        # ---------------------------------------------------------

        qkv = (
            qkv
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

        # q_base:
        # [B,Hmain,N,Dmain]

        # =========================================================
        # 3. Mini-derived Main Seed
        # =========================================================

        normalized_seed = (
            self.seed_norm(
                main_seeds
            )
        )

        # =========================================================
        # 4. Head별 Seed Scale
        #
        # [Hmain]
        #
        # ->
        #
        # [1,Hmain,1,1]
        # =========================================================

        seed_scale = (
            self.seed_scale[
                None,
                :,
                None,
                None,
            ]
        )

        # =========================================================
        # 5. Q-Seeding
        #
        # ★ Mini -> Main 실제 정보 연결 ★
        #
        # q_h =
        # q_base_h + alpha_h * seed_h
        # =========================================================

        q = (
            q_base
            +
            seed_scale
            * normalized_seed
        )

        # =========================================================
        # 6. Main Attention
        # =========================================================

        # q:
        # [B,H,N,Dh]
        #
        # k^T:
        # [B,H,Dh,N]
        #
        # ->
        #
        # [B,H,N,N]

        main_attn = (
            q
            @
            k.transpose(
                -2,
                -1,
            )
        )

        main_attn = (
            main_attn
            * self.scale
        )

        main_attn = torch.softmax(
            main_attn,
            dim=-1,
        )

        main_attn = self.attn_drop(
            main_attn
        )

        # =========================================================
        # 7. Head Output
        # =========================================================

        # [B,H,N,N]
        # @
        # [B,H,N,Dh]
        #
        # ->
        #
        # [B,H,N,Dh]

        head_out = (
            main_attn
            @
            v
        )

        # =========================================================
        # 8. Main Heads merge
        #
        # [B,H,N,Dh]
        #
        # ->
        #
        # [B,N,H,Dh]
        #
        # ->
        #
        # [B,N,D]
        # =========================================================

        merged = (
            head_out
            .transpose(
                1,
                2,
            )
            .reshape(
                B,
                N,
                self.dim,
            )
        )

        # =========================================================
        # 9. Output projection
        # =========================================================

        out = self.proj(
            merged
        )

        out = self.proj_drop(
            out
        )

        # =========================================================
        # 10. Info
        # =========================================================

        if return_info:

            info: Dict[str, torch.Tensor] = {

                "q_base":
                    q_base,

                "normalized_seed":
                    normalized_seed,

                "seed_scale":
                    self.seed_scale,

                "seeded_q":
                    q,

                "main_attn":
                    main_attn,

                "head_out":
                    head_out,

            }

            return (
                out,
                info,
            )

        return out