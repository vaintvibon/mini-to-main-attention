# models/dynamic_mini_main_attention.py

from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

from models.multi_mini_attention import MultiMiniAttention
from models.mini_head_utility import MiniHeadUtility
from models.dynamic_mini_selector import DynamicMiniSelector
from models.mini_mixer import MiniMixer
from models.mini_main_binder import MiniMainBinder
from models.bound_main_attention import BoundMainAttention


class DynamicMiniMainAttention(nn.Module):
    """
    Dynamic Mini-to-Main Attention.

    전체 연구 구조를 하나로 묶은 Attention module.

    구조
    ----
    1. Multi Mini Attention
        여러 Mini Head가 저비용 attention 수행

    2. Mini Head Utility Prediction
        각 Mini Head가 현재 입력에서 얼마나 유용한지 예측

    3. Dynamic Mini Selection
        utility가 높은 Top-K Mini Head를 Direct 대상으로 선택

    4. Remaining Mini Mixing
        Direct되지 않은 Mini Head 정보는 버리지 않고
        utility-weighted Mix로 요약

    5. Dynamic Mini -> Main Binding
        Direct Mini들을 입력마다 적절한 Main Head에 1:1 binding

        남은 Main Head는 Mixed Mini 정보를 받음

    6. Bound Main Attention
        Main Head별 Mini-derived seed를 해당 Main Query에 주입


    핵심 연구 의미
    -------------
    입력마다:

        어떤 Mini가 중요한지,
        어떤 Mini가 Direct인지,
        나머지 Mini가 어떻게 Mix되는지,
        Direct Mini가 어느 Main에 연결되는지

    가 달라질 수 있다.


    Input
    -----
    x:
        [B, N, D]


    Output
    ------
    out:
        [B, N, D]
    """

    def __init__(
        self,
        dim: int,

        # Mini
        mini_heads: int = 4,
        mini_head_dim: int = 16,
        pool_ratio: int = 2,

        # Mini utility
        utility_hidden_dim: int = 64,
        utility_dropout: float = 0.0,

        # Direct selection
        direct_k: int = 2,

        # Mini mixing
        mix_temperature: float = 1.0,

        # Main
        main_heads: int = 3,

        # Dynamic binding
        bind_dim: int = 64,
        bind_temperature: float = 1.0,

        # Attention
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,

        has_cls_token: bool = True,
    ):
        super().__init__()

        # =========================================================
        # Validation
        # =========================================================

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

        if direct_k > mini_heads:
            raise ValueError(
                "direct_k cannot exceed mini_heads. "
                f"direct_k={direct_k}, "
                f"mini_heads={mini_heads}"
            )

        # 현재 Binder는
        # Direct Mini 하나 ↔ Main 하나의
        # collision-free 1:1 binding 구조다.
        if direct_k > main_heads:
            raise ValueError(
                "direct_k cannot exceed main_heads "
                "for 1:1 Direct Binding. "
                f"direct_k={direct_k}, "
                f"main_heads={main_heads}"
            )

        self.dim = dim

        self.mini_heads = mini_heads
        self.mini_head_dim = mini_head_dim

        self.main_heads = main_heads
        self.main_head_dim = (
            dim // main_heads
        )

        self.direct_k = direct_k

        # =========================================================
        # 1. Multi Mini Attention
        # =========================================================

        self.mini_attention = MultiMiniAttention(
            dim=dim,
            mini_heads=mini_heads,
            mini_head_dim=mini_head_dim,
            pool_ratio=pool_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            has_cls_token=has_cls_token,
        )

        # =========================================================
        # 2. Mini Head Utility Predictor
        # =========================================================

        self.utility_predictor = MiniHeadUtility(
            mini_head_dim=mini_head_dim,
            hidden_dim=utility_hidden_dim,
            dropout=utility_dropout,
            has_cls_token=has_cls_token,
        )

        # =========================================================
        # 3. Dynamic Mini Selector
        # =========================================================

        self.selector = DynamicMiniSelector(
            mini_heads=mini_heads,
            direct_k=direct_k,
        )

        # =========================================================
        # 4. Remaining Mini Mixer
        # =========================================================

        self.mixer = MiniMixer(
            mini_heads=mini_heads,
            temperature=mix_temperature,
        )

        # =========================================================
        # 5. Dynamic Mini -> Main Binder
        # =========================================================

        self.binder = MiniMainBinder(
            mini_head_dim=mini_head_dim,
            main_heads=main_heads,
            main_head_dim=self.main_head_dim,
            bind_dim=bind_dim,
            temperature=bind_temperature,
            has_cls_token=has_cls_token,
        )

        # =========================================================
        # 6. Main Attention
        # =========================================================

        self.main_attention = BoundMainAttention(
            dim=dim,
            main_heads=main_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            normalize_seed=True,
        )

    def set_mix_temperature(
        self,
        temperature: float,
    ):
        """
        Remaining Mini Mix temperature 변경.
        """

        self.mixer.set_temperature(
            temperature
        )

    def set_binding_temperature(
        self,
        temperature: float,
    ):
        """
        Mini -> Main binding temperature 변경.
        """

        self.binder.set_temperature(
            temperature
        )

    def forward(
        self,
        x: torch.Tensor,
        patch_hw: Optional[
            Tuple[int, int]
        ] = None,
        return_info: bool = False,
    ):
        """
        Forward.

        Parameters
        ----------
        x:
            [B,N,D]

        patch_hw:
            patch grid.

            예:
                14x14 patches
                -> (14,14)

        return_info:
            True이면 routing/debug 정보를 함께 반환.


        Returns
        -------
        out:
            [B,N,D]

        return_info=True:
            out, info
        """

        if x.dim() != 3:
            raise ValueError(
                "Expected x [B,N,D], "
                f"got {x.shape}"
            )

        B, N, D = x.shape

        if D != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, "
                f"got {D}"
            )

        # =========================================================
        # 1. Multi Mini Attention
        #
        # 각 Mini Head의 identity를 유지한다.
        # =========================================================

        (
            mini_contexts,
            mini_attn,
        ) = self.mini_attention(
            x,
            patch_hw=patch_hw,
        )

        # mini_contexts:
        # [B,Hmini,N,Dmini]
        #
        # mini_attn:
        # [B,Hmini,N,M]

        # =========================================================
        # 2. Input-dependent Mini Head Utility
        # =========================================================

        (
            utility_logits,
            utility_info,
        ) = self.utility_predictor(
            mini_contexts,
            mini_attn,
            return_info=True,
        )

        # utility_logits:
        # [B,Hmini]

        # =========================================================
        # 3. Dynamic Direct Mini Selection
        # =========================================================

        (
            direct_mask,
            remaining_mask,
            selection_info,
        ) = self.selector(
            utility_logits,
            return_info=True,
        )

        # =========================================================
        # 4. Remaining Mini Dynamic Mixing
        # =========================================================

        (
            mixed_context,
            mix_info,
        ) = self.mixer(
            mini_contexts,
            utility_logits,
            remaining_mask,
            return_info=True,
        )

        # mixed_context:
        # [B,N,Dmini]

        # =========================================================
        # 5. Dynamic Mini -> Main Binding
        # =========================================================

        (
            main_seeds,
            binding_info,
        ) = self.binder(
            mini_contexts,
            selection_info[
                "direct_indices"
            ],
            mixed_context,
            return_info=True,
        )

        # main_seeds:
        #
        # [B,Hmain,N,Dmain]

        # =========================================================
        # 6. Main Attention
        #
        # each Main Query receives its own
        # dynamically assigned Mini seed.
        # =========================================================

        (
            out,
            main_info,
        ) = self.main_attention(
            x,
            main_seeds,
            return_info=True,
        )

        # out:
        # [B,N,D]

        # =========================================================
        # 7. Return
        # =========================================================

        if return_info:

            info: Dict[str, Any] = {

                # -------------------------------------------------
                # Mini
                # -------------------------------------------------

                "mini_contexts":
                    mini_contexts,

                "mini_attn":
                    mini_attn,

                # -------------------------------------------------
                # Utility
                # -------------------------------------------------

                "utility_logits":
                    utility_logits,

                "utility_probs":
                    utility_info[
                        "utility_probs"
                    ],

                "attention_entropy":
                    utility_info[
                        "attention_entropy"
                    ],

                "max_confidence":
                    utility_info[
                        "max_confidence"
                    ],

                # -------------------------------------------------
                # Selection
                # -------------------------------------------------

                "direct_mask":
                    direct_mask,

                "remaining_mask":
                    remaining_mask,

                "direct_indices":
                    selection_info[
                        "direct_indices"
                    ],

                # -------------------------------------------------
                # Mix
                # -------------------------------------------------

                "mix_weights":
                    mix_info[
                        "mix_weights"
                    ],

                "mixed_context":
                    mixed_context,

                # -------------------------------------------------
                # Binding
                # -------------------------------------------------

                "binding_logits":
                    binding_info[
                        "binding_logits"
                    ],

                "binding_hard":
                    binding_info[
                        "binding_hard"
                    ],

                "binding_gate":
                    binding_info[
                        "binding_gate"
                    ],

                "bound_main_mask":
                    binding_info[
                        "bound_main_mask"
                    ],

                "mixed_main_mask":
                    binding_info[
                        "mixed_main_mask"
                    ],

                # -------------------------------------------------
                # Main
                # -------------------------------------------------

                "main_seeds":
                    main_seeds,

                "seeded_q":
                    main_info[
                        "seeded_q"
                    ],

                "main_attn":
                    main_info[
                        "main_attn"
                    ],

                "main_head_out":
                    main_info[
                        "head_out"
                    ],

                "seed_scale":
                    main_info[
                        "seed_scale"
                    ],
            }

            return (
                out,
                info,
            )

        return out