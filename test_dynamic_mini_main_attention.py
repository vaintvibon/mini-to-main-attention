from typing import Dict, Any, Optional

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

    Final dynamic mode:
        Mini -> Utility Predictor -> Top-K Direct / Remaining
             -> Remaining utility-weighted Mix
             -> Dynamic Mini->Main Binding
             -> Main Q-Seeding

    Stage-1 / Counterfactual mode:
        forced_direct_indices를 전달하면 Utility Predictor의 Top-K를 우회한다.
        forced_uniform_mix=True이면 Remaining Mix 역시 Utility Predictor에
        의존하지 않고 균등 혼합한다.

    이렇게 해야 warm-up / teacher 생성 시 초기 random Utility Predictor가
    routing을 자기강화하는 문제를 막을 수 있다.
    """

    def __init__(
        self,
        dim: int,
        mini_heads: int = 4,
        mini_head_dim: int = 16,
        pool_ratio: int = 2,
        utility_hidden_dim: int = 64,
        utility_dropout: float = 0.0,
        direct_k: int = 2,
        mix_temperature: float = 1.0,
        main_heads: int = 3,
        bind_dim: int = 64,
        bind_temperature: float = 1.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        has_cls_token: bool = True,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError(
                f"dim must be > 0, got {dim}"
            )

        if mini_heads <= 0:
            raise ValueError(
                f"mini_heads must be > 0, got {mini_heads}"
            )

        if mini_head_dim <= 0:
            raise ValueError(
                f"mini_head_dim must be > 0, got {mini_head_dim}"
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

        if direct_k <= 0:
            raise ValueError(
                f"direct_k must be > 0, got {direct_k}"
            )

        if direct_k > mini_heads:
            raise ValueError(
                "direct_k cannot exceed mini_heads. "
                f"direct_k={direct_k}, mini_heads={mini_heads}"
            )

        if direct_k > main_heads:
            raise ValueError(
                "direct_k cannot exceed main_heads "
                "for collision-free 1:1 binding. "
                f"direct_k={direct_k}, main_heads={main_heads}"
            )

        self.dim = dim
        self.mini_heads = mini_heads
        self.mini_head_dim = mini_head_dim
        self.main_heads = main_heads
        self.main_head_dim = (
            dim // main_heads
        )
        self.direct_k = direct_k

        self.mini_attention = MultiMiniAttention(
            dim=dim,
            mini_heads=mini_heads,
            mini_head_dim=mini_head_dim,
            pool_ratio=pool_ratio,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            has_cls_token=has_cls_token,
        )

        self.utility_predictor = MiniHeadUtility(
            mini_head_dim=mini_head_dim,
            hidden_dim=utility_hidden_dim,
            dropout=utility_dropout,
            has_cls_token=has_cls_token,
        )

        self.selector = DynamicMiniSelector(
            mini_heads=mini_heads,
            direct_k=direct_k,
        )

        self.mixer = MiniMixer(
            mini_heads=mini_heads,
            temperature=mix_temperature,
        )

        self.binder = MiniMainBinder(
            mini_head_dim=mini_head_dim,
            main_heads=main_heads,
            main_head_dim=self.main_head_dim,
            bind_dim=bind_dim,
            temperature=bind_temperature,
            has_cls_token=has_cls_token,
        )

        # Taylor gate의 magnitude 효과를 보존하기 위해 seed LN은 끈다.
        self.main_attention = BoundMainAttention(
            dim=dim,
            main_heads=main_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            normalize_seed=False,
        )

    def set_mix_temperature(
        self,
        temperature: float,
    ):
        self.mixer.set_temperature(
            temperature
        )

    def set_binding_temperature(
        self,
        temperature: float,
    ):
        self.binder.set_temperature(
            temperature
        )

    def _build_forced_selection(
        self,
        utility_logits: torch.Tensor,
        forced_direct_indices: torch.Tensor,
    ):
        """
        Utility Predictor의 Top-K를 우회해 명시적으로 Direct Mini를 지정한다.
        """

        B, Hm = (
            utility_logits.shape
        )

        forced_direct_indices = (
            forced_direct_indices
            .to(
                device=utility_logits.device,
                dtype=torch.long,
            )
        )

        expected_shape = (
            B,
            self.direct_k,
        )

        if tuple(
            forced_direct_indices.shape
        ) != expected_shape:
            raise ValueError(
                "forced_direct_indices shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {tuple(forced_direct_indices.shape)}"
            )

        if torch.any(
            forced_direct_indices < 0
        ) or torch.any(
            forced_direct_indices >= Hm
        ):
            raise ValueError(
                "forced_direct_indices contains an invalid Mini Head index."
            )

        sorted_indices = (
            forced_direct_indices
            .sort(dim=-1)
            .values
        )

        if self.direct_k > 1:
            duplicate = (
                sorted_indices[:, 1:]
                ==
                sorted_indices[:, :-1]
            )

            if torch.any(
                duplicate
            ):
                raise ValueError(
                    "Each sample must contain unique forced Direct Mini indices."
                )

        direct_mask = torch.zeros(
            B,
            Hm,
            dtype=torch.bool,
            device=utility_logits.device,
        )

        direct_mask.scatter_(
            dim=1,
            index=forced_direct_indices,
            value=True,
        )

        remaining_mask = (
            ~direct_mask
        )

        direct_scores = torch.gather(
            utility_logits,
            dim=1,
            index=forced_direct_indices,
        )

        selection_info = {
            "direct_indices":
                forced_direct_indices,

            # logging용. forced routing에서는 선택 근거가 아님.
            "direct_scores":
                direct_scores,

            "direct_count":
                direct_mask.sum(dim=-1),

            "remaining_count":
                remaining_mask.sum(dim=-1),

            "forced_routing":
                True,
        }

        return (
            direct_mask,
            remaining_mask,
            selection_info,
        )

    def forward(
        self,
        x: torch.Tensor,
        patch_hw=None,
        return_info: bool = False,
        collect_taylor: bool = False,
        forced_direct_indices: Optional[torch.Tensor] = None,
        forced_uniform_mix: bool = False,
    ):
        if x.dim() != 3:
            raise ValueError(
                "Expected x [B,N,D], "
                f"got {x.shape}"
            )

        B, _, D = x.shape

        if D != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {D}"
            )

        if collect_taylor and not return_info:
            raise ValueError(
                "collect_taylor=True requires return_info=True."
            )

        # =========================================================
        # 1. Mini
        # =========================================================

        (
            mini_contexts,
            mini_attn,
        ) = self.mini_attention(
            x,
            patch_hw=patch_hw,
        )

        # =========================================================
        # 2. Utility Predictor
        #
        # forced mode에서도 logging / Stage-2 student 입력을 위해 계산은 한다.
        # 단, forced routing과 uniform mix에서는 routing 결정에 쓰이지 않는다.
        # =========================================================

        (
            utility_logits,
            utility_info,
        ) = self.utility_predictor(
            mini_contexts,
            mini_attn,
            return_info=True,
        )

        # =========================================================
        # 3. Direct / Remaining
        # =========================================================

        if forced_direct_indices is None:
            (
                direct_mask,
                remaining_mask,
                selection_info,
            ) = self.selector(
                utility_logits,
                return_info=True,
            )

            selection_info = dict(
                selection_info
            )

            selection_info[
                "forced_routing"
            ] = False

        else:
            (
                direct_mask,
                remaining_mask,
                selection_info,
            ) = self._build_forced_selection(
                utility_logits,
                forced_direct_indices,
            )

        # =========================================================
        # 4. Taylor gate
        # =========================================================

        taylor_gate = torch.ones(
            B,
            self.mini_heads,
            dtype=mini_contexts.dtype,
            device=mini_contexts.device,
        )

        if collect_taylor:
            taylor_gate.requires_grad_()

            if not taylor_gate.requires_grad:
                raise RuntimeError(
                    "Taylor gate failed to enable gradients."
                )

            if not taylor_gate.is_leaf:
                raise RuntimeError(
                    "Taylor gate must be a leaf tensor."
                )

        gated_mini_contexts = (
            mini_contexts
            *
            taylor_gate[
                :,
                :,
                None,
                None,
            ]
        )

        # =========================================================
        # 5. Remaining Mix
        #
        # forced_uniform_mix=True이면 utility logits를 완전히 제거하고
        # remaining head들을 균등하게 mix한다.
        # =========================================================

        if forced_uniform_mix:
            mixer_logits = torch.zeros_like(
                utility_logits
            )
        else:
            mixer_logits = (
                utility_logits
            )

        (
            mixed_context,
            mix_info,
        ) = self.mixer(
            gated_mini_contexts,
            mixer_logits,
            remaining_mask,
            return_info=True,
        )

        # =========================================================
        # 6. Dynamic Mini -> Main Binding
        #
        # binding compatibility는 ungated Mini representation을 사용한다.
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
            head_gate=taylor_gate,
            return_info=True,
        )

        # =========================================================
        # 7. Main
        # =========================================================

        (
            out,
            main_info,
        ) = self.main_attention(
            x,
            main_seeds,
            return_info=True,
        )

        if return_info:
            info: Dict[str, Any] = {
                "taylor_gate":
                    taylor_gate,

                "gated_mini_contexts":
                    gated_mini_contexts,

                "collect_taylor":
                    collect_taylor,

                "mini_contexts":
                    mini_contexts,

                "mini_attn":
                    mini_attn,

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

                "direct_mask":
                    direct_mask,

                "remaining_mask":
                    remaining_mask,

                "direct_indices":
                    selection_info[
                        "direct_indices"
                    ],

                "forced_routing":
                    selection_info[
                        "forced_routing"
                    ],

                "forced_uniform_mix":
                    forced_uniform_mix,

                "mix_weights":
                    mix_info[
                        "mix_weights"
                    ],

                "mixed_context":
                    mixed_context,

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
