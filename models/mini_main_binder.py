# models/mini_main_binder.py

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HardForwardSoftBackward(torch.autograd.Function):
    """
    Forward:
        hard assignment 사용

    Backward:
        soft assignment surrogate를 통해 gradient 전달

    즉 실제 forward에서는 Mini -> Main 연결이 명확한 1:1 binding이지만,
    training에서는 binding compatibility network도 학습할 수 있다.
    """

    @staticmethod
    def forward(
        ctx,
        hard: torch.Tensor,
        soft: torch.Tensor,
    ):
        return hard

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ):
        return None, grad_output


class MiniMainBinder(nn.Module):
    """
    Dynamic Mini-to-Main Binder.

    역할
    ----
    1. Direct로 선택된 Mini Head 각각에 대해
       어떤 Main Head와 가장 잘 맞는지 compatibility를 계산한다.

    2. Direct Mini Head를 서로 다른 Main Head에
       동적으로 1:1 binding한다.

    3. Direct Binding되지 않은 Main Head에는
       Remaining Mini들이 만든 mixed_context를 전달한다.

    따라서:

        좋은 Mini -> Direct -> 특정 Main

        나머지 Mini -> Mix -> 남은 Main

    구조가 된다.


    Inputs
    ------
    mini_contexts:
        [B, Hm, N, Dmini]

    direct_indices:
        [B, K]

        DynamicMiniSelector에서 선택된
        Direct Mini Head indices.

    mixed_context:
        [B, N, Dmini]


    Outputs
    -------
    main_seeds:
        [B, Hmain, N, Dmain]

        각 Main Head가 받을 Mini-derived seed.

    binding_logits:
        [B, Hm, Hmain]

        Mini-Main pairwise compatibility.
    """

    def __init__(
        self,
        mini_head_dim: int,
        main_heads: int,
        main_head_dim: int,
        bind_dim: int = 64,
        temperature: float = 1.0,
        has_cls_token: bool = True,
    ):
        super().__init__()

        if mini_head_dim <= 0:
            raise ValueError(
                f"mini_head_dim must be > 0, got {mini_head_dim}"
            )

        if main_heads <= 0:
            raise ValueError(
                f"main_heads must be > 0, got {main_heads}"
            )

        if main_head_dim <= 0:
            raise ValueError(
                f"main_head_dim must be > 0, got {main_head_dim}"
            )

        if bind_dim <= 0:
            raise ValueError(
                f"bind_dim must be > 0, got {bind_dim}"
            )

        if temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {temperature}"
            )

        self.mini_head_dim = mini_head_dim
        self.main_heads = main_heads
        self.main_head_dim = main_head_dim
        self.bind_dim = bind_dim

        self.temperature = temperature
        self.has_cls_token = has_cls_token

        # =========================================================
        # 1. Mini Head -> Binding feature
        #
        # CLS + Patch mean
        #
        # [Dmini + Dmini]
        #       ↓
        # bind_dim
        # =========================================================

        self.mini_bind_proj = nn.Sequential(
            nn.LayerNorm(
                mini_head_dim * 2
            ),

            nn.Linear(
                mini_head_dim * 2,
                bind_dim,
            ),

            nn.GELU(),

            nn.Linear(
                bind_dim,
                bind_dim,
            ),
        )

        # =========================================================
        # 2. Main Head identity / slot embedding
        #
        # 각 Main Head는 학습 가능한 slot representation을 가진다.
        #
        # Mini의 현재 representation과
        # Main slot 사이 compatibility를 계산한다.
        # =========================================================

        self.main_slots = nn.Parameter(
            torch.randn(
                main_heads,
                bind_dim,
            ) * 0.02
        )

        # =========================================================
        # 3. Direct Mini -> Main Head dimension projection
        #
        # Mini context:
        # Dmini
        #
        # ->
        #
        # Main head:
        # Dmain
        # =========================================================

        self.direct_proj = nn.Linear(
            mini_head_dim,
            main_head_dim,
        )

        # =========================================================
        # 4. Mixed information projection
        #
        # 각 Main Head가 같은 Mixed Context를 받더라도
        # 서로 다른 projection을 사용한다.
        #
        # 따라서 Main H0/H1/H2가
        # Mixed 정보를 서로 다른 방식으로 해석할 수 있다.
        # =========================================================

        self.mixed_projs = nn.ModuleList(
            [
                nn.Linear(
                    mini_head_dim,
                    main_head_dim,
                )
                for _ in range(main_heads)
            ]
        )

    def set_temperature(
        self,
        temperature: float,
    ):
        if temperature <= 0:
            raise ValueError(
                f"temperature must be > 0, got {temperature}"
            )

        self.temperature = float(
            temperature
        )

    def _summarize_mini_heads(
        self,
        mini_contexts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mini Head별 representation을 binding용으로 요약.

        Input:
            [B, Hm, N, Dmini]

        Output:
            [B, Hm, 2*Dmini]
        """

        if self.has_cls_token:

            if mini_contexts.shape[2] < 2:
                raise ValueError(
                    "CLS + at least one patch token required."
                )

            # [B, Hm, Dmini]
            cls_summary = (
                mini_contexts[:, :, 0, :]
            )

            # [B, Hm, Dmini]
            patch_summary = (
                mini_contexts[:, :, 1:, :]
                .mean(dim=2)
            )

        else:

            cls_summary = (
                mini_contexts
                .max(dim=2)
                .values
            )

            patch_summary = (
                mini_contexts
                .mean(dim=2)
            )

        # [B, Hm, 2Dmini]
        summary = torch.cat(
            [
                cls_summary,
                patch_summary,
            ],
            dim=-1,
        )

        return summary

    def _compute_binding_logits(
        self,
        mini_contexts: torch.Tensor,
    ):
        """
        Mini Head x Main Head compatibility 계산.

        Returns
        -------
        binding_logits:
            [B, Hm, Hmain]
        """

        # ---------------------------------------------------------
        # Mini summary
        #
        # [B,Hm,N,Dmini]
        # ->
        # [B,Hm,2Dmini]
        # ---------------------------------------------------------

        mini_summary = (
            self._summarize_mini_heads(
                mini_contexts
            )
        )

        # ---------------------------------------------------------
        # Mini binding representation
        #
        # [B,Hm,bind_dim]
        # ---------------------------------------------------------

        mini_binding_features = (
            self.mini_bind_proj(
                mini_summary
            )
        )

        # =========================================================
        # Mini-Main compatibility
        #
        # Mini:
        # [B,Hm,Db]
        #
        # Main slot:
        # [Hmain,Db]
        #
        # ->
        #
        # [B,Hm,Hmain]
        # =========================================================

        binding_logits = torch.einsum(
            "bid,jd->bij",
            mini_binding_features,
            self.main_slots,
        )

        binding_logits = (
            binding_logits
            / math.sqrt(
                self.bind_dim
            )
        )

        return (
            binding_logits,
            mini_summary,
            mini_binding_features,
        )

    def _assign_direct_heads(
        self,
        binding_logits: torch.Tensor,
        direct_indices: torch.Tensor,
    ):
        """
        Direct Mini를 서로 다른 Main Head에 binding한다.

        중요:
            collision을 허용하지 않는다.

        예:

            Mini H0 -> Main H2

            그 다음 Mini H3는
            Main H2를 다시 사용할 수 없다.

        Direct Mini는 direct_indices 순서대로 배정한다.
        DynamicMiniSelector의 direct_indices가 utility descending
        순서이므로 가장 중요한 Mini가 먼저 Main을 선택한다.


        Returns
        -------
        binding_hard:
            [B, Hm, Hmain] bool

        binding_gate:
            [B, Hm, Hmain] float

            forward = exact hard binding
            backward = soft assignment gradient
        """

        B, Hm, Hmain = (
            binding_logits.shape
        )

        if direct_indices.dim() != 2:
            raise ValueError(
                "Expected direct_indices [B,K], "
                f"got {direct_indices.shape}"
            )

        if direct_indices.shape[0] != B:
            raise ValueError(
                "Batch mismatch in direct_indices."
            )

        direct_k = (
            direct_indices.shape[1]
        )

        if direct_k > Hmain:
            raise ValueError(
                "direct_k cannot exceed main_heads "
                "when collision-free direct binding is used. "
                f"direct_k={direct_k}, "
                f"main_heads={Hmain}"
            )

        # Hard assignment matrix
        binding_hard = torch.zeros(
            B,
            Hm,
            Hmain,
            dtype=torch.bool,
            device=binding_logits.device,
        )

        # Differentiable assignment
        binding_gate = torch.zeros_like(
            binding_logits
        )

        # 각 sample에서 아직 비어있는 Main
        available_main = torch.ones(
            B,
            Hmain,
            dtype=torch.bool,
            device=binding_logits.device,
        )

        # =========================================================
        # 가장 utility가 높은 Direct Mini부터 하나씩 배정
        # =========================================================

        for rank in range(direct_k):

            # 현재 Direct Mini index
            #
            # [B]

            mini_idx = (
                direct_indices[:, rank]
            )

            # -----------------------------------------------------
            # 해당 Mini의 Main compatibility만 추출
            #
            # [B,Hmain]
            # -----------------------------------------------------

            current_logits = (
                binding_logits.gather(
                    dim=1,
                    index=mini_idx[
                        :,
                        None,
                        None,
                    ].expand(
                        -1,
                        1,
                        Hmain,
                    ),
                )
                .squeeze(1)
            )

            # -----------------------------------------------------
            # 이미 사용된 Main은 선택 불가능
            # -----------------------------------------------------

            masked_logits = (
                current_logits
                .masked_fill(
                    ~available_main,
                    float("-inf"),
                )
            )

            # -----------------------------------------------------
            # Soft assignment
            #
            # backward용
            # -----------------------------------------------------

            soft_assignment = (
                torch.softmax(
                    masked_logits
                    / self.temperature,
                    dim=-1,
                )
            )

            # -----------------------------------------------------
            # Hard assignment
            #
            # forward용
            # -----------------------------------------------------

            selected_main = (
                masked_logits
                .argmax(dim=-1)
            )

            hard_assignment = (
                F.one_hot(
                    selected_main,
                    num_classes=Hmain,
                )
                .to(
                    dtype=binding_logits.dtype
                )
            )

            # -----------------------------------------------------
            # Training:
            #
            # Forward = hard
            # Backward = soft
            #
            # Eval:
            #
            # completely hard
            # -----------------------------------------------------

            if self.training:

                st_assignment = (
                    _HardForwardSoftBackward
                    .apply(
                        hard_assignment,
                        soft_assignment,
                    )
                )

            else:

                st_assignment = (
                    hard_assignment
                )

            # -----------------------------------------------------
            # 현재 Mini index를 one-hot
            #
            # [B,Hm]
            # -----------------------------------------------------

            mini_one_hot = (
                F.one_hot(
                    mini_idx,
                    num_classes=Hm,
                )
                .to(
                    dtype=binding_logits.dtype
                )
            )

            # -----------------------------------------------------
            # Mini x Main pair assignment
            #
            # [B,Hm,1]
            # *
            # [B,1,Hmain]
            #
            # ->
            #
            # [B,Hm,Hmain]
            # -----------------------------------------------------

            pair_gate = (
                mini_one_hot[
                    :,
                    :,
                    None,
                ]
                *
                st_assignment[
                    :,
                    None,
                    :,
                ]
            )

            binding_gate = (
                binding_gate
                + pair_gate
            )

            # Hard logging matrix
            pair_hard = (
                mini_one_hot.bool()[
                    :,
                    :,
                    None,
                ]
                &
                hard_assignment.bool()[
                    :,
                    None,
                    :,
                ]
            )

            binding_hard = (
                binding_hard
                | pair_hard
            )

            # -----------------------------------------------------
            # 선택된 Main을 다음 Mini가 사용할 수 없게 한다.
            # -----------------------------------------------------

            available_main = (
                available_main
                &
                ~hard_assignment.bool()
            )

        return (
            binding_hard,
            binding_gate,
            available_main,
        )

    def forward(
        self,
        mini_contexts: torch.Tensor,
        direct_indices: torch.Tensor,
        mixed_context: torch.Tensor,
        return_info: bool = False,
    ):
        """
        Inputs
        ------
        mini_contexts:
            [B,Hm,N,Dmini]

        direct_indices:
            [B,K]

        mixed_context:
            [B,N,Dmini]


        Outputs
        -------
        main_seeds:
            [B,Hmain,N,Dmain]
        """

        # =========================================================
        # 1. Shape 검사
        # =========================================================

        if mini_contexts.dim() != 4:
            raise ValueError(
                "Expected mini_contexts [B,Hm,N,Dmini], "
                f"got {mini_contexts.shape}"
            )

        B, Hm, N, Dmini = (
            mini_contexts.shape
        )

        if Dmini != self.mini_head_dim:
            raise ValueError(
                f"Expected mini_head_dim={self.mini_head_dim}, "
                f"got {Dmini}"
            )

        if mixed_context.shape != (
            B,
            N,
            Dmini,
        ):
            raise ValueError(
                "mixed_context shape mismatch. "
                f"Expected {(B, N, Dmini)}, "
                f"got {mixed_context.shape}"
            )

        # =========================================================
        # 2. Mini -> Main compatibility
        # =========================================================

        (
            binding_logits,
            mini_summary,
            mini_binding_features,
        ) = self._compute_binding_logits(
            mini_contexts
        )

        # binding_logits:
        #
        # [B,Hm,Hmain]

        # =========================================================
        # 3. Direct Mini -> Main dynamic assignment
        # =========================================================

        (
            binding_hard,
            binding_gate,
            available_main,
        ) = self._assign_direct_heads(
            binding_logits,
            direct_indices,
        )

        # =========================================================
        # 4. Direct Mini Context projection
        #
        # [B,Hm,N,Dmini]
        #
        # ->
        #
        # [B,Hm,N,Dmain]
        # =========================================================

        projected_direct = (
            self.direct_proj(
                mini_contexts
            )
        )

        # =========================================================
        # 5. Binding을 이용해 Main Head별 Direct Seed 생성
        #
        # binding:
        # [B,Hm,Hmain]
        #
        # context:
        # [B,Hm,N,Dmain]
        #
        # ->
        #
        # [B,Hmain,N,Dmain]
        # =========================================================

        direct_seed = torch.einsum(
            "bim,bind->bmnd",
            binding_gate,
            projected_direct,
        )

        # =========================================================
        # 6. Direct Binding된 Main mask
        # =========================================================

        bound_main_mask = (
            binding_hard
            .any(dim=1)
        )
        # [B,Hmain]

        mixed_main_mask = (
            ~bound_main_mask
        )

        # =========================================================
        # 7. Mixed Context를 Main Head별로 projection
        #
        # 각 Main마다 별도 Linear를 사용.
        #
        # [B,N,Dmini]
        #
        # ->
        #
        # [B,Hmain,N,Dmain]
        # =========================================================

        mixed_candidates = torch.stack(
            [
                proj(
                    mixed_context
                )
                for proj in self.mixed_projs
            ],
            dim=1,
        )

        # =========================================================
        # 8. Direct Binding되지 않은 Main에만 Mixed 전달
        # =========================================================

        mixed_seed = (
            mixed_candidates
            *
            mixed_main_mask[
                :,
                :,
                None,
                None,
            ].to(
                dtype=mixed_candidates.dtype
            )
        )

        # =========================================================
        # 9. Main별 최종 Mini-derived seed
        # =========================================================

        main_seeds = (
            direct_seed
            +
            mixed_seed
        )

        # [B,Hmain,N,Dmain]

        # =========================================================
        # 10. Invariant
        # =========================================================

        # 하나의 Main에 Direct Mini가 둘 이상 붙으면 안 됨.
        incoming_direct_count = (
            binding_hard
            .sum(dim=1)
        )

        if torch.any(
            incoming_direct_count > 1
        ):
            raise RuntimeError(
                "Multiple Direct Mini Heads were bound "
                "to the same Main Head."
            )

        # 각 Direct Mini는 정확히 하나의 Main으로 간다.
        direct_binding_count = (
            binding_hard
            .sum(dim=(1, 2))
        )

        expected_direct_count = (
            direct_indices.shape[1]
        )

        if not torch.all(
            direct_binding_count
            == expected_direct_count
        ):
            raise RuntimeError(
                "Not every Direct Mini Head was bound "
                "to exactly one Main Head."
            )

        # =========================================================
        # 11. Return
        # =========================================================

        if return_info:

            info: Dict[str, torch.Tensor] = {

                "binding_logits":
                    binding_logits,

                "binding_hard":
                    binding_hard,

                "binding_gate":
                    binding_gate,

                "bound_main_mask":
                    bound_main_mask,

                "mixed_main_mask":
                    mixed_main_mask,

                "direct_seed":
                    direct_seed,

                "mixed_seed":
                    mixed_seed,

                "mini_summary":
                    mini_summary,

                "mini_binding_features":
                    mini_binding_features,

                "incoming_direct_count":
                    incoming_direct_count,

            }

            return (
                main_seeds,
                info,
            )

        return main_seeds