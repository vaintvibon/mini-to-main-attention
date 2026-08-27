import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _HardForwardSoftBackward(torch.autograd.Function):
    """
    Forward에서는 hard assignment를 사용하고,
    backward에서는 soft assignment 쪽으로 gradient를 전달한다.
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
    Direct Mini Head를 Main Head에 collision-free 1:1로 동적 binding한다.

    Direct Mini:
        개별 context를 유지한 채 특정 Main Head에 전달.

    Remaining Mini:
        MiniMixer가 만든 mixed_context를 아직 Direct binding되지 않은
        Main Head에 전달.

    Taylor head_gate:
        binding 결정 자체에는 영향을 주지 않고,
        실제 전달되는 Mini contribution의 크기만 조절한다.
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
        self.temperature = float(temperature)
        self.has_cls_token = has_cls_token

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

        self.main_slots = nn.Parameter(
            torch.randn(
                main_heads,
                bind_dim,
            ) * 0.02
        )

        self.direct_proj = nn.Linear(
            mini_head_dim,
            main_head_dim,
        )

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
        if self.has_cls_token:
            if mini_contexts.shape[2] < 2:
                raise ValueError(
                    "CLS + at least one patch token required."
                )

            cls_summary = (
                mini_contexts[:, :, 0, :]
            )

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

        return torch.cat(
            [
                cls_summary,
                patch_summary,
            ],
            dim=-1,
        )

    def _compute_binding_logits(
        self,
        mini_contexts: torch.Tensor,
    ):
        mini_summary = (
            self._summarize_mini_heads(
                mini_contexts
            )
        )

        mini_binding_features = (
            self.mini_bind_proj(
                mini_summary
            )
        )

        binding_logits = torch.einsum(
            "bid,jd->bij",
            mini_binding_features,
            self.main_slots,
        )

        binding_logits = (
            binding_logits
            / math.sqrt(self.bind_dim)
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
                "for collision-free binding. "
                f"direct_k={direct_k}, "
                f"main_heads={Hmain}"
            )

        binding_hard = torch.zeros(
            B,
            Hm,
            Hmain,
            dtype=torch.bool,
            device=binding_logits.device,
        )

        binding_gate = torch.zeros_like(
            binding_logits
        )

        available_main = torch.ones(
            B,
            Hmain,
            dtype=torch.bool,
            device=binding_logits.device,
        )

        for rank in range(direct_k):
            mini_idx = (
                direct_indices[:, rank]
            )

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

            masked_logits = (
                current_logits
                .masked_fill(
                    ~available_main,
                    float("-inf"),
                )
            )

            soft_assignment = (
                torch.softmax(
                    masked_logits
                    / self.temperature,
                    dim=-1,
                )
            )

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

            mini_one_hot = (
                F.one_hot(
                    mini_idx,
                    num_classes=Hm,
                )
                .to(
                    dtype=binding_logits.dtype
                )
            )

            pair_gate = (
                mini_one_hot[:, :, None]
                *
                st_assignment[:, None, :]
            )

            binding_gate = (
                binding_gate
                + pair_gate
            )

            pair_hard = (
                mini_one_hot.bool()[:, :, None]
                &
                hard_assignment.bool()[:, None, :]
            )

            binding_hard = (
                binding_hard
                | pair_hard
            )

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
        head_gate: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ):
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

        if tuple(mixed_context.shape) != (
            B,
            N,
            Dmini,
        ):
            raise ValueError(
                "mixed_context shape mismatch. "
                f"Expected {(B, N, Dmini)}, "
                f"got {tuple(mixed_context.shape)}"
            )

        if head_gate is None:
            head_gate = torch.ones(
                B,
                Hm,
                dtype=mini_contexts.dtype,
                device=mini_contexts.device,
            )

        if tuple(head_gate.shape) != (
            B,
            Hm,
        ):
            raise ValueError(
                "head_gate shape mismatch. "
                f"Expected {(B, Hm)}, "
                f"got {tuple(head_gate.shape)}"
            )

        (
            binding_logits,
            mini_summary,
            mini_binding_features,
        ) = self._compute_binding_logits(
            mini_contexts
        )

        (
            binding_hard,
            binding_gate,
            available_main,
        ) = self._assign_direct_heads(
            binding_logits,
            direct_indices,
        )

        # 먼저 projection한 뒤 gate를 곱한다.
        # 이렇게 해야 xi=0일 때 direct contribution이 정확히 0이 된다.
        projected_direct = (
            self.direct_proj(
                mini_contexts
            )
        )

        projected_direct = (
            projected_direct
            *
            head_gate[
                :,
                :,
                None,
                None,
            ]
        )

        direct_seed = torch.einsum(
            "bim,bind->bmnd",
            binding_gate,
            projected_direct,
        )

        bound_main_mask = (
            binding_hard
            .any(dim=1)
        )

        mixed_main_mask = (
            ~bound_main_mask
        )

        mixed_candidates = torch.stack(
            [
                proj(
                    mixed_context
                )
                for proj in self.mixed_projs
            ],
            dim=1,
        )

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

        main_seeds = (
            direct_seed
            + mixed_seed
        )

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

                "available_main_mask":
                    available_main,
            }

            return (
                main_seeds,
                info,
            )

        return main_seeds
