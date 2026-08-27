from itertools import combinations
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class CounterfactualDirectUtilityEvaluator:
    """
    Mini Head의 utility를 다음과 같이 측정한다.

        U_h =
            E[L | h not in Direct]
            -
            E[L | h in Direct]

    U_h > 0:
        h를 Direct에 포함했을 때 평균 task loss가 낮아진다.
        즉 Direct 후보로 유용하다.

    중요
    ----
    - Utility Predictor의 Top-K는 사용하지 않는다.
    - Remaining Mix도 uniform으로 고정한다.
    - target block 하나의 Direct subset만 바꾸고,
      다른 block은 neutral reference routing으로 고정한다.

    따라서 predictor가 만든 routing -> teacher -> predictor의
    즉각적인 feedback loop를 끊는다.

    이 teacher는 정확히는:
        "reference routing 하의 block-local Direct inclusion benefit"
    이다.
    """

    def __init__(
        self,
        model,
        mini_heads: int,
        direct_k: int,
        target_temperature: float = 1.0,
    ):
        if mini_heads <= 1:
            raise ValueError(
                "mini_heads must be > 1."
            )

        if direct_k <= 0:
            raise ValueError(
                "direct_k must be > 0."
            )

        if direct_k >= mini_heads:
            raise ValueError(
                "Counterfactual utility requires "
                "0 < direct_k < mini_heads."
            )

        if target_temperature <= 0:
            raise ValueError(
                "target_temperature must be > 0."
            )

        self.model = model
        self.mini_heads = mini_heads
        self.direct_k = direct_k
        self.target_temperature = float(
            target_temperature
        )

        self.combinations = list(
            combinations(
                range(mini_heads),
                direct_k,
            )
        )

        self.num_combinations = len(
            self.combinations
        )

        include_mask = torch.zeros(
            self.num_combinations,
            mini_heads,
            dtype=torch.bool,
        )

        for combo_idx, combo in enumerate(
            self.combinations
        ):
            include_mask[
                combo_idx,
                list(combo),
            ] = True

        self.include_mask_cpu = (
            include_mask
        )

    def _combo_batch(
        self,
        combo_idx: int,
        batch_size: int,
        device,
    ) -> torch.Tensor:
        combo = torch.tensor(
            self.combinations[
                combo_idx
            ],
            dtype=torch.long,
            device=device,
        )

        return (
            combo[
                None,
                :,
            ]
            .expand(
                batch_size,
                -1,
            )
            .clone()
        )

    def _default_reference(
        self,
        batch_size: int,
        depth: int,
        device,
    ) -> List[torch.Tensor]:
        """
        block마다 다른 neutral reference subset을 사용한다.
        """

        references = []

        for block_idx in range(
            depth
        ):
            combo_idx = (
                block_idx
                % self.num_combinations
            )

            references.append(
                self._combo_batch(
                    combo_idx=combo_idx,
                    batch_size=batch_size,
                    device=device,
                )
            )

        return references

    def _utility_to_target(
        self,
        head_utility: torch.Tensor,
    ) -> torch.Tensor:
        """
        Utility는 양/음 값을 가질 수 있으므로 단순 합 정규화하지 않는다.

        sample/block별로 center + standardize 후 softmax하여
        ranking 중심의 soft teacher target을 만든다.
        """

        centered = (
            head_utility
            -
            head_utility.mean(
                dim=-1,
                keepdim=True,
            )
        )

        scale = (
            centered
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
            .sqrt()
        )

        eps = torch.finfo(
            head_utility.dtype
        ).eps

        standardized = torch.where(
            scale > eps,
            centered / scale.clamp_min(eps),
            torch.zeros_like(
                centered
            ),
        )

        teacher_target = torch.softmax(
            standardized
            / self.target_temperature,
            dim=-1,
        )

        return teacher_target

    @torch.no_grad()
    def evaluate(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        reference_direct_indices_per_block=None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        subset_losses:
            [B, depth, num_combinations]

        head_utility:
            [B, depth, Hmini]

        teacher_target:
            [B, depth, Hmini]

        best_subset_indices:
            [B, depth]

        best_subset:
            [B, depth, direct_k]
        """

        if x.dim() != 4:
            raise ValueError(
                "Expected x [B,C,H,W]."
            )

        B = x.shape[0]
        depth = self.model.depth
        device = x.device

        if labels.shape[0] != B:
            raise ValueError(
                "labels batch size mismatch."
            )

        if reference_direct_indices_per_block is None:
            references = (
                self._default_reference(
                    batch_size=B,
                    depth=depth,
                    device=device,
                )
            )
        else:
            if len(
                reference_direct_indices_per_block
            ) != depth:
                raise ValueError(
                    "reference_direct_indices_per_block length mismatch."
                )

            references = [
                tensor.to(
                    device=device,
                    dtype=torch.long,
                )
                for tensor in reference_direct_indices_per_block
            ]

        previous_training = (
            self.model.training
        )

        self.model.eval()

        subset_losses = torch.empty(
            B,
            depth,
            self.num_combinations,
            dtype=x.dtype,
            device=device,
        )

        for target_block in range(
            depth
        ):
            for combo_idx in range(
                self.num_combinations
            ):
                forced = [
                    ref.clone()
                    for ref in references
                ]

                forced[
                    target_block
                ] = self._combo_batch(
                    combo_idx=combo_idx,
                    batch_size=B,
                    device=device,
                )

                logits = self.model(
                    x,
                    return_info=False,
                    collect_taylor=False,
                    forced_direct_indices_per_block=forced,
                    forced_uniform_mix=True,
                )

                losses = F.cross_entropy(
                    logits,
                    labels,
                    reduction="none",
                )

                subset_losses[
                    :,
                    target_block,
                    combo_idx,
                ] = losses

        include_mask = (
            self.include_mask_cpu
            .to(
                device=device
            )
        )

        head_utility = torch.empty(
            B,
            depth,
            self.mini_heads,
            dtype=x.dtype,
            device=device,
        )

        for head_idx in range(
            self.mini_heads
        ):
            included = (
                include_mask[
                    :,
                    head_idx,
                ]
            )

            excluded = (
                ~included
            )

            include_loss = (
                subset_losses[
                    :,
                    :,
                    included,
                ]
                .mean(dim=-1)
            )

            exclude_loss = (
                subset_losses[
                    :,
                    :,
                    excluded,
                ]
                .mean(dim=-1)
            )

            # positive => Direct inclusion lowers loss
            head_utility[
                :,
                :,
                head_idx,
            ] = (
                exclude_loss
                -
                include_loss
            )

        teacher_target = (
            self._utility_to_target(
                head_utility
            )
        )

        best_subset_indices = (
            subset_losses
            .argmin(dim=-1)
        )

        combo_table = torch.tensor(
            self.combinations,
            dtype=torch.long,
            device=device,
        )

        best_subset = (
            combo_table[
                best_subset_indices
            ]
        )

        if previous_training:
            self.model.train()

        return {
            "subset_losses":
                subset_losses,

            "head_utility":
                head_utility,

            "teacher_target":
                teacher_target,

            "best_subset_indices":
                best_subset_indices,

            "best_subset":
                best_subset,

            "combination_table":
                combo_table,

            "reference_direct_indices_per_block":
                torch.stack(
                    references,
                    dim=1,
                ),
        }
