from itertools import combinations
from typing import List

import torch


class BalancedDirectSubsetScheduler:
    """
    Stage-1 warm-up용 Balanced Direct routing scheduler.

    Utility Predictor를 사용하지 않고 가능한 Direct Mini Head 조합을
    균등하게 순환시킨다.

    예:
        mini_heads=4, direct_k=2

        가능한 조합:
        (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

    각 block과 sample이 이 조합들을 순환하도록 [B, direct_k] index를 만든다.
    """

    def __init__(
        self,
        mini_heads: int,
        direct_k: int,
    ):
        if mini_heads <= 0:
            raise ValueError(
                f"mini_heads must be > 0, got {mini_heads}"
            )

        if direct_k <= 0:
            raise ValueError(
                f"direct_k must be > 0, got {direct_k}"
            )

        if direct_k >= mini_heads:
            raise ValueError(
                "Balanced warm-up expects 0 < direct_k < mini_heads. "
                f"mini_heads={mini_heads}, direct_k={direct_k}"
            )

        self.mini_heads = mini_heads
        self.direct_k = direct_k

        self.combinations = list(
            combinations(
                range(mini_heads),
                direct_k,
            )
        )

        if len(self.combinations) == 0:
            raise RuntimeError(
                "No Direct subsets were generated."
            )

    @property
    def num_combinations(
        self,
    ) -> int:
        return len(
            self.combinations
        )

    def get_for_block(
        self,
        batch_size: int,
        step: int,
        block_idx: int = 0,
        device=None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        direct_indices:
            [B, direct_k]

        batch와 step을 함께 사용해서 combination을 순환한다.
        block_idx마다 offset을 두어 모든 block이 같은 조합만 보지 않게 한다.
        """

        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be > 0, got {batch_size}"
            )

        if step < 0:
            raise ValueError(
                f"step must be >= 0, got {step}"
            )

        if block_idx < 0:
            raise ValueError(
                f"block_idx must be >= 0, got {block_idx}"
            )

        combo_ids = (
            step * batch_size
            + torch.arange(
                batch_size,
                dtype=torch.long,
            )
            + block_idx
        ) % self.num_combinations

        table = torch.tensor(
            self.combinations,
            dtype=torch.long,
        )

        direct_indices = table[
            combo_ids
        ]

        if device is not None:
            direct_indices = (
                direct_indices.to(
                    device=device
                )
            )

        return direct_indices

    def get_for_all_blocks(
        self,
        batch_size: int,
        depth: int,
        step: int,
        device=None,
    ) -> List[torch.Tensor]:
        """
        각 Transformer block에 사용할 forced Direct subset을 반환한다.

        Returns:
            list length = depth
            각 원소 shape = [B, direct_k]
        """

        if depth <= 0:
            raise ValueError(
                f"depth must be > 0, got {depth}"
            )

        return [
            self.get_for_block(
                batch_size=batch_size,
                step=step,
                block_idx=block_idx,
                device=device,
            )
            for block_idx in range(depth)
        ]
