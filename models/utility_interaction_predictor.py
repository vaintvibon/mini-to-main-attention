import itertools

import torch
import torch.nn as nn


class UtilityInteractionPredictor(nn.Module):
    """
    Individual Utility + Pair Interaction predictor.

    Final pair score:
        score(i, j) = utility_i + utility_j + interaction(i, j)

    This keeps the original research semantics:
        1. each Mini Head has its own utility score
        2. pair interaction only corrects redundancy / synergy
        3. the best Direct pair is selected from the corrected pair scores

    H=4, K=2:
        (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    """

    def __init__(
        self,
        feature_dim: int,
        mini_heads: int = 4,
        direct_k: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        if direct_k != 2:
            raise ValueError(
                "UtilityInteractionPredictor v1 currently supports direct_k=2."
            )

        self.feature_dim = int(feature_dim)
        self.mini_heads = int(mini_heads)
        self.direct_k = int(direct_k)
        self.hidden_dim = int(hidden_dim)

        self.combinations = list(
            itertools.combinations(
                range(self.mini_heads),
                self.direct_k,
            )
        )

        self.num_combinations = len(self.combinations)

        self.register_buffer(
            "combination_table",
            torch.tensor(
                self.combinations,
                dtype=torch.long,
            ),
            persistent=False,
        )

        # ---------------------------------------------------------
        # Shared Mini Head encoder
        # ---------------------------------------------------------

        self.head_encoder = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(
                self.feature_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # Individual utility scorer
        #
        # head-local representation
        # + set mean
        # + set max
        # + relative(head - mean)
        # ---------------------------------------------------------

        utility_input_dim = (
            4 * hidden_dim
        )

        self.utility_scorer = nn.Sequential(
            nn.LayerNorm(
                utility_input_dim
            ),
            nn.Linear(
                utility_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        # ---------------------------------------------------------
        # Pair interaction scorer
        #
        # Must be symmetric:
        #   pair(i,j) == pair(j,i)
        #
        # Features:
        #   pair mean
        #   abs difference
        #   elementwise product
        #   set mean
        #   set max
        #   pair mean - set mean
        # ---------------------------------------------------------

        interaction_input_dim = (
            6 * hidden_dim
        )

        self.interaction_scorer = nn.Sequential(
            nn.LayerNorm(
                interaction_input_dim
            ),
            nn.Linear(
                interaction_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
        return_info: bool = False,
    ):
        """
        Parameters
        ----------
        features:
            [B,H,F]

        Returns
        -------
        pair_scores:
            [B,C]

        return_info=True additionally returns:
            utility_logits:     [B,H]
            interaction_scores: [B,C]
            utility_pair_scores:[B,C]
        """

        if features.dim() != 3:
            raise ValueError(
                f"Expected features [B,H,F], got {features.shape}"
            )

        B, H, F = features.shape

        if H != self.mini_heads:
            raise ValueError(
                f"Expected mini_heads={self.mini_heads}, got {H}"
            )

        if F != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {F}"
            )

        z = self.head_encoder(
            features
        )

        # ---------------------------------------------------------
        # Set context
        # ---------------------------------------------------------

        set_mean = z.mean(
            dim=1,
            keepdim=True,
        )

        set_max = z.max(
            dim=1,
            keepdim=True,
        ).values

        relative = (
            z
            -
            set_mean
        )

        utility_features = torch.cat(
            [
                z,
                set_mean.expand(
                    -1,
                    H,
                    -1,
                ),
                set_max.expand(
                    -1,
                    H,
                    -1,
                ),
                relative,
            ],
            dim=-1,
        )

        utility_logits = (
            self.utility_scorer(
                utility_features
            )
            .squeeze(-1)
        )

        # ---------------------------------------------------------
        # Pair scores
        # ---------------------------------------------------------

        utility_pair_scores = []
        raw_interactions = []

        set_mean_flat = (
            set_mean.squeeze(1)
        )

        set_max_flat = (
            set_max.squeeze(1)
        )

        for i, j in self.combinations:
            zi = z[:, i, :]
            zj = z[:, j, :]

            pair_mean = (
                zi + zj
            ) * 0.5

            pair_abs_diff = (
                zi - zj
            ).abs()

            pair_product = (
                zi * zj
            )

            pair_relative = (
                pair_mean
                -
                set_mean_flat
            )

            pair_feature = torch.cat(
                [
                    pair_mean,
                    pair_abs_diff,
                    pair_product,
                    set_mean_flat,
                    set_max_flat,
                    pair_relative,
                ],
                dim=-1,
            )

            interaction = (
                self.interaction_scorer(
                    pair_feature
                )
                .squeeze(-1)
            )

            raw_interactions.append(
                interaction
            )

            utility_pair_scores.append(
                utility_logits[:, i]
                +
                utility_logits[:, j]
            )

        utility_pair_scores = torch.stack(
            utility_pair_scores,
            dim=-1,
        )

        interaction_scores = torch.stack(
            raw_interactions,
            dim=-1,
        )

        # Remove common pair bias.
        # Interaction should express RELATIVE pair correction.
        interaction_scores = (
            interaction_scores
            -
            interaction_scores.mean(
                dim=-1,
                keepdim=True,
            )
        )

        pair_scores = (
            utility_pair_scores
            +
            interaction_scores
        )

        if return_info:
            return pair_scores, {
                "utility_logits":
                    utility_logits,

                "utility_probs":
                    torch.softmax(
                        utility_logits,
                        dim=-1,
                    ),

                "utility_pair_scores":
                    utility_pair_scores,

                "interaction_scores":
                    interaction_scores,

                "pair_probs":
                    torch.softmax(
                        pair_scores,
                        dim=-1,
                    ),
            }

        return pair_scores
