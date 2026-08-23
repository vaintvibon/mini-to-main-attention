import itertools

import torch
import torch.nn as nn


class ContextualUtilityInteractionPredictor(nn.Module):
    """
    Diagnostic predictor that preserves the original decomposition:

        pair_score(i,j)
        =
        utility(i)
        +
        utility(j)
        +
        interaction(i,j)

    Differences from the current predictor:
      - each Mini head receives richer per-head features;
      - utility is conditioned on the current block state;
      - pair interaction receives explicit Mini-head relation features.

    This module is intended first as a feature-sufficiency probe.
    """

    def __init__(
        self,
        head_feature_dim,
        global_feature_dim,
        pair_feature_dim,
        mini_heads=4,
        direct_k=2,
        hidden_dim=64,
        dropout=0.0,
    ):
        super().__init__()

        if direct_k != 2:
            raise ValueError(
                "Current contextual predictor supports direct_k=2."
            )

        self.mini_heads = mini_heads
        self.direct_k = direct_k
        self.hidden_dim = hidden_dim

        pairs = list(
            itertools.combinations(
                range(mini_heads),
                direct_k,
            )
        )

        self.register_buffer(
            "pair_indices",
            torch.tensor(
                pairs,
                dtype=torch.long,
            ),
            persistent=False,
        )

        self.head_encoder = nn.Sequential(
            nn.Linear(
                head_feature_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim,
            ),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        self.global_encoder = nn.Sequential(
            nn.Linear(
                global_feature_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(
                hidden_dim,
            ),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
        )

        self.utility_scorer = nn.Sequential(
            nn.Linear(
                2 * hidden_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

        interaction_input_dim = (
            3 * hidden_dim
            +
            pair_feature_dim
        )

        self.interaction_scorer = nn.Sequential(
            nn.Linear(
                interaction_input_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def forward(
        self,
        head_features,
        global_features,
        pair_features,
        return_info=False,
    ):
        """
        head_features:
            [B, Hmini, Fh]

        global_features:
            [B, Fg]

        pair_features:
            [B, C(Hmini,2), Fp]
            pair order must match self.pair_indices.
        """

        head_encoded = self.head_encoder(
            head_features
        )

        global_encoded = self.global_encoder(
            global_features
        )

        global_per_head = global_encoded[
            :,
            None,
            :,
        ].expand(
            -1,
            self.mini_heads,
            -1,
        )

        utility_input = torch.cat(
            [
                head_encoded,
                global_per_head,
            ],
            dim=-1,
        )

        utility_logits = (
            self.utility_scorer(
                utility_input
            ).squeeze(
                -1
            )
        )

        pair_i = self.pair_indices[
            :,
            0,
        ]

        pair_j = self.pair_indices[
            :,
            1,
        ]

        hi = head_encoded[
            :,
            pair_i,
            :,
        ]

        hj = head_encoded[
            :,
            pair_j,
            :,
        ]

        # Symmetric pair representation.
        pair_sum = hi + hj
        pair_abs_diff = (
            hi - hj
        ).abs()

        global_per_pair = global_encoded[
            :,
            None,
            :,
        ].expand(
            -1,
            self.pair_indices.shape[0],
            -1,
        )

        interaction_input = torch.cat(
            [
                pair_sum,
                pair_abs_diff,
                global_per_pair,
                pair_features,
            ],
            dim=-1,
        )

        raw_interaction = (
            self.interaction_scorer(
                interaction_input
            ).squeeze(
                -1
            )
        )

        # Keep interaction as a correction term rather than a
        # free global pair bias.
        interaction_scores = (
            raw_interaction
            -
            raw_interaction.mean(
                dim=-1,
                keepdim=True,
            )
        )

        pair_scores = (
            utility_logits[
                :,
                pair_i,
            ]
            +
            utility_logits[
                :,
                pair_j,
            ]
            +
            interaction_scores
        )

        if not return_info:
            return pair_scores

        return pair_scores, {
            "utility_logits":
                utility_logits,

            "interaction_scores":
                interaction_scores,

            "head_encoded":
                head_encoded,

            "global_encoded":
                global_encoded,
        }
