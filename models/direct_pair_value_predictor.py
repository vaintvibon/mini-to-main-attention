import itertools

import torch
import torch.nn as nn


class DirectPairValuePredictor(nn.Module):
    """
    Directly scores the six possible Mini-head pairs.

    Unlike Utility+Interaction, this model does NOT require

        score(i,j) = utility(i) + utility(j) + interaction(i,j)

    Instead, each pair gets one learned value from the current Mini-head set.

    Input:
        head_features: [B, Hmini, F]

    Output:
        pair_scores: [B, C(Hmini,2)]

    Higher score = lower predicted continuation cost.
    """

    def __init__(
        self,
        feature_dim,
        mini_heads=4,
        direct_k=2,
        hidden_dim=64,
        dropout=0.0,
    ):
        super().__init__()

        if direct_k != 2:
            raise ValueError(
                "DirectPairValuePredictor currently supports direct_k=2."
            )

        self.mini_heads = mini_heads
        self.direct_k = direct_k

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
                feature_dim,
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

        # Pair representation:
        #   pair sum
        #   pair absolute difference
        #   whole-set mean context
        pair_input_dim = 3 * hidden_dim

        self.pair_scorer = nn.Sequential(
            nn.Linear(
                pair_input_dim,
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
        return_info=False,
    ):
        encoded = self.head_encoder(
            head_features
        )

        set_context = encoded.mean(
            dim=1
        )

        pair_i = self.pair_indices[
            :,
            0,
        ]

        pair_j = self.pair_indices[
            :,
            1,
        ]

        hi = encoded[
            :,
            pair_i,
            :,
        ]

        hj = encoded[
            :,
            pair_j,
            :,
        ]

        pair_sum = hi + hj
        pair_abs_diff = (
            hi - hj
        ).abs()

        set_per_pair = set_context[
            :,
            None,
            :,
        ].expand(
            -1,
            self.pair_indices.shape[0],
            -1,
        )

        pair_input = torch.cat(
            [
                pair_sum,
                pair_abs_diff,
                set_per_pair,
            ],
            dim=-1,
        )

        pair_scores = (
            self.pair_scorer(
                pair_input
            ).squeeze(
                -1
            )
        )

        if not return_info:
            return pair_scores

        return pair_scores, {
            "head_encoded":
                encoded,

            "set_context":
                set_context,
        }


class ContextualDirectPairValuePredictor(nn.Module):
    """
    Direct pair-value predictor with the richer contextual features
    produced by the previous Block-0 feature probe.

    Inputs:
        head_features:   [B, Hmini, Fh]
        global_features: [B, Fg]
        pair_features:   [B, C(Hmini,2), Fp]

    Output:
        pair_scores: [B, C(Hmini,2)]
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
                "ContextualDirectPairValuePredictor currently supports direct_k=2."
            )

        self.mini_heads = mini_heads
        self.direct_k = direct_k

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

        # pair sum + pair abs diff + set mean + global + explicit pair relations
        pair_input_dim = (
            4 * hidden_dim
            +
            pair_feature_dim
        )

        self.pair_scorer = nn.Sequential(
            nn.Linear(
                pair_input_dim,
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
        encoded = self.head_encoder(
            head_features
        )

        global_encoded = self.global_encoder(
            global_features
        )

        set_context = encoded.mean(
            dim=1
        )

        pair_i = self.pair_indices[
            :,
            0,
        ]

        pair_j = self.pair_indices[
            :,
            1,
        ]

        hi = encoded[
            :,
            pair_i,
            :,
        ]

        hj = encoded[
            :,
            pair_j,
            :,
        ]

        pair_sum = hi + hj
        pair_abs_diff = (
            hi - hj
        ).abs()

        set_per_pair = set_context[
            :,
            None,
            :,
        ].expand(
            -1,
            self.pair_indices.shape[0],
            -1,
        )

        global_per_pair = global_encoded[
            :,
            None,
            :,
        ].expand(
            -1,
            self.pair_indices.shape[0],
            -1,
        )

        pair_input = torch.cat(
            [
                pair_sum,
                pair_abs_diff,
                set_per_pair,
                global_per_pair,
                pair_features,
            ],
            dim=-1,
        )

        pair_scores = (
            self.pair_scorer(
                pair_input
            ).squeeze(
                -1
            )
        )

        if not return_info:
            return pair_scores

        return pair_scores, {
            "head_encoded":
                encoded,

            "global_encoded":
                global_encoded,

            "set_context":
                set_context,
        }
