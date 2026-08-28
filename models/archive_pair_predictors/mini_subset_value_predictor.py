import itertools
import math
import torch
import torch.nn as nn


class MiniSubsetValuePredictor(nn.Module):
    """
    Predicts a value for each Direct subset directly.

    H=4, K=2 -> 6 outputs:
    (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

    Larger score means the subset is predicted to be better.
    """

    def __init__(
        self,
        mini_head_dim,
        mini_heads=4,
        direct_k=2,
        hidden_dim=64,
        dropout=0.0,
        has_cls_token=True,
        eps=1e-6,
    ):
        super().__init__()

        self.mini_head_dim = int(mini_head_dim)
        self.mini_heads = int(mini_heads)
        self.direct_k = int(direct_k)
        self.hidden_dim = int(hidden_dim)
        self.has_cls_token = bool(has_cls_token)
        self.eps = float(eps)

        self.combinations = list(
            itertools.combinations(
                range(self.mini_heads),
                self.direct_k,
            )
        )
        self.num_combinations = len(self.combinations)

        self.register_buffer(
            "combination_table",
            torch.tensor(self.combinations, dtype=torch.long),
            persistent=False,
        )

        # CLS + patch mean + entropy + max confidence
        self.local_feature_dim = 2 * self.mini_head_dim + 2

        self.local_encoder = nn.Sequential(
            nn.LayerNorm(self.local_feature_dim),
            nn.Linear(self.local_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # subset mean/max/min/spread/interaction
        # + global mean/max + relative mean = 8 * hidden
        self.subset_scorer = nn.Sequential(
            nn.LayerNorm(8 * hidden_dim),
            nn.Linear(8 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def extract_local_features(self, mini_contexts, mini_attn):
        if mini_contexts.dim() != 4:
            raise ValueError(
                f"Expected mini_contexts [B,H,N,Dh], got {mini_contexts.shape}"
            )
        if mini_attn.dim() != 4:
            raise ValueError(
                f"Expected mini_attn [B,H,N,M], got {mini_attn.shape}"
            )

        B, H, N, Dh = mini_contexts.shape

        if H != self.mini_heads:
            raise ValueError(
                f"Expected H={self.mini_heads}, got {H}"
            )
        if Dh != self.mini_head_dim:
            raise ValueError(
                f"Expected Dh={self.mini_head_dim}, got {Dh}"
            )
        if mini_attn.shape[:3] != (B, H, N):
            raise ValueError("Context/attention shape mismatch.")

        if self.has_cls_token:
            cls_context = mini_contexts[:, :, 0, :]
            patch_mean = (
                mini_contexts[:, :, 1:, :].mean(dim=2)
                if N > 1
                else cls_context
            )
        else:
            cls_context = mini_contexts.mean(dim=2)
            patch_mean = cls_context

        M = mini_attn.shape[-1]
        p = mini_attn.clamp_min(self.eps)

        entropy = -(p * p.log()).sum(dim=-1).mean(dim=-1)
        normalized_entropy = entropy / max(
            math.log(float(M)),
            self.eps,
        )

        max_confidence = (
            mini_attn.max(dim=-1).values.mean(dim=-1)
        )

        return torch.cat(
            [
                cls_context,
                patch_mean,
                normalized_entropy[..., None],
                max_confidence[..., None],
            ],
            dim=-1,
        )

    def forward_from_features(self, local_features):
        if local_features.dim() != 3:
            raise ValueError(
                f"Expected [B,H,F], got {local_features.shape}"
            )

        B, H, F = local_features.shape

        if H != self.mini_heads:
            raise ValueError(
                f"Expected H={self.mini_heads}, got {H}"
            )
        if F != self.local_feature_dim:
            raise ValueError(
                f"Expected F={self.local_feature_dim}, got {F}"
            )

        z = self.local_encoder(local_features)

        global_mean = z.mean(dim=1)
        global_max = z.max(dim=1).values

        scores = []

        for combo in self.combinations:
            selected = z[:, list(combo), :]

            subset_mean = selected.mean(dim=1)
            subset_max = selected.max(dim=1).values
            subset_min = selected.min(dim=1).values
            subset_spread = subset_max - subset_min

            if self.direct_k == 2:
                interaction = selected[:, 0, :] * selected[:, 1, :]
            else:
                centered = selected - subset_mean[:, None, :]
                interaction = centered.pow(2).mean(dim=1)

            relative_mean = subset_mean - global_mean

            feature = torch.cat(
                [
                    subset_mean,
                    subset_max,
                    subset_min,
                    subset_spread,
                    interaction,
                    global_mean,
                    global_max,
                    relative_mean,
                ],
                dim=-1,
            )

            scores.append(
                self.subset_scorer(feature).squeeze(-1)
            )

        return torch.stack(scores, dim=-1)

    def forward(self, mini_contexts, mini_attn):
        features = self.extract_local_features(
            mini_contexts,
            mini_attn,
        )
        return self.forward_from_features(features)
