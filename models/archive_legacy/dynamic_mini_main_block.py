COUNTERFACTUAL_API_VERSION = "cf_v2"

from typing import Optional

import torch
import torch.nn as nn

from models.dynamic_mini_main_attention import DynamicMiniMainAttention


class DropPath(nn.Module):
    def __init__(
        self,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        if not 0.0 <= drop_prob < 1.0:
            raise ValueError(
                f"drop_prob must be in [0, 1), got {drop_prob}"
            )

        self.drop_prob = float(
            drop_prob
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.drop_prob == 0.0
            or not self.training
        ):
            return x

        keep_prob = (
            1.0 - self.drop_prob
        )

        shape = (
            x.shape[0],
        ) + (
            1,
        ) * (
            x.ndim - 1
        )

        random_tensor = (
            keep_prob
            +
            torch.rand(
                shape,
                dtype=x.dtype,
                device=x.device,
            )
        )

        random_tensor.floor_()

        return (
            x
            / keep_prob
            * random_tensor
        )


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            dim,
            hidden_dim,
        )

        self.act = nn.GELU()

        self.drop1 = nn.Dropout(
            drop
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            dim,
        )

        self.drop2 = nn.Dropout(
            drop
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


class DynamicMiniMainBlock(nn.Module):
    def __init__(
        self,
        dim: int = 192,
        main_heads: int = 3,
        mini_heads: int = 4,
        mini_head_dim: int = 16,
        pool_ratio: int = 2,
        utility_hidden_dim: int = 64,
        utility_dropout: float = 0.0,
        direct_k: int = 2,
        mix_temperature: float = 1.0,
        bind_dim: int = 64,
        bind_temperature: float = 1.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        has_cls_token: bool = True,
    ):
        super().__init__()

        if dim <= 0:
            raise ValueError(
                f"dim must be > 0, got {dim}"
            )

        if mlp_ratio <= 0:
            raise ValueError(
                f"mlp_ratio must be > 0, got {mlp_ratio}"
            )

        self.norm1 = nn.LayerNorm(
            dim
        )

        self.attn = DynamicMiniMainAttention(
            dim=dim,
            main_heads=main_heads,
            mini_heads=mini_heads,
            mini_head_dim=mini_head_dim,
            pool_ratio=pool_ratio,
            utility_hidden_dim=utility_hidden_dim,
            utility_dropout=utility_dropout,
            direct_k=direct_k,
            mix_temperature=mix_temperature,
            bind_dim=bind_dim,
            bind_temperature=bind_temperature,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            has_cls_token=has_cls_token,
        )

        self.drop_path1 = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

        self.norm2 = nn.LayerNorm(
            dim
        )

        mlp_hidden_dim = int(
            dim * mlp_ratio
        )

        self.mlp = Mlp(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            drop=drop,
        )

        self.drop_path2 = (
            DropPath(drop_path)
            if drop_path > 0
            else nn.Identity()
        )

    def set_mix_temperature(
        self,
        temperature: float,
    ):
        self.attn.set_mix_temperature(
            temperature
        )

    def set_binding_temperature(
        self,
        temperature: float,
    ):
        self.attn.set_binding_temperature(
            temperature
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
        if collect_taylor and not return_info:
            raise ValueError(
                "collect_taylor=True requires return_info=True."
            )

        norm_x = self.norm1(
            x
        )

        if return_info:
            (
                attn_out,
                info,
            ) = self.attn(
                norm_x,
                patch_hw=patch_hw,
                return_info=True,
                collect_taylor=collect_taylor,
                forced_direct_indices=forced_direct_indices,
                forced_uniform_mix=forced_uniform_mix,
            )
        else:
            attn_out = self.attn(
                norm_x,
                patch_hw=patch_hw,
                return_info=False,
                collect_taylor=False,
                forced_direct_indices=forced_direct_indices,
                forced_uniform_mix=forced_uniform_mix,
            )

        x = (
            x
            +
            self.drop_path1(
                attn_out
            )
        )

        mlp_out = self.mlp(
            self.norm2(
                x
            )
        )

        x = (
            x
            +
            self.drop_path2(
                mlp_out
            )
        )

        if return_info:
            return (
                x,
                info,
            )

        return x
