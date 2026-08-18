COUNTERFACTUAL_API_VERSION = "cf_v2"

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from models.dynamic_mini_main_block import DynamicMiniMainBlock


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()

        if img_size % patch_size != 0:
            raise ValueError(
                "img_size must be divisible by patch_size. "
                f"img_size={img_size}, patch_size={patch_size}"
            )

        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (
            img_size // patch_size,
            img_size // patch_size,
        )

        self.num_patches = (
            self.grid_size[0]
            * self.grid_size[1]
        )

        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):
        if x.dim() != 4:
            raise ValueError(
                "Expected image [B,C,H,W], "
                f"got {x.shape}"
            )

        _, _, H, W = x.shape

        if (
            H != self.img_size
            or W != self.img_size
        ):
            raise ValueError(
                f"Expected image size "
                f"{self.img_size}x{self.img_size}, "
                f"got {H}x{W}"
            )

        x = self.proj(
            x
        )

        x = (
            x
            .flatten(2)
            .transpose(1, 2)
        )

        return x


class DynamicMiniMainViT(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 4,
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
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()

        if depth <= 0:
            raise ValueError(
                f"depth must be > 0, got {depth}"
            )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.main_heads = main_heads
        self.mini_heads = mini_heads
        self.direct_k = direct_k

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.patch_hw = (
            self.patch_embed.grid_size
        )

        num_patches = (
            self.patch_embed.num_patches
        )

        self.cls_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim,
            )
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                1 + num_patches,
                embed_dim,
            )
        )

        self.pos_drop = nn.Dropout(
            drop_rate
        )

        if depth > 1:
            drop_path_values = (
                torch.linspace(
                    0,
                    drop_path_rate,
                    depth,
                )
                .tolist()
            )
        else:
            drop_path_values = [
                drop_path_rate
            ]

        self.blocks = nn.ModuleList(
            [
                DynamicMiniMainBlock(
                    dim=embed_dim,
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
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_values[i],
                    has_cls_token=True,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(
            embed_dim
        )

        self.head = nn.Linear(
            embed_dim,
            num_classes,
        )

        self._init_weights()

    def _init_weights(
        self,
    ):
        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )

        for module in self.modules():
            if isinstance(
                module,
                nn.Linear,
            ):
                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.Conv2d,
            ):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                )

                if module.bias is not None:
                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.LayerNorm,
            ):
                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    def set_mix_temperature(
        self,
        temperature: float,
    ):
        for block in self.blocks:
            block.set_mix_temperature(
                temperature
            )

    def set_binding_temperature(
        self,
        temperature: float,
    ):
        for block in self.blocks:
            block.set_binding_temperature(
                temperature
            )

    def _resolve_forced_indices(
        self,
        forced_direct_indices_per_block,
        block_idx: int,
    ):
        if forced_direct_indices_per_block is None:
            return None

        # 하나의 [B,K] tensor를 모든 block에 동일하게 적용할 수도 있다.
        if torch.is_tensor(
            forced_direct_indices_per_block
        ):
            return (
                forced_direct_indices_per_block
            )

        if not isinstance(
            forced_direct_indices_per_block,
            (list, tuple),
        ):
            raise TypeError(
                "forced_direct_indices_per_block must be None, "
                "a [B,K] Tensor, or a list/tuple of length depth."
            )

        if len(
            forced_direct_indices_per_block
        ) != self.depth:
            raise ValueError(
                "forced_direct_indices_per_block length mismatch. "
                f"Expected {self.depth}, "
                f"got {len(forced_direct_indices_per_block)}"
            )

        return (
            forced_direct_indices_per_block[
                block_idx
            ]
        )

    def forward_features(
        self,
        x: torch.Tensor,
        return_info: bool = False,
        collect_taylor: bool = False,
        forced_direct_indices_per_block=None,
        forced_uniform_mix: bool = False,
    ):
        if collect_taylor and not return_info:
            raise ValueError(
                "collect_taylor=True requires return_info=True."
            )

        B = x.shape[0]

        x = self.patch_embed(
            x
        )

        cls_token = (
            self.cls_token
            .expand(
                B,
                -1,
                -1,
            )
        )

        x = torch.cat(
            [
                cls_token,
                x,
            ],
            dim=1,
        )

        x = (
            x
            +
            self.pos_embed
        )

        x = self.pos_drop(
            x
        )

        block_info_list = []

        for block_idx, block in enumerate(
            self.blocks
        ):
            forced_indices = (
                self._resolve_forced_indices(
                    forced_direct_indices_per_block,
                    block_idx,
                )
            )

            if return_info:
                (
                    x,
                    block_info,
                ) = block(
                    x,
                    patch_hw=self.patch_hw,
                    return_info=True,
                    collect_taylor=collect_taylor,
                    forced_direct_indices=forced_indices,
                    forced_uniform_mix=forced_uniform_mix,
                )

                block_info_list.append(
                    block_info
                )

            else:
                x = block(
                    x,
                    patch_hw=self.patch_hw,
                    return_info=False,
                    collect_taylor=False,
                    forced_direct_indices=forced_indices,
                    forced_uniform_mix=forced_uniform_mix,
                )

        x = self.norm(
            x
        )

        cls = x[:, 0]

        if return_info:
            return (
                cls,
                block_info_list,
            )

        return cls

    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False,
        collect_taylor: bool = False,
        forced_direct_indices_per_block=None,
        forced_uniform_mix: bool = False,
    ):
        if collect_taylor and not return_info:
            raise ValueError(
                "collect_taylor=True requires return_info=True."
            )

        if return_info:
            (
                cls,
                block_info_list,
            ) = self.forward_features(
                x,
                return_info=True,
                collect_taylor=collect_taylor,
                forced_direct_indices_per_block=forced_direct_indices_per_block,
                forced_uniform_mix=forced_uniform_mix,
            )

            logits = self.head(
                cls
            )

            return (
                logits,
                block_info_list,
            )

        cls = self.forward_features(
            x,
            return_info=False,
            collect_taylor=False,
            forced_direct_indices_per_block=forced_direct_indices_per_block,
            forced_uniform_mix=forced_uniform_mix,
        )

        logits = self.head(
            cls
        )

        return logits
