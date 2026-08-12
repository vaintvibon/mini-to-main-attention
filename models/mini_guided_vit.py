# models/mini_guided_vit.py

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from models.mini_guided_block import MiniGuidedBlock


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.

    입력:
        x: [B, 3, H, W]

    출력:
        x: [B, num_patches, embed_dim]
    """

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
                f"img_size must be divisible by patch_size. "
                f"Got img_size={img_size}, patch_size={patch_size}."
            )

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected image input [B, C, H, W], got {x.shape}.")

        B, C, H, W = x.shape

        if H != self.img_size or W != self.img_size:
            raise ValueError(
                f"Expected image size {self.img_size}x{self.img_size}, "
                f"but got {H}x{W}."
            )

        x = self.proj(x)              # [B, D, Hp, Wp]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]

        return x


class MiniGuidedViT(nn.Module):
    """
    Mini-to-Main Two-Level Head Scheduler 기반 Vision Transformer.

    구조:
        image
        → patch embedding
        → cls token 추가
        → position embedding 추가
        → MiniGuidedBlock x depth
        → LayerNorm
        → cls token classifier

    주의:
        이 모델은 DeiT-tiny scale을 참고한 독립 구현이다.
        아직 timm DeiT pretrained weight를 자동 로딩하지 않는다.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 192,
        depth: int = 12,
        main_heads: int = 3,
        mlp_ratio: float = 4.0,
        mini_heads: int = 1,
        mini_dim: int = 64,
        pool_ratio: int = 2,
        direct_ratio: float = 0.34,
        alpha_direct: float = 1.0,
        alpha_mixed: float = 0.2,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        allocator_hidden_dim: int = 128,
    ):
        super().__init__()

        if embed_dim % main_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by main_heads. "
                f"Got embed_dim={embed_dim}, main_heads={main_heads}."
            )

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.main_heads = main_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches
        self.patch_hw = self.patch_embed.grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # layer별 drop_path 값
        if depth > 1:
            dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        else:
            dpr = [drop_path_rate]

        self.blocks = nn.ModuleList(
            [
                MiniGuidedBlock(
                    dim=embed_dim,
                    main_heads=main_heads,
                    mlp_ratio=mlp_ratio,
                    mini_heads=mini_heads,
                    mini_dim=mini_dim,
                    pool_ratio=pool_ratio,
                    direct_ratio=direct_ratio,
                    alpha_direct=alpha_direct,
                    alpha_mixed=alpha_mixed,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    allocator_hidden_dim=allocator_hidden_dim,
                    has_cls_token=True,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(
        self,
        x: torch.Tensor,
        budget: int,
        return_info: bool = False,
    ):
        B = x.shape[0]

        x = self.patch_embed(x)  # [B, num_patches, D]

        cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_token, x), dim=1)          # [B, 1+num_patches, D]

        x = x + self.pos_embed
        x = self.pos_drop(x)

        info_list = []

        for block in self.blocks:
            if return_info:
                x, info = block(
                    x,
                    budget=budget,
                    patch_hw=self.patch_hw,
                    return_info=True,
                )
                info_list.append(info)
            else:
                x = block(
                    x,
                    budget=budget,
                    patch_hw=self.patch_hw,
                    return_info=False,
                )

        x = self.norm(x)

        cls = x[:, 0]  # [B, D]

        if return_info:
            return cls, info_list

        return cls

    def forward(
        self,
        x: torch.Tensor,
        budget: int,
        return_info: bool = False,
    ):
        if return_info:
            cls, info_list = self.forward_features(
                x,
                budget=budget,
                return_info=True,
            )
            logits = self.head(cls)

            return logits, info_list

        cls = self.forward_features(
            x,
            budget=budget,
            return_info=False,
        )
        logits = self.head(cls)

        return logits