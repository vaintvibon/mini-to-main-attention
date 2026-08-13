# models/mini_guided_block.py

from typing import Optional, Tuple

import torch
import torch.nn as nn

from models.two_level_attention import TwoLevelMiniMainAttention


class DropPath(nn.Module):
    """
    Stochastic Depth.
    timm 의존성을 줄이기 위해 직접 구현한다.

    drop_prob=0이면 identity처럼 동작한다.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    """
    ViT block에서 사용하는 기본 MLP.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MiniGuidedBlock(nn.Module):
    """
    Transformer block with Two-Level Mini-to-Main Attention.

    구조:
        x = x + TwoLevelMiniMainAttention(LN(x))
        x = x + MLP(LN(x))

    입력:
        x: [B, N, D]
        budget: active Main head 수
        patch_hw: patch token grid size, 예: (14, 14)

    출력:
        x: [B, N, D]
    """

    def __init__(
        self,
        dim: int,
        main_heads: int,
        mlp_ratio: float = 4.0,
        mini_heads: int = 1,
        mini_dim: int = 64,
        pool_ratio: int = 2,
        direct_ratio: float = 0.34,
        alpha_direct: float = 1.0,
        alpha_mixed: float = 0.2,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        allocator_hidden_dim: int = 128,
        has_cls_token: bool = True,
        gumbel_tau: float = 1.0,
        use_gumbel: bool = True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = TwoLevelMiniMainAttention(
            dim=dim,
            main_heads=main_heads,
            mini_heads=mini_heads,
            mini_dim=mini_dim,
            pool_ratio=pool_ratio,
            direct_ratio=direct_ratio,
            alpha_direct=alpha_direct,
            alpha_mixed=alpha_mixed,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            allocator_hidden_dim=allocator_hidden_dim,
            has_cls_token=has_cls_token,
            gumbel_tau=gumbel_tau,
            use_gumbel=use_gumbel,
        )

        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        budget: int,
        patch_hw: Optional[Tuple[int, int]] = None,
        return_info: bool = False,
    ):
        if return_info:
            attn_out, info = self.attn(
                self.norm1(x),
                budget=budget,
                patch_hw=patch_hw,
                return_info=True,
            )
            x = x + self.drop_path1(attn_out)
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
            return x, info

        attn_out = self.attn(
            self.norm1(x),
            budget=budget,
            patch_hw=patch_hw,
            return_info=False,
        )

        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))

        return x