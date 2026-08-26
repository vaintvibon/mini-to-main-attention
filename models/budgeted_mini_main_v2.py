# -*- coding: utf-8 -*-
"""Budgeted Mini->Main Attention v2.

핵심:
- Mini 4 heads는 항상 먼저 실행되어 base representation을 만든다.
- Mini utility + Mini->Main binding으로 필요한 Main heads를 정한다.
- B=0은 true Mini-only attention path.
- B>=2에서는 selected Main heads만 추가 계산한다.
- train: router gradient를 위해 dense Main + Straight-Through gate.
- eval: selected sample/head의 Q/K/V/attention/output projection만 실제 계산.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, device=x.device, dtype=x.dtype)
        mask.floor_()
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(f"Expected {self.img_size}x{self.img_size}, got {x.shape[-2:]}")
        return self.proj(x).flatten(2).transpose(1, 2)


class MultiMiniAttentionV2(nn.Module):
    """Q=all tokens, K/V=pooled tokens. Output keeps Mini head identity."""
    def __init__(
        self,
        dim: int,
        mini_heads: int = 4,
        mini_head_dim: int = 16,
        pool_ratio: int = 2,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.mini_heads = mini_heads
        self.mini_head_dim = mini_head_dim
        self.pool_ratio = pool_ratio
        total = mini_heads * mini_head_dim
        self.q_proj = nn.Linear(dim, total, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, total, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, total, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = mini_head_dim ** -0.5

    def _pool_tokens(self, x, patch_hw: Optional[Tuple[int, int]]):
        cls, patch = x[:, :1], x[:, 1:]
        if patch_hw is None:
            return torch.cat([cls, patch[:, :: self.pool_ratio]], dim=1)
        ph, pw = patch_hw
        if patch.shape[1] != ph * pw:
            raise ValueError("patch_hw mismatch")
        B, _, D = patch.shape
        patch = patch.reshape(B, ph, pw, D).permute(0, 3, 1, 2)
        k = min(self.pool_ratio, ph, pw)
        patch = F.avg_pool2d(patch, kernel_size=k, stride=k)
        patch = patch.flatten(2).transpose(1, 2)
        return torch.cat([cls, patch], dim=1)

    def forward(self, x, patch_hw=None):
        B, N, _ = x.shape
        pooled = self._pool_tokens(x, patch_hw)
        M = pooled.shape[1]
        q = self.q_proj(x).reshape(B, N, self.mini_heads, self.mini_head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(pooled).reshape(B, M, self.mini_heads, self.mini_head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(pooled).reshape(B, M, self.mini_heads, self.mini_head_dim).permute(0, 2, 1, 3)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        return attn @ v, attn


class MiniUtilityV2(nn.Module):
    def __init__(self, mini_head_dim: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        feat_dim = 2 * mini_head_dim + 2
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, contexts, attn, return_features=False):
        cls = contexts[:, :, 0]
        patch_mean = contexts[:, :, 1:].mean(dim=2) if contexts.shape[2] > 1 else cls
        M = attn.shape[-1]
        entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1)
        entropy = entropy / max(math.log(max(M, 2)), 1e-8)
        entropy = entropy.mean(dim=-1, keepdim=True)
        max_conf = attn.max(dim=-1).values.mean(dim=-1, keepdim=True)
        feats = torch.cat([cls, patch_mean, entropy, max_conf], dim=-1)
        logits = self.scorer(feats).squeeze(-1)
        if not return_features:
            return logits
        return logits, {
            "utility_features": feats,
            "attention_entropy": entropy.squeeze(-1),
            "max_confidence": max_conf.squeeze(-1),
        }


def _relaxed_topk(scores: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(scores)
    relaxed = torch.zeros_like(scores)
    for _ in range(k):
        remaining = (1.0 - relaxed).clamp_min(1e-6)
        probs = (scores / tau + remaining.log()).softmax(dim=-1)
        relaxed = relaxed + probs
    return relaxed


def _hard_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    idx = torch.topk(scores, k=k, dim=-1).indices
    out = torch.zeros_like(scores, dtype=torch.bool)
    out.scatter_(1, idx, True)
    return out


def _st_khot(scores: torch.Tensor, k: int, tau: float):
    hard = _hard_topk(scores, k)
    soft = _relaxed_topk(scores, k, tau)
    gate = hard.to(scores.dtype) + soft - soft.detach()
    return hard, gate


class MiniBindingBudgetRouter(nn.Module):
    def __init__(
        self,
        mini_heads: int,
        mini_head_dim: int,
        main_heads: int,
        main_head_dim: int,
        direct_k: int = 2,
        bind_dim: int = 64,
        hidden_dim: int = 128,
        bind_tau: float = 1.0,
        route_tau: float = 1.0,
    ):
        super().__init__()
        if direct_k > min(mini_heads, main_heads):
            raise ValueError("invalid direct_k")
        self.mini_heads = mini_heads
        self.mini_head_dim = mini_head_dim
        self.main_heads = main_heads
        self.main_head_dim = main_head_dim
        self.direct_k = direct_k
        self.bind_tau = bind_tau
        self.route_tau = route_tau

        self.desc_proj = nn.Linear(2 * mini_head_dim, bind_dim)
        self.main_slots = nn.Parameter(torch.empty(main_heads, bind_dim))
        nn.init.trunc_normal_(self.main_slots, std=0.02)
        self.direct_proj = nn.Linear(mini_head_dim, main_head_dim)
        self.mixed_proj = nn.ModuleList(
            [nn.Linear(mini_head_dim, main_head_dim) for _ in range(main_heads)]
        )
        router_in = 2 * mini_heads * mini_head_dim + mini_heads
        self.need_router = nn.Sequential(
            nn.Linear(router_in, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, main_heads)
        )

    @staticmethod
    def _gather_ctx(contexts, idx):
        B, _, N, D = contexts.shape
        g = idx[:, None, None, None].expand(B, 1, N, D)
        return contexts.gather(1, g).squeeze(1)

    @staticmethod
    def _gather_desc(desc, idx):
        B, _, D = desc.shape
        g = idx[:, None, None].expand(B, 1, D)
        return desc.gather(1, g).squeeze(1)

    def forward(self, mini_contexts, utility_logits, budget: int, training_st: bool):
        B, Hm, N, _ = mini_contexts.shape
        if budget < 0 or budget > self.main_heads:
            raise ValueError("invalid budget")
        if budget != 0 and budget < self.direct_k:
            raise ValueError(f"budget must be 0 or >= direct_k={self.direct_k}")

        cls = mini_contexts[:, :, 0]
        patch_mean = mini_contexts[:, :, 1:].mean(dim=2) if N > 1 else cls
        utility_probs = utility_logits.softmax(dim=-1)
        global_feat = torch.cat(
            [cls.reshape(B, -1), patch_mean.reshape(B, -1), utility_probs], dim=-1
        )
        need_scores = self.need_router(global_feat)

        if budget == 0:
            zero_seed = mini_contexts.new_zeros(B, self.main_heads, N, self.main_head_dim)
            zero_mask = torch.zeros(B, self.main_heads, dtype=torch.bool, device=mini_contexts.device)
            return zero_seed, {
                "direct_indices": torch.empty(B, 0, dtype=torch.long, device=mini_contexts.device),
                "binding_hard": torch.empty(B, 0, self.main_heads, dtype=torch.bool, device=mini_contexts.device),
                "bound_main_mask": zero_mask,
                "active_main_mask": zero_mask,
                "active_main_gate": mini_contexts.new_zeros(B, self.main_heads),
                "need_scores": need_scores,
                "mixed_weights": utility_probs,
            }

        direct_indices = torch.topk(utility_logits, k=self.direct_k, dim=-1).indices
        direct_mask = torch.zeros(B, Hm, dtype=torch.bool, device=mini_contexts.device)
        direct_mask.scatter_(1, direct_indices, True)
        rem_logits = utility_logits.masked_fill(direct_mask, torch.finfo(utility_logits.dtype).min,)
        mixed_weights = rem_logits.softmax(dim=-1)
        mixed_context = (mini_contexts * mixed_weights[:, :, None, None]).sum(dim=1)

        desc = self.desc_proj(torch.cat([cls, patch_mean], dim=-1))
        desc = F.normalize(desc, dim=-1)
        slots = F.normalize(self.main_slots, dim=-1)

        hard_used = torch.zeros(B, self.main_heads, dtype=torch.bool, device=mini_contexts.device)
        binding_hard_list = []
        direct_seed_sum = mini_contexts.new_zeros(B, self.main_heads, N, self.main_head_dim)

        for rank in range(self.direct_k):
            mini_idx = direct_indices[:, rank]
            d = self._gather_desc(desc, mini_idx)
            compat = (d @ slots.t()).masked_fill(hard_used, torch.finfo(utility_logits.dtype).min,)
            soft = (compat / self.bind_tau).softmax(dim=-1)
            hard_idx = compat.argmax(dim=-1)
            hard = F.one_hot(hard_idx, num_classes=self.main_heads).to(soft.dtype)
            assign = hard + soft - soft.detach() if training_st else hard
            hard_bool = hard.bool()
            hard_used = hard_used | hard_bool

            c = self.direct_proj(self._gather_ctx(mini_contexts, mini_idx))
            direct_seed_sum = direct_seed_sum + assign[:, :, None, None] * c[:, None]
            binding_hard_list.append(hard_bool)

        binding_hard = torch.stack(binding_hard_list, dim=1)
        bound_main_mask = hard_used
        mixed_seeds = torch.stack([proj(mixed_context) for proj in self.mixed_proj], dim=1)
        main_seeds = direct_seed_sum + (~bound_main_mask)[:, :, None, None].to(mixed_seeds.dtype) * mixed_seeds

        extra_k = budget - self.direct_k
        if budget == self.main_heads:
            active_hard = torch.ones_like(bound_main_mask)
            active_gate = torch.ones(B, self.main_heads, dtype=mini_contexts.dtype, device=mini_contexts.device)
        elif extra_k == 0:
            active_hard = bound_main_mask
            active_gate = bound_main_mask.to(mini_contexts.dtype)
        else:
            available_scores = need_scores.masked_fill(bound_main_mask, torch.finfo(utility_logits.dtype).min,)
            hard_extra, st_extra = _st_khot(available_scores, extra_k, self.route_tau)
            hard_extra = hard_extra & (~bound_main_mask)
            active_hard = bound_main_mask | hard_extra
            active_gate = (
                bound_main_mask.to(st_extra.dtype)
                + (~bound_main_mask).to(st_extra.dtype) * st_extra
                if training_st
                else active_hard.to(st_extra.dtype)
            )

        if not torch.all(active_hard.sum(dim=-1) == budget):
            raise RuntimeError("active count != budget")

        return main_seeds, {
            "direct_indices": direct_indices,
            "binding_hard": binding_hard,
            "bound_main_mask": bound_main_mask,
            "active_main_mask": active_hard,
            "active_main_gate": active_gate,
            "need_scores": need_scores,
            "mixed_weights": mixed_weights,
            "mixed_context": mixed_context,
        }


class XBudgetRouter(nn.Module):
    """Main-only baseline router using current block input, not Mini."""
    def __init__(self, dim: int, main_heads: int, hidden_dim: int = 128, route_tau: float = 1.0):
        super().__init__()
        self.main_heads = main_heads
        self.route_tau = route_tau
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, main_heads)
        )

    def forward(self, x, budget: int, training_st: bool):
        if budget <= 0 or budget > self.main_heads:
            raise ValueError("main_only budget must be 1..main_heads")
        cls = x[:, 0]
        mean = x[:, 1:].mean(dim=1) if x.shape[1] > 1 else cls
        scores = self.net(torch.cat([cls, mean], dim=-1))
        if budget == self.main_heads:
            hard = torch.ones_like(scores, dtype=torch.bool)
            gate = torch.ones_like(scores)
        else:
            hard, st = _st_khot(scores, budget, self.route_tau)
            gate = st if training_st else hard.to(scores.dtype)
        return hard, gate, scores


class SelectiveSeededMainAttention(nn.Module):
    """Per-head projection allows true sparse eval."""
    def __init__(self, dim: int, main_heads: int = 8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % main_heads != 0:
            raise ValueError("dim must be divisible by main_heads")
        self.dim = dim
        self.main_heads = main_heads
        self.head_dim = dim // main_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.ModuleList([nn.Linear(dim, self.head_dim, bias=qkv_bias) for _ in range(main_heads)])
        self.k_proj = nn.ModuleList([nn.Linear(dim, self.head_dim, bias=qkv_bias) for _ in range(main_heads)])
        self.v_proj = nn.ModuleList([nn.Linear(dim, self.head_dim, bias=qkv_bias) for _ in range(main_heads)])
        self.out_proj = nn.ModuleList([nn.Linear(self.head_dim, dim, bias=False) for _ in range(main_heads)])
        self.out_bias = nn.Parameter(torch.zeros(dim))
        self.seed_scale = nn.Parameter(torch.ones(main_heads))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _one_head(self, x, seed, h):
        q = self.q_proj[h](x) + self.seed_scale[h] * seed
        k = self.k_proj[h](x)
        v = self.v_proj[h](x)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        return self.out_proj[h](attn @ v)

    def forward_dense(self, x, seeds, active_gate, budget, return_info=False):
        B = x.shape[0]
        if budget == 0:
            out = torch.zeros_like(x)
            info = {"computed_sample_heads": 0, "effective_active_sample_heads": 0, "dense_training": True}
            return (out, info) if return_info else out
        out = torch.zeros_like(x)
        for h in range(self.main_heads):
            out = out + self._one_head(x, seeds[:, h], h) * active_gate[:, h, None, None]
        out = self.proj_drop(out + self.out_bias[None, None])
        info = {
            "computed_sample_heads": B * self.main_heads,
            "effective_active_sample_heads": int((active_gate.detach() > 0.5).sum()),
            "dense_training": True,
        }
        return (out, info) if return_info else out

    def forward_sparse(self, x, seeds, active_mask, budget, return_info=False):
        B = x.shape[0]
        if budget == 0:
            out = torch.zeros_like(x)
            info = {"computed_sample_heads": 0, "effective_active_sample_heads": 0, "dense_training": False}
            return (out, info) if return_info else out
        out = torch.zeros_like(x)
        computed = 0
        for h in range(self.main_heads):
            idx = torch.nonzero(active_mask[:, h], as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            contrib = self._one_head(x.index_select(0, idx), seeds[:, h].index_select(0, idx), h)
            out.index_add_(0, idx, contrib)
            computed += int(idx.numel())
        out = self.proj_drop(out + self.out_bias[None, None])
        if computed != B * budget:
            raise RuntimeError(f"computed={computed}, expected={B*budget}")
        info = {"computed_sample_heads": computed, "effective_active_sample_heads": int(active_mask.sum()), "dense_training": False}
        return (out, info) if return_info else out

    def forward(self, x, seeds, active_mask, active_gate, budget, force_dense=False, return_info=False):
        if self.training or force_dense:
            return self.forward_dense(x, seeds, active_gate, budget, return_info)
        return self.forward_sparse(x, seeds, active_mask, budget, return_info)


class BudgetedMiniMainAttentionV2(nn.Module):
    def __init__(
        self,
        dim=192,
        main_heads=8,
        mini_heads=4,
        mini_head_dim=16,
        direct_k=2,
        pool_ratio=2,
        mode="mini_main",
        utility_hidden_dim=64,
        router_hidden_dim=128,
        bind_dim=64,
        route_tau=1.0,
        bind_tau=1.0,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if mode not in {"mini_main", "main_only"}:
            raise ValueError("mode must be mini_main or main_only")
        self.mode = mode
        self.dim = dim
        self.main_heads = main_heads
        self.mini_heads = mini_heads
        self.mini_head_dim = mini_head_dim
        self.direct_k = direct_k

        self.main = SelectiveSeededMainAttention(dim, main_heads, qkv_bias, attn_drop, proj_drop)

        if mode == "mini_main":
            self.mini = MultiMiniAttentionV2(dim, mini_heads, mini_head_dim, pool_ratio, qkv_bias, attn_drop)
            self.utility = MiniUtilityV2(mini_head_dim, utility_hidden_dim, proj_drop)
            self.mini_base_proj = nn.Linear(mini_heads * mini_head_dim, dim)
            self.mini_base_scale = nn.Parameter(torch.tensor(1.0))
            self.router = MiniBindingBudgetRouter(
                mini_heads, mini_head_dim, main_heads, dim // main_heads,
                direct_k, bind_dim, router_hidden_dim, bind_tau, route_tau,
            )
            self.x_router = None
        else:
            self.mini = None
            self.utility = None
            self.mini_base_proj = None
            self.register_parameter("mini_base_scale", None)
            self.router = None
            self.x_router = XBudgetRouter(dim, main_heads, router_hidden_dim, route_tau)

    def set_route_temperature(self, tau):
        if self.mode == "mini_main":
            self.router.route_tau = float(tau)
        else:
            self.x_router.route_tau = float(tau)

    def forward(self, x, budget: int, patch_hw=None, force_dense_main=False, return_info=False):
        B, N, _ = x.shape
        if self.mode == "mini_main":
            if budget != 0 and budget < self.direct_k:
                raise ValueError(f"budget must be 0 or >= {self.direct_k}")
            mini_contexts, mini_attn = self.mini(x, patch_hw)
            utility_logits, utility_info = self.utility(mini_contexts, mini_attn, return_features=True)
            mini_cat = mini_contexts.transpose(1, 2).reshape(B, N, self.mini_heads * self.mini_head_dim)
            mini_base = self.mini_base_proj(mini_cat)
            seeds, route_info = self.router(mini_contexts, utility_logits, budget, self.training)
            main_out, main_info = self.main(
                x, seeds,
                route_info["active_main_mask"], route_info["active_main_gate"],
                budget, force_dense_main, True,
            )
            out = self.mini_base_scale * mini_base + main_out
            if return_info:
                return out, {
                    "mode": self.mode,
                    "mini_contexts": mini_contexts,
                    "mini_attn": mini_attn,
                    "mini_base": mini_base,
                    "mini_base_scale": self.mini_base_scale.detach(),
                    "utility_logits": utility_logits,
                    **utility_info,
                    **route_info,
                    **main_info,
                }
            return out

        if budget <= 0:
            raise ValueError("main_only does not support B=0")
        active_mask, active_gate, scores = self.x_router(x, budget, self.training)
        seeds = x.new_zeros(B, self.main_heads, N, self.dim // self.main_heads)
        main_out, main_info = self.main(
            x, seeds, active_mask, active_gate, budget, force_dense_main, True
        )
        if return_info:
            return main_out, {
                "mode": self.mode,
                "active_main_mask": active_mask,
                "active_main_gate": active_gate,
                "need_scores": scores,
                **main_info,
            }
        return main_out


class BudgetedMiniMainBlockV2(nn.Module):
    def __init__(
        self,
        dim=192,
        main_heads=8,
        mini_heads=4,
        mini_head_dim=16,
        direct_k=2,
        pool_ratio=2,
        mode="mini_main",
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        route_tau=1.0,
        bind_tau=1.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BudgetedMiniMainAttentionV2(
            dim, main_heads, mini_heads, mini_head_dim, direct_k, pool_ratio,
            mode, route_tau=route_tau, bind_tau=bind_tau,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.dp1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.dp2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def set_route_temperature(self, tau):
        self.attn.set_route_temperature(tau)

    def forward(self, x, budget, patch_hw=None, force_dense_main=False, return_info=False):
        if return_info:
            a, info = self.attn(self.norm1(x), budget, patch_hw, force_dense_main, True)
            x = x + self.dp1(a)
            x = x + self.dp2(self.mlp(self.norm2(x)))
            return x, info
        a = self.attn(self.norm1(x), budget, patch_hw, force_dense_main, False)
        x = x + self.dp1(a)
        x = x + self.dp2(self.mlp(self.norm2(x)))
        return x


class BudgetedMiniMainViTV2(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=192,
        depth=4,
        main_heads=8,
        mini_heads=4,
        mini_head_dim=16,
        direct_k=2,
        pool_ratio=2,
        mode="mini_main",
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        route_tau=1.0,
        bind_tau=1.0,
    ):
        super().__init__()
        if embed_dim % main_heads != 0:
            raise ValueError("embed_dim must be divisible by main_heads")
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.main_heads = main_heads
        self.mini_heads = mini_heads
        self.mini_head_dim = mini_head_dim
        self.direct_k = direct_k
        self.pool_ratio = pool_ratio
        self.mode = mode

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_hw = self.patch_embed.grid_size
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+n, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist() if depth > 1 else [drop_path_rate]
        self.blocks = nn.ModuleList([
            BudgetedMiniMainBlockV2(
                embed_dim, main_heads, mini_heads, mini_head_dim, direct_k,
                pool_ratio, mode, mlp_ratio, drop_rate, attn_drop_rate,
                dpr[i], route_tau, bind_tau,
            ) for i in range(depth)
        ])
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

    def set_route_temperature(self, tau):
        for block in self.blocks:
            block.set_route_temperature(tau)

    def forward_features(self, x, budget, force_dense_main=False, return_info=False):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        infos = []
        for block in self.blocks:
            if return_info:
                x, info = block(x, budget, self.patch_hw, force_dense_main, True)
                infos.append(info)
            else:
                x = block(x, budget, self.patch_hw, force_dense_main, False)
        cls = self.norm(x)[:, 0]
        return (cls, infos) if return_info else cls

    def forward(self, x, budget, force_dense_main=False, return_info=False):
        if return_info:
            cls, infos = self.forward_features(x, budget, force_dense_main, True)
            return self.head(cls), infos
        return self.head(self.forward_features(x, budget, force_dense_main, False))

    def estimate_block_attention_macs(self, budget: int) -> Dict[str, float]:
        D, H = self.embed_dim, self.main_heads
        Dh = D // H
        N = 1 + self.patch_embed.num_patches
        main_per_head = 3*N*D*Dh + 2*N*N*Dh + N*Dh*D
        main = budget * main_per_head
        mini = 0.0
        if self.mode == "mini_main":
            Hm, Dm = self.mini_heads, self.mini_head_dim
            ph, pw = self.patch_hw
            pooled_h = max(1, ph // self.pool_ratio)
            pooled_w = max(1, pw // self.pool_ratio)
            M = 1 + pooled_h * pooled_w
            mini = (
                N*D*(Hm*Dm)
                + 2*M*D*(Hm*Dm)
                + 2*Hm*N*M*Dm
                + N*(Hm*Dm)*D
            )
        return {
            "mini_macs": float(mini),
            "main_macs": float(main),
            "attention_total_macs": float(mini + main),
        }
