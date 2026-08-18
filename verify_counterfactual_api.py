import inspect
import os

import models.dynamic_mini_main_attention as attn_mod
import models.dynamic_mini_main_block as block_mod
import models.dynamic_mini_main_vit as vit_mod

from models.dynamic_mini_main_attention import DynamicMiniMainAttention
from models.dynamic_mini_main_block import DynamicMiniMainBlock
from models.dynamic_mini_main_vit import DynamicMiniMainViT


EXPECTED = "cf_v2"


def show_module(name, module):
    version = getattr(
        module,
        "COUNTERFACTUAL_API_VERSION",
        None,
    )

    print(f"{name} version: {version}")
    print(f"{name} file: {os.path.abspath(module.__file__)}")

    assert version == EXPECTED, (
        f"{name} is not the v2 counterfactual file. "
        f"Expected {EXPECTED}, got {version}. "
        f"Loaded from: {module.__file__}"
    )


print("[VERIFY] Counterfactual API v2")

show_module("Attention", attn_mod)
show_module("Block", block_mod)
show_module("ViT", vit_mod)

attn_sig = inspect.signature(
    DynamicMiniMainAttention.forward
)

block_sig = inspect.signature(
    DynamicMiniMainBlock.forward
)

vit_sig = inspect.signature(
    DynamicMiniMainViT.forward
)

print("\nAttention.forward:")
print(attn_sig)

print("\nBlock.forward:")
print(block_sig)

print("\nViT.forward:")
print(vit_sig)

assert (
    "forced_direct_indices"
    in attn_sig.parameters
), (
    "Attention.forward is missing forced_direct_indices."
)

assert (
    "forced_uniform_mix"
    in attn_sig.parameters
), (
    "Attention.forward is missing forced_uniform_mix."
)

assert (
    "forced_direct_indices"
    in block_sig.parameters
), (
    "Block.forward is missing forced_direct_indices."
)

assert (
    "forced_direct_indices_per_block"
    in vit_sig.parameters
), (
    "ViT.forward is missing forced_direct_indices_per_block."
)

assert (
    "forced_uniform_mix"
    in vit_sig.parameters
), (
    "ViT.forward is missing forced_uniform_mix."
)

print(
    "\nCounterfactual API v2 verification passed."
)
