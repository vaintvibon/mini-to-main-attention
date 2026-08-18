import torch
import torch.nn.functional as F

from models.dynamic_mini_main_vit import DynamicMiniMainViT


torch.manual_seed(42)


# =============================================================
# 1. Model
# =============================================================

model = DynamicMiniMainViT(
    img_size=224,
    patch_size=16,
    num_classes=10,
    embed_dim=192,
    depth=2,
    main_heads=3,
    mini_heads=4,
    mini_head_dim=16,
    pool_ratio=2,
    utility_hidden_dim=64,
    direct_k=2,
    mix_temperature=1.0,
    bind_dim=64,
    bind_temperature=1.0,
    mlp_ratio=4.0,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
)

model.train()


# =============================================================
# 2. Dummy data
# =============================================================

x = torch.randn(
    2,
    3,
    224,
    224,
)

labels = torch.tensor(
    [3, 7],
    dtype=torch.long,
)


# =============================================================
# 3. Forward with Taylor gates
# =============================================================

logits, info_list = model(
    x,
    return_info=True,
    collect_taylor=True,
)

print("Logits:")
print(logits)


# =============================================================
# 4. Per-sample CE
# =============================================================

sample_losses = F.cross_entropy(
    logits,
    labels,
    reduction="none",
)

print("\nPer-sample CE loss:")
print(sample_losses)


# =============================================================
# 5. Taylor gates
# =============================================================

taylor_gates = [
    info["taylor_gate"]
    for info in info_list
]

for block_idx, gate in enumerate(
    taylor_gates
):
    print(
        f"\nBlock {block_idx} Taylor gate:"
    )
    print(gate)

    print(
        "shape:",
        gate.shape,
    )

    print(
        "requires_grad:",
        gate.requires_grad,
    )

    print(
        "is_leaf:",
        gate.is_leaf,
    )

    assert gate.requires_grad, (
        f"Block {block_idx} Taylor gate "
        "does not require grad."
    )

    assert gate.is_leaf, (
        f"Block {block_idx} Taylor gate "
        "must be a leaf tensor."
    )


# =============================================================
# 6. dL / dxi
#
# 각 gate[b,h]는 해당 sample의 경로에만 있으므로
# sample_losses.sum()의 gradient에서 [b,h] 원소는
# 해당 sample loss에 대한 gate sensitivity가 된다.
# =============================================================

gate_grads = torch.autograd.grad(
    outputs=sample_losses.sum(),
    inputs=taylor_gates,
    create_graph=False,
    retain_graph=False,
    allow_unused=False,
)


# =============================================================
# 7. First-order Taylor importance
#
# xi = 1 이므로:
#
# |xi * dL/dxi| = |dL/dxi|
# =============================================================

taylor_importance_per_block = [
    grad.abs()
    for grad in gate_grads
]

taylor_importance = torch.stack(
    taylor_importance_per_block,
    dim=1,
)

print(
    "\nTaylor importance shape:"
)
print(
    taylor_importance.shape
)

print(
    "\nTaylor importance:"
)
print(
    taylor_importance
)


# =============================================================
# 8. Ranking comparison
# =============================================================

for block_idx in range(
    taylor_importance.shape[1]
):
    block_taylor = (
        taylor_importance[
            :,
            block_idx,
            :,
        ]
    )

    taylor_ranking = torch.argsort(
        block_taylor,
        dim=-1,
        descending=True,
    )

    predicted_utility = (
        info_list[
            block_idx
        ][
            "utility_logits"
        ]
    )

    predicted_ranking = torch.argsort(
        predicted_utility,
        dim=-1,
        descending=True,
    )

    print(
        f"\n================ Block {block_idx} ================"
    )

    print(
        "Taylor importance:"
    )
    print(
        block_taylor
    )

    print(
        "\nTaylor ranking:"
    )
    print(
        taylor_ranking
    )

    print(
        "\nPredicted utility:"
    )
    print(
        predicted_utility
    )

    print(
        "\nPredicted ranking:"
    )
    print(
        predicted_ranking
    )


# =============================================================
# 9. Normalized Taylor teacher target
#
# 기존 코드의 문제:
#
#   target = importance / (sum + 1e-8)
#
# importance 합 자체가 1e-5 수준일 때 +1e-8도
# 상대적으로 무시할 수 없는 오차를 만들 수 있다.
#
# 수정:
#   sum이 양수이면 정확히 sum으로 나눈다.
#   모든 importance가 0인 특수 경우에만 uniform target.
# =============================================================

importance_sum = (
    taylor_importance
    .sum(
        dim=-1,
        keepdim=True,
    )
)

uniform_target = torch.full_like(
    taylor_importance,
    1.0 / taylor_importance.shape[-1],
)

positive_sum_mask = (
    importance_sum > 0
)

safe_denominator = torch.where(
    positive_sum_mask,
    importance_sum,
    torch.ones_like(
        importance_sum
    ),
)

taylor_target = (
    taylor_importance
    / safe_denominator
)

taylor_target = torch.where(
    positive_sum_mask.expand_as(
        taylor_target
    ),
    taylor_target,
    uniform_target,
)

target_sum = (
    taylor_target
    .sum(dim=-1)
)

print(
    "\nNormalized Taylor target:"
)
print(
    taylor_target
)

print(
    "\nTarget sum per sample/block:"
)
print(
    target_sum
)


# =============================================================
# 10. Invariants
# =============================================================

assert taylor_importance.shape == (
    2,
    2,
    4,
)

assert torch.isfinite(
    taylor_importance
).all()

assert torch.all(
    taylor_importance >= 0
)

assert torch.isfinite(
    taylor_target
).all()

assert torch.all(
    taylor_target >= 0
)

assert torch.allclose(
    target_sum,
    torch.ones_like(
        target_sum
    ),
    atol=1e-6,
    rtol=1e-6,
)

print(
    "\nTaylor Mini Head importance test passed."
)
