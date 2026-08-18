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
# 2. Dummy batch
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
# 3. Forward
# =============================================================

logits, info_list = model(
    x,
    return_info=True,
    collect_taylor=True,
)


# =============================================================
# 4. Per-sample task loss
# =============================================================

sample_losses = F.cross_entropy(
    logits,
    labels,
    reduction="none",
)

task_loss = (
    sample_losses.mean()
)


print(
    "Task loss:"
)

print(
    task_loss
)


# =============================================================
# 5. Taylor gates
# =============================================================

taylor_gates = [
    info["taylor_gate"]
    for info in info_list
]


# =============================================================
# 6. Taylor importance
#
# retain_graph=True가 중요하다.
#
# 이후 Utility loss.backward()를 해야 하므로
# forward graph를 여기서 버리면 안 된다.
# =============================================================

gate_grads = torch.autograd.grad(
    outputs=sample_losses.sum(),
    inputs=taylor_gates,

    create_graph=False,

    # 중요
    retain_graph=True,

    allow_unused=False,
)


taylor_importance = torch.stack(
    [
        grad.abs()
        for grad in gate_grads
    ],
    dim=1,
)

# [B, depth, Hmini]


print(
    "\nTaylor importance:"
)

print(
    taylor_importance
)


# =============================================================
# 7. Normalize Taylor target
# =============================================================

importance_sum = (
    taylor_importance.sum(
        dim=-1,
        keepdim=True,
    )
)

has_signal = (
    importance_sum > 0
)


safe_sum = torch.where(
    has_signal,
    importance_sum,
    torch.ones_like(
        importance_sum
    ),
)


taylor_target = (
    taylor_importance
    / safe_sum
)


uniform_target = torch.full_like(
    taylor_target,
    1.0 / taylor_target.shape[-1],
)


taylor_target = torch.where(
    has_signal.expand_as(
        taylor_target
    ),
    taylor_target,
    uniform_target,
)


# =============================================================
# 매우 중요
#
# Taylor는 teacher target이다.
#
# Utility loss backward가 Taylor 계산 graph까지
# 다시 미분하지 않도록 detach한다.
# =============================================================

taylor_target = (
    taylor_target.detach()
)


print(
    "\nTaylor target:"
)

print(
    taylor_target
)


print(
    "\nTaylor target sum:"
)

print(
    taylor_target.sum(
        dim=-1
    )
)


# =============================================================
# 8. Utility supervision loss
#
# 각 Block:
#
# predicted logits [B,Hmini]
#
# teacher target   [B,Hmini]
# =============================================================

utility_losses = []


for block_idx, info in enumerate(
    info_list
):

    utility_logits = (
        info["utility_logits"]
    )

    target = (
        taylor_target[
            :,
            block_idx,
            :,
        ]
    )

    log_probs = F.log_softmax(
        utility_logits,
        dim=-1,
    )

    block_utility_loss = (
        -(
            target
            *
            log_probs
        )
        .sum(dim=-1)
        .mean()
    )

    utility_losses.append(
        block_utility_loss
    )

    print(
        f"\nBlock {block_idx} "
        f"utility loss:"
    )

    print(
        block_utility_loss
    )


utility_loss = torch.stack(
    utility_losses
).mean()


print(
    "\nMean utility loss:"
)

print(
    utility_loss
)


# =============================================================
# 9. Utility Predictor만 gradient 확인
#
# 여기서는 구조 검증이 목적이라
# task_loss는 backward하지 않는다.
#
# Utility teacher loss가 실제 predictor에
# gradient를 전달하는지만 확인.
# =============================================================

model.zero_grad(
    set_to_none=True
)


utility_loss.backward()


# =============================================================
# 10. Block별 Utility Predictor gradient
# =============================================================

total_utility_grad_norm = 0.0


for block_idx, block in enumerate(
    model.blocks
):

    grad_norm = 0.0

    grad_parameter_count = 0

    for name, parameter in (
        block
        .attn
        .utility_predictor
        .named_parameters()
    ):

        if parameter.grad is None:
            continue

        current_norm = (
            parameter.grad
            .detach()
            .norm()
            .item()
        )

        grad_norm += (
            current_norm ** 2
        )

        grad_parameter_count += 1


    grad_norm = (
        grad_norm ** 0.5
    )

    total_utility_grad_norm += (
        grad_norm ** 2
    )


    print(
        f"\nBlock {block_idx} "
        f"Utility Predictor grad norm:"
    )

    print(
        grad_norm
    )


    assert (
        grad_parameter_count > 0
    ), (
        f"Block {block_idx}: "
        "Utility Predictor received no gradients."
    )


    assert (
        grad_norm > 0.0
    ), (
        f"Block {block_idx}: "
        "Utility Predictor gradient is zero."
    )


total_utility_grad_norm = (
    total_utility_grad_norm
    ** 0.5
)


print(
    "\nTotal Utility Predictor grad norm:"
)

print(
    total_utility_grad_norm
)


# =============================================================
# 11. Important invariant
# =============================================================

assert taylor_target.shape == (
    2,
    2,
    4,
)


assert torch.allclose(
    taylor_target.sum(dim=-1),
    torch.ones(
        2,
        2,
        device=taylor_target.device,
    ),
    atol=1e-6,
)


assert torch.isfinite(
    utility_loss
)


assert (
    total_utility_grad_norm > 0
)


print(
    "\nTaylor Utility Supervision test passed."
)