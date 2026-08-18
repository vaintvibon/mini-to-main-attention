import torch
import torch.nn.functional as F

from models.dynamic_mini_main_vit import DynamicMiniMainViT


torch.manual_seed(42)


# =============================================================
# Configuration
# =============================================================

LAMBDA_UTILITY = 0.5


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
# 2. Optimizer
# =============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,
)


# =============================================================
# 3. Dummy batch
# =============================================================

x = torch.randn(
    4,
    3,
    224,
    224,
)

labels = torch.tensor(
    [3, 7, 1, 5],
    dtype=torch.long,
)


# =============================================================
# Utility function: module grad norm
# =============================================================

def module_grad_norm(module):
    total = 0.0
    count = 0

    for parameter in module.parameters():

        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        total += (
            grad.norm().item()
            ** 2
        )

        count += 1

    return (
        total ** 0.5,
        count,
    )


# =============================================================
# 4. Save one parameter before optimizer step
#
# 실제 update가 일어나는지도 확인한다.
# =============================================================

before_utility_weight = (
    model.blocks[0]
    .attn
    .utility_predictor
    .scorer[-1]
    .weight
    .detach()
    .clone()
)


# =============================================================
# 5. Forward
# =============================================================

logits, info_list = model(
    x,
    return_info=True,
    collect_taylor=True,
)


# =============================================================
# 6. Task Loss
# =============================================================

sample_losses = F.cross_entropy(
    logits,
    labels,
    reduction="none",
)

task_loss = (
    sample_losses.mean()
)


print("Task loss:")
print(task_loss)


# =============================================================
# 7. Taylor importance
#
# 중요:
# 뒤에서 total_loss.backward()를 해야 하므로
# retain_graph=True
# =============================================================

taylor_gates = [
    info["taylor_gate"]
    for info in info_list
]


gate_grads = torch.autograd.grad(
    outputs=sample_losses.sum(),
    inputs=taylor_gates,

    create_graph=False,
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


print(
    "\nTaylor importance shape:"
)

print(
    taylor_importance.shape
)


# =============================================================
# 8. Taylor target normalization
# =============================================================

importance_sum = (
    taylor_importance
    .sum(
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
# Teacher target은 gradient를 끊는다.
# =============================================================

taylor_target = (
    taylor_target.detach()
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
# 9. Utility supervision loss
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

    block_loss = (
        -(
            target
            *
            log_probs
        )
        .sum(dim=-1)
        .mean()
    )

    utility_losses.append(
        block_loss
    )


utility_loss = (
    torch.stack(
        utility_losses
    )
    .mean()
)


print(
    "\nUtility loss:"
)

print(
    utility_loss
)


# =============================================================
# 10. Joint Loss
#
# L =
# CE
# +
# lambda * Utility
# =============================================================

total_loss = (
    task_loss
    +
    LAMBDA_UTILITY
    *
    utility_loss
)


print(
    "\nTotal loss:"
)

print(
    total_loss
)


# =============================================================
# 11. Joint backward
# =============================================================

optimizer.zero_grad(
    set_to_none=True
)


total_loss.backward()


print(
    "\nBackward completed."
)


# =============================================================
# 12. 각 핵심 module gradient 검사
# =============================================================

modules_to_check = {

    "Mini Attention":
        model.blocks[0]
        .attn
        .mini_attention,

    "Utility Predictor":
        model.blocks[0]
        .attn
        .utility_predictor,

    "Mini-Main Binder":
        model.blocks[0]
        .attn
        .binder,

    "Main Attention":
        model.blocks[0]
        .attn
        .main_attention,

    "Classifier":
        model.head,
}


for name, module in (
    modules_to_check.items()
):

    grad_norm, grad_count = (
        module_grad_norm(
            module
        )
    )

    print(
        f"\n{name} grad norm:"
    )

    print(
        grad_norm
    )

    print(
        "parameters with grad:",
        grad_count,
    )

    assert (
        grad_count > 0
    ), (
        f"{name} received no gradients."
    )

    assert (
        grad_norm > 0.0
    ), (
        f"{name} gradient is zero."
    )


# =============================================================
# 13. Finite gradient 검사
# =============================================================

for name, parameter in (
    model.named_parameters()
):

    if parameter.grad is None:
        continue

    assert torch.isfinite(
        parameter.grad
    ).all(), (
        f"Non-finite gradient: {name}"
    )


# =============================================================
# 14. Optimizer step
# =============================================================

optimizer.step()


# =============================================================
# 15. Utility Predictor parameter가
# 실제로 변경됐는지 확인
# =============================================================

after_utility_weight = (
    model.blocks[0]
    .attn
    .utility_predictor
    .scorer[-1]
    .weight
    .detach()
)


parameter_change = (
    after_utility_weight
    -
    before_utility_weight
).norm()


print(
    "\nUtility Predictor parameter change:"
)

print(
    parameter_change
)


assert (
    parameter_change.item()
    > 0.0
)


# =============================================================
# 16. Final invariants
# =============================================================

assert torch.isfinite(
    task_loss
)

assert torch.isfinite(
    utility_loss
)

assert torch.isfinite(
    total_loss
)


print(
    "\nJoint training step test passed."
)