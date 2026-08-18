import torch
import torch.nn.functional as F

from models.dynamic_mini_main_vit import DynamicMiniMainViT
from models.balanced_direct_scheduler import BalancedDirectSubsetScheduler


torch.manual_seed(42)


STEPS = 30
LR = 1e-3
BATCH_SIZE = 8

MINI_HEADS = 4
DIRECT_K = 2
DEPTH = 2


# =============================================================
# 1. Small CIFAR-size model
# =============================================================

model = DynamicMiniMainViT(
    img_size=32,
    patch_size=4,
    num_classes=10,
    embed_dim=192,
    depth=DEPTH,
    main_heads=3,
    mini_heads=MINI_HEADS,
    mini_head_dim=16,
    pool_ratio=2,
    utility_hidden_dim=64,
    direct_k=DIRECT_K,
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
# 2. Utility Predictor freeze
#
# Stage-1에서는 predictor가 routing과 Mix 어느 쪽에도 개입하지 않는다.
# =============================================================

utility_parameters = []

for block in model.blocks:
    predictor = (
        block
        .attn
        .utility_predictor
    )

    for parameter in predictor.parameters():
        parameter.requires_grad_(
            False
        )

        utility_parameters.append(
            parameter
        )


before_utility = [
    parameter.detach().clone()
    for parameter in utility_parameters
]


optimizer = torch.optim.AdamW(
    [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ],
    lr=LR,
    weight_decay=0.01,
)


# =============================================================
# 3. Fixed batch
# =============================================================

x = torch.randn(
    BATCH_SIZE,
    3,
    32,
    32,
)

labels = torch.arange(
    BATCH_SIZE,
    dtype=torch.long,
) % 10


# =============================================================
# 4. Balanced scheduler
# =============================================================

scheduler = BalancedDirectSubsetScheduler(
    mini_heads=MINI_HEADS,
    direct_k=DIRECT_K,
)


combo_to_id = {
    tuple(combo): idx
    for idx, combo in enumerate(
        scheduler.combinations
    )
}


coverage = torch.zeros(
    DEPTH,
    scheduler.num_combinations,
    dtype=torch.long,
)


# =============================================================
# 5. Stage-1 balanced warm-up sanity training
# =============================================================

first_loss = None
last_loss = None


for step in range(
    STEPS
):
    optimizer.zero_grad(
        set_to_none=True
    )

    forced = scheduler.get_for_all_blocks(
        batch_size=BATCH_SIZE,
        depth=DEPTH,
        step=step,
        device=x.device,
    )

    # coverage logging
    for block_idx in range(
        DEPTH
    ):
        for row in forced[
            block_idx
        ].tolist():
            coverage[
                block_idx,
                combo_to_id[
                    tuple(row)
                ],
            ] += 1

    logits, info_list = model(
        x,
        return_info=True,
        collect_taylor=False,
        forced_direct_indices_per_block=forced,
        forced_uniform_mix=True,
    )

    loss = F.cross_entropy(
        logits,
        labels,
    )

    if first_loss is None:
        first_loss = (
            loss.detach().item()
        )

    loss.backward()

    # Utility Predictor는 freeze + routing 우회 상태이므로 grad가 없어야 한다.
    for block_idx, block in enumerate(
        model.blocks
    ):
        for name, parameter in (
            block
            .attn
            .utility_predictor
            .named_parameters()
        ):
            assert parameter.grad is None, (
                f"Stage-1 leaked gradient into Utility Predictor: "
                f"block={block_idx}, parameter={name}"
            )

    optimizer.step()

    last_loss = (
        loss.detach().item()
    )

    if (
        step == 0
        or (step + 1) % 5 == 0
        or step == STEPS - 1
    ):
        accuracy = (
            logits.argmax(dim=-1)
            ==
            labels
        ).float().mean().item()

        print(
            f"\nStep {step + 1:02d}/{STEPS}"
        )

        print(
            f"CE loss: {last_loss:.6f}"
        )

        print(
            f"Accuracy: {accuracy * 100:.2f}%"
        )


# =============================================================
# 6. Verify Utility Predictor did not change
# =============================================================

after_utility = [
    parameter.detach()
    for parameter in utility_parameters
]


max_utility_change = 0.0

for before, after in zip(
    before_utility,
    after_utility,
):
    change = (
        after
        -
        before
    ).abs().max().item()

    max_utility_change = max(
        max_utility_change,
        change,
    )


# =============================================================
# 7. Coverage
#
# 30 steps * 8 samples = 240 routing assignments / block.
# 6 combinations => exactly 40 each.
# =============================================================

print(
    "\nBalanced combination coverage:"
)
print(
    coverage
)


print(
    "\nUtility Predictor max parameter change:"
)
print(
    max_utility_change
)


print(
    "\nCE loss:"
)
print(
    first_loss,
    "->",
    last_loss,
)


expected_count = (
    STEPS
    *
    BATCH_SIZE
    //
    scheduler.num_combinations
)


assert (
    STEPS
    *
    BATCH_SIZE
    %
    scheduler.num_combinations
    == 0
)


assert torch.all(
    coverage
    ==
    expected_count
), (
    f"Unbalanced coverage: {coverage}"
)


assert max_utility_change == 0.0, (
    "Utility Predictor changed during Stage-1."
)


assert last_loss < first_loss, (
    "Stage-1 CE loss did not decrease."
)


print(
    "\nStage-1 balanced warm-up test passed."
)
