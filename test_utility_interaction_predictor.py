import torch

from models.utility_interaction_predictor import (
    UtilityInteractionPredictor,
)


torch.manual_seed(42)

B = 4
H = 4
F = 34

model = UtilityInteractionPredictor(
    feature_dim=F,
    mini_heads=H,
    direct_k=2,
    hidden_dim=64,
)

features = torch.randn(
    B,
    H,
    F,
    requires_grad=True,
)

pair_scores, info = model(
    features,
    return_info=True,
)

print(
    "Combinations:"
)

print(
    model.combinations
)

print(
    "\nUtility logits shape:"
)

print(
    info[
        "utility_logits"
    ].shape
)

print(
    "\nInteraction shape:"
)

print(
    info[
        "interaction_scores"
    ].shape
)

print(
    "\nFinal pair scores shape:"
)

print(
    pair_scores.shape
)

print(
    "\nMean interaction per sample:"
)

print(
    info[
        "interaction_scores"
    ].mean(
        dim=-1
    )
)

assert info[
    "utility_logits"
].shape == (
    B,
    H,
)

assert info[
    "interaction_scores"
].shape == (
    B,
    6,
)

assert pair_scores.shape == (
    B,
    6,
)

# Interaction is centered to be a relative correction.
assert torch.allclose(
    info[
        "interaction_scores"
    ].mean(
        dim=-1
    ),
    torch.zeros(
        B
    ),
    atol=1e-6,
)

loss = (
    pair_scores.pow(2).mean()
    +
    info[
        "utility_logits"
    ].pow(2).mean()
)

loss.backward()

grad_norm_sq = 0.0

for parameter in model.parameters():
    if parameter.grad is not None:
        grad_norm_sq += (
            parameter.grad
            .detach()
            .pow(2)
            .sum()
            .item()
        )

grad_norm = (
    grad_norm_sq
    **
    0.5
)

print(
    "\nGradient norm:"
)

print(
    grad_norm
)

assert grad_norm > 0.0

print(
    "\nUtility + Interaction predictor test passed."
)
