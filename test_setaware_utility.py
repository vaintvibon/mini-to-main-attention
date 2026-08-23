import torch

from models.set_aware_mini_head_utility import (
    SetAwareMiniHeadUtility,
)


torch.manual_seed(42)


B = 3
H = 4
N = 65
Dh = 16
M = 17


predictor = SetAwareMiniHeadUtility(
    mini_head_dim=Dh,
    hidden_dim=64,
    dropout=0.0,
    has_cls_token=True,
)


mini_contexts = torch.randn(
    B,
    H,
    N,
    Dh,
    requires_grad=True,
)


raw_attn = torch.randn(
    B,
    H,
    N,
    M,
)


mini_attn = torch.softmax(
    raw_attn,
    dim=-1,
)


logits, info = predictor(
    mini_contexts,
    mini_attn,
    return_info=True,
)


print(
    "Utility logits shape:"
)

print(
    logits.shape
)


print(
    "\nUtility probabilities:"
)

print(
    info[
        "utility_probs"
    ]
)


print(
    "\nProbability sums:"
)

print(
    info[
        "utility_probs"
    ].sum(
        dim=-1
    )
)


print(
    "\nLocal encoded shape:"
)

print(
    info[
        "local_encoded"
    ].shape
)


print(
    "\nSet mean shape:"
)

print(
    info[
        "set_mean"
    ].shape
)


loss = (
    logits.pow(2).mean()
)


loss.backward()


grad_norm = 0.0

for parameter in predictor.parameters():
    if parameter.grad is not None:
        grad_norm += (
            parameter.grad
            .detach()
            .pow(2)
            .sum()
            .item()
        )

grad_norm = (
    grad_norm ** 0.5
)


print(
    "\nPredictor grad norm:"
)

print(
    grad_norm
)


assert logits.shape == (
    B,
    H,
)


assert torch.allclose(
    info[
        "utility_probs"
    ].sum(
        dim=-1
    ),
    torch.ones(B),
    atol=1e-6,
)


assert info[
    "local_encoded"
].shape == (
    B,
    H,
    64,
)


assert info[
    "set_mean"
].shape == (
    B,
    64,
)


assert grad_norm > 0.0


print(
    "\nSet-Aware Utility Predictor test passed."
)
