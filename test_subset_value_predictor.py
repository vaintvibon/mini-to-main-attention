import torch

from models.mini_subset_value_predictor import MiniSubsetValuePredictor


torch.manual_seed(42)

B = 3
H = 4
K = 2
N = 65
Dh = 16
M = 17

predictor = MiniSubsetValuePredictor(
    mini_head_dim=Dh,
    mini_heads=H,
    direct_k=K,
    hidden_dim=64,
    dropout=0.0,
)

mini_contexts = torch.randn(
    B, H, N, Dh,
    requires_grad=True,
)

mini_attn = torch.softmax(
    torch.randn(B, H, N, M),
    dim=-1,
)

features = predictor.extract_local_features(
    mini_contexts,
    mini_attn,
)

scores = predictor.forward_from_features(
    features
)

print("Combinations:")
print(predictor.combinations)

print("\nFeature shape:")
print(features.shape)

print("\nSubset score shape:")
print(scores.shape)

print("\nSubset scores:")
print(scores)

assert predictor.num_combinations == 6
assert features.shape == (
    B,
    H,
    2 * Dh + 2,
)
assert scores.shape == (
    B,
    6,
)

loss = scores.pow(2).mean()
loss.backward()

grad_norm_sq = 0.0

for parameter in predictor.parameters():
    if parameter.grad is not None:
        grad_norm_sq += (
            parameter.grad
            .detach()
            .pow(2)
            .sum()
            .item()
        )

grad_norm = grad_norm_sq ** 0.5

print("\nPredictor grad norm:")
print(grad_norm)

assert grad_norm > 0.0

scores2 = predictor(
    mini_contexts,
    mini_attn,
)

assert torch.allclose(
    scores,
    scores2,
    atol=1e-6,
)

print(
    "\nMiniSubsetValuePredictor test passed."
)
