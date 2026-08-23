import torch

from models.contextual_utility_interaction_predictor import (
    ContextualUtilityInteractionPredictor,
)


def test_contextual_utility_interaction_predictor():
    torch.manual_seed(0)

    B = 3
    H = 4

    model = ContextualUtilityInteractionPredictor(
        head_feature_dim=37,
        global_feature_dim=384,
        pair_feature_dim=5,
        mini_heads=H,
        direct_k=2,
        hidden_dim=64,
        dropout=0.0,
    )

    head_features = torch.randn(
        B,
        H,
        37,
    )

    global_features = torch.randn(
        B,
        384,
    )

    pair_features = torch.randn(
        B,
        6,
        5,
    )

    pair_scores, info = model(
        head_features,
        global_features,
        pair_features,
        return_info=True,
    )

    assert pair_scores.shape == (
        B,
        6,
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

    # Interaction correction is centered per sample.
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

    loss = pair_scores.pow(
        2
    ).mean()

    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad
    ]

    assert any(
        g is not None
        and
        g.abs().sum().item()
        >
        0
        for g in grads
    )

    print(
        "Contextual Utility+Interaction predictor test: PASS"
    )


if __name__ == "__main__":
    test_contextual_utility_interaction_predictor()
