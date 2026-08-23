import torch

from models.direct_pair_value_predictor import (
    DirectPairValuePredictor,
    ContextualDirectPairValuePredictor,
)


def test_direct_pair_value_predictors():
    torch.manual_seed(0)

    B = 3
    H = 4

    basic = DirectPairValuePredictor(
        feature_dim=34,
        mini_heads=H,
        direct_k=2,
        hidden_dim=64,
        dropout=0.0,
    )

    head = torch.randn(
        B,
        H,
        34,
    )

    scores, info = basic(
        head,
        return_info=True,
    )

    assert scores.shape == (
        B,
        6,
    )

    assert info[
        "head_encoded"
    ].shape == (
        B,
        H,
        64,
    )

    scores.pow(
        2
    ).mean().backward()

    assert any(
        p.grad is not None
        and
        p.grad.abs().sum().item()
        >
        0
        for p in basic.parameters()
    )

    contextual = ContextualDirectPairValuePredictor(
        head_feature_dim=37,
        global_feature_dim=384,
        pair_feature_dim=5,
        mini_heads=H,
        direct_k=2,
        hidden_dim=64,
        dropout=0.0,
    )

    rich_head = torch.randn(
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

    scores, info = contextual(
        rich_head,
        global_features,
        pair_features,
        return_info=True,
    )

    assert scores.shape == (
        B,
        6,
    )

    scores.mean().backward()

    assert any(
        p.grad is not None
        and
        p.grad.abs().sum().item()
        >
        0
        for p in contextual.parameters()
    )

    print(
        "Direct pair-value predictor tests: PASS"
    )


if __name__ == "__main__":
    test_direct_pair_value_predictors()
