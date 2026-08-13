import torch

from losses.diversity_loss import HeadDiversityLoss


def test_diversity_zero_pair_backward_is_safe():
    """budget=1처럼 direct-mixed pair가 없을 때도 backward가 깨지지 않아야 한다."""
    head_out = torch.randn(2, 3, 5, 4, requires_grad=True)
    active_mask = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool)
    direct_mask = active_mask.clone()
    mixed_mask = torch.zeros_like(active_mask)

    loss_fn = HeadDiversityLoss(mode="direct_mixed")
    loss = loss_fn(
        head_out=head_out,
        active_mask=active_mask,
        direct_mask=direct_mask,
        mixed_mask=mixed_mask,
    )

    assert loss.item() == 0.0
    loss.backward()
    assert head_out.grad is not None
    assert torch.isfinite(head_out.grad).all()


if __name__ == "__main__":
    test_diversity_zero_pair_backward_is_safe()
    print("test_diversity_zero_pair_backward_is_safe passed")
