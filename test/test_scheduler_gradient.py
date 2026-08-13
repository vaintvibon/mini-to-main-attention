import torch

from models.two_level_scheduler import TwoLevelHeadScheduler


def test_scheduler_hard_budget_and_st_gradient():
    """
    1) forward hard mask는 exact budget을 지킨다.
    2) direct는 active subset이다.
    3) ST gate의 forward 값은 hard mask와 동일하다.
    4) backward gradient는 alloc_logits까지 흐른다.
    """
    torch.manual_seed(11)

    scheduler = TwoLevelHeadScheduler(
        main_heads=3,
        direct_ratio=0.34,
        gumbel_tau=1.0,
        use_gumbel=True,
    )
    scheduler.train()

    alloc_logits = torch.tensor(
        [[0.2, -0.1, 0.7], [0.4, 0.3, -0.5]],
        dtype=torch.float32,
        requires_grad=True,
    )

    out = scheduler(alloc_logits, budget=2)

    active_mask = out["active_mask"]
    direct_mask = out["direct_mask"]
    mixed_mask = out["mixed_mask"]

    assert torch.all(active_mask.sum(dim=1) == 2)
    assert torch.all(direct_mask.sum(dim=1) == 1)
    assert torch.all(mixed_mask.sum(dim=1) == 1)
    assert not torch.any(direct_mask & (~active_mask))

    # Straight-through gate는 forward에서는 hard mask와 완전히 동일해야 한다.
    assert torch.equal(out["active_gate"].detach(), active_mask.float())
    assert torch.equal(out["direct_gate"].detach(), direct_mask.float())
    assert torch.equal(out["mixed_gate"].detach(), mixed_mask.float())

    # head마다 다른 downstream utility를 준다고 가정해 gradient 경로를 직접 검사한다.
    utility = torch.tensor([[1.0, -0.3, 0.7], [-0.5, 0.2, 1.3]])
    surrogate_loss = (out["active_gate"] * utility).sum()
    surrogate_loss = surrogate_loss + 0.5 * (out["direct_gate"] * utility).sum()
    surrogate_loss.backward()

    assert alloc_logits.grad is not None
    assert torch.isfinite(alloc_logits.grad).all()
    assert alloc_logits.grad.norm().item() > 0.0


if __name__ == "__main__":
    test_scheduler_hard_budget_and_st_gradient()
    print("test_scheduler_hard_budget_and_st_gradient passed")
