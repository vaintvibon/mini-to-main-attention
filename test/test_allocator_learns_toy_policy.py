import copy

import torch
import torch.nn.functional as F

from models.mini_to_main_allocator import MiniToMainAllocator
from models.two_level_scheduler import TwoLevelHeadScheduler


def _build_toy_batch(
    batch_size: int = 96,
    num_tokens: int = 3,
    dim: int = 8,
):
    """
    Mini가 서로 다른 두 종류의 sample을 구분할 수 있다고 가정한 synthetic 입력.

    type A:
        Mini context/confidence -> target Main head 0

    type B:
        Mini context/confidence -> target Main head 2

    중요한 점:
        정답 head label을 allocator의 CE loss에 직접 넣지 않는다.
        scheduler의 active_gate를 통해 얻는 downstream reward만 사용한다.
        따라서 allocator update는 Gumbel-ST scheduling path를 반드시 통과한다.
    """
    if batch_size % 2 != 0:
        raise ValueError("batch_size must be even for this toy test.")

    sample_type = torch.arange(batch_size) % 2

    target_head = torch.where(
        sample_type == 0,
        torch.tensor(0),
        torch.tensor(2),
    ).long()

    c_m = torch.zeros(batch_size, num_tokens, dim)

    # Allocator가 사용하는 CLS Mini context에 두 sample type의 차이를 심는다.
    c_m[:, 0, 0] = torch.where(
        sample_type == 0,
        torch.tensor(2.0),
        torch.tensor(-2.0),
    )
    c_m[:, 0, 1] = torch.where(
        sample_type == 0,
        torch.tensor(0.5),
        torch.tensor(-0.5),
    )

    # Mini confidence 역시 sample-dependent하게 둔다.
    importance_feat = torch.zeros(batch_size, 2)
    importance_feat[:, 0] = torch.where(
        sample_type == 0,
        torch.tensor(0.8),
        torch.tensor(0.2),
    )
    importance_feat[:, 1] = torch.where(
        sample_type == 0,
        torch.tensor(0.7),
        torch.tensor(0.3),
    )

    return c_m, importance_feat, target_head


@torch.no_grad()
def _evaluate_policy(
    allocator,
    scheduler,
    c_m,
    importance_feat,
    target_head,
):
    """
    Evaluation에서는 scheduler.eval()을 사용하므로 Gumbel noise 없는
    deterministic hard Top-1 선택을 검사한다.
    """
    allocator.eval()
    scheduler.eval()

    alloc_logits = allocator(c_m, importance_feat)
    out = scheduler(alloc_logits, budget=1)

    pred_head = out["active_mask"].float().argmax(dim=-1)
    accuracy = (pred_head == target_head).float().mean().item()

    target_logit = alloc_logits.gather(
        dim=1,
        index=target_head.unsqueeze(1),
    ).squeeze(1)

    other_logits = alloc_logits.clone()
    other_logits.scatter_(1, target_head.unsqueeze(1), float("-inf"))
    best_other_logit = other_logits.max(dim=1).values

    target_margin = (target_logit - best_other_logit).mean().item()

    return {
        "accuracy": accuracy,
        "target_margin": target_margin,
        "pred_head": pred_head,
        "alloc_logits": alloc_logits,
    }


def test_allocator_learns_toy_policy():
    """
    Gumbel-ST allocator learning regression test.

    이 테스트가 확인하는 것:
      1. allocator parameter에 gradient가 존재하는 것에서 끝나지 않는다.
      2. Gumbel-ST active_gate를 통한 reward만으로 allocator가 실제로 변한다.
      3. 서로 다른 Mini input이 서로 다른 Main head를 선택하도록 학습된다.
      4. eval 시 hard Top-1 policy가 높은 정확도로 target policy를 재현한다.

    주의:
      이 테스트는 실제 이미지에서 좋은 allocation을 학습한다는 증거가 아니다.
      "현재 differentiable scheduling pipeline이 policy optimization 자체는 할 수 있다"
      는 것을 검증하는 controlled toy test다.
    """
    torch.manual_seed(123)

    dim = 8
    main_heads = 3

    allocator = MiniToMainAllocator(
        dim=dim,
        main_heads=main_heads,
        importance_dim=2,
        hidden_dim=16,
        dropout=0.0,
        use_cls_token=True,
    )

    scheduler = TwoLevelHeadScheduler(
        main_heads=main_heads,
        direct_ratio=0.34,
        gumbel_tau=1.0,
        use_gumbel=True,
    )

    c_m, importance_feat, target_head = _build_toy_batch(
        batch_size=96,
        num_tokens=3,
        dim=dim,
    )

    before_state = {
        name: p.detach().clone()
        for name, p in allocator.named_parameters()
        if p.requires_grad
    }

    optimizer = torch.optim.Adam(
        allocator.parameters(),
        lr=3e-2,
    )

    allocator.train()
    scheduler.train()

    num_steps = 60

    for _ in range(num_steps):
        alloc_logits = allocator(
            c_m,
            importance_feat,
        )

        schedule_out = scheduler(
            alloc_logits,
            budget=1,
        )

        # target head를 선택했을 때 reward=1, 아니면 reward=0.
        # forward에서는 hard selected head의 reward가 사용되고,
        # backward에서는 active_gate의 ST surrogate를 통해
        # alloc_logits -> allocator로 gradient가 흐른다.
        target_utility = F.one_hot(
            target_head,
            num_classes=main_heads,
        ).float()

        reward = (
            schedule_out["active_gate"] * target_utility
        ).sum(dim=-1)

        loss = -reward.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 실제 allocator parameter가 update됐는지 확인.
    changed = False
    for name, p in allocator.named_parameters():
        if p.requires_grad and not torch.equal(
            before_state[name],
            p.detach(),
        ):
            changed = True
            break

    assert changed, "Allocator parameters did not change during toy policy training."

    result = _evaluate_policy(
        allocator,
        scheduler,
        c_m,
        importance_feat,
        target_head,
    )

    # Balanced binary mapping이므로 한 head로 collapse하면 최대 50% 수준.
    # 95% 이상이면 input-conditioned mapping을 실제로 학습했다고 볼 수 있다.
    assert result["accuracy"] >= 0.95, (
        "Allocator failed to learn the toy input-conditioned policy. "
        f"accuracy={result['accuracy']:.4f}"
    )

    # target head logit이 다른 head보다 평균적으로 높아졌는지 확인.
    assert result["target_margin"] > 0.0, (
        "Target Main-head logit is not larger than competing head logits. "
        f"mean target margin={result['target_margin']:.6f}"
    )

    # 두 sample type이 서로 다른 head를 선택하는지 명시적으로 확인.
    even_pred = result["pred_head"][0::2]
    odd_pred = result["pred_head"][1::2]

    assert torch.all(even_pred == 0), (
        "Type-A samples should select Main head 0 after training."
    )
    assert torch.all(odd_pred == 2), (
        "Type-B samples should select Main head 2 after training."
    )


if __name__ == "__main__":
    test_allocator_learns_toy_policy()
    print("test_allocator_learns_toy_policy passed")