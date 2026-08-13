import torch
import torch.nn.functional as F

from models.mini_guided_vit import MiniGuidedViT


def _allocator_grad_report(model):
    report = {}
    for name, p in model.named_parameters():
        if "allocator" in name and p.requires_grad:
            report[name] = None if p.grad is None else float(p.grad.norm().item())
    return report


def test_allocator_receives_gradient():
    """
    Critical regression test.

    전체 model grad norm이 아니라 allocator 자체가 task loss에서 gradient를 받는지 검사한다.
    budget=0에서는 allocation decision이 output에 영향을 주지 않으므로 budget>0을 사용한다.
    """
    torch.manual_seed(7)

    model = MiniGuidedViT(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=10,
        embed_dim=48,
        depth=2,
        main_heads=3,
        mlp_ratio=2.0,
        mini_heads=1,
        mini_dim=16,
        pool_ratio=2,
        direct_ratio=0.34,
        alpha_direct=1.0,
        alpha_mixed=0.2,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        allocator_hidden_dim=32,
        gumbel_tau=1.0,
        use_gumbel=True,
    )
    model.train()

    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    logits = model(x, budget=2)
    loss = F.cross_entropy(logits, y)

    model.zero_grad(set_to_none=True)
    loss.backward()

    report = _allocator_grad_report(model)

    assert report, "No allocator parameters were found in model.named_parameters()."

    missing = [name for name, norm in report.items() if norm is None]
    assert not missing, f"Allocator parameters got no gradient: {missing}"

    zero = [name for name, norm in report.items() if norm == 0.0]
    assert not zero, f"Allocator parameters got zero gradient: {zero}"

    non_finite = [
        name for name, norm in report.items()
        if norm is not None and not torch.isfinite(torch.tensor(norm))
    ]
    assert not non_finite, f"Allocator gradient is non-finite: {non_finite}"


if __name__ == "__main__":
    test_allocator_receives_gradient()
    print("test_allocator_receives_gradient passed")
