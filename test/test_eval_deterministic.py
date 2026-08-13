import torch

from models.mini_guided_vit import MiniGuidedViT


def _build_model():
    return MiniGuidedViT(
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
        use_gumbel=True,  # train에서는 Gumbel 사용, eval에서는 자동 비활성화되어야 함
    )


def _assert_same_infos(ref_infos, new_infos):
    assert len(ref_infos) == len(new_infos)

    for block_idx, (ref, new) in enumerate(zip(ref_infos, new_infos)):
        for key in [
            "active_mask",
            "direct_mask",
            "mixed_mask",
            "inactive_mask",
        ]:
            assert torch.equal(ref[key], new[key]), (
                f"Block {block_idx}: {key} changed across repeated eval forwards."
            )

        for key in [
            "active_gate",
            "direct_gate",
            "mixed_gate",
            "selection_scores",
            "alloc_logits",
        ]:
            assert torch.allclose(
                ref[key],
                new[key],
                atol=0.0,
                rtol=0.0,
            ), (
                f"Block {block_idx}: {key} changed across repeated eval forwards."
            )


def test_eval_is_deterministic_and_gumbel_free():
    """
    Full MiniGuidedViT evaluation regression test.

    확인 항목:
      1. use_gumbel=True로 만든 모델이어도 model.eval()에서는 Gumbel noise를 쓰지 않는다.
      2. 같은 input + 같은 budget의 반복 forward에서 logits가 동일하다.
      3. 모든 block의 active/direct/mixed mask가 동일하다.
      4. eval에서는 selection_scores == alloc_logits 이어야 한다.
         즉 scheduler가 Gumbel perturbation을 추가하지 않았음을 직접 확인한다.
      5. eval gate의 forward 값은 hard mask와 정확히 동일하다.
    """
    torch.manual_seed(2026)

    model = _build_model()
    model.eval()

    x = torch.randn(4, 3, 32, 32)

    # budget=1,2,3 모두 확인.
    # budget=3은 active set은 all-head지만 direct/mixed 분할은 여전히 allocator에 의존한다.
    for budget in [1, 2, 3]:
        with torch.no_grad():
            ref_logits, ref_infos = model(
                x,
                budget=budget,
                return_info=True,
            )

            # eval에서 Gumbel noise가 완전히 제거됐는지 직접 확인.
            for block_idx, info in enumerate(ref_infos):
                assert torch.allclose(
                    info["selection_scores"],
                    info["alloc_logits"],
                    atol=0.0,
                    rtol=0.0,
                ), (
                    f"Block {block_idx}, budget={budget}: "
                    "selection_scores != alloc_logits in eval mode. "
                    "Gumbel perturbation may still be active."
                )

                assert torch.equal(
                    info["active_gate"],
                    info["active_mask"].float(),
                ), (
                    f"Block {block_idx}, budget={budget}: "
                    "eval active_gate is not exact hard mask."
                )

                assert torch.equal(
                    info["direct_gate"],
                    info["direct_mask"].float(),
                ), (
                    f"Block {block_idx}, budget={budget}: "
                    "eval direct_gate is not exact hard mask."
                )

                assert torch.equal(
                    info["mixed_gate"],
                    info["mixed_mask"].float(),
                ), (
                    f"Block {block_idx}, budget={budget}: "
                    "eval mixed_gate is not exact hard mask."
                )

            # 같은 입력을 여러 번 넣어도 완전히 동일해야 한다.
            for _ in range(4):
                logits, infos = model(
                    x,
                    budget=budget,
                    return_info=True,
                )

                assert torch.allclose(
                    ref_logits,
                    logits,
                    atol=0.0,
                    rtol=0.0,
                ), (
                    f"budget={budget}: logits changed across repeated eval forwards."
                )

                _assert_same_infos(
                    ref_infos,
                    infos,
                )


if __name__ == "__main__":
    test_eval_is_deterministic_and_gumbel_free()
    print("test_eval_is_deterministic_and_gumbel_free passed")