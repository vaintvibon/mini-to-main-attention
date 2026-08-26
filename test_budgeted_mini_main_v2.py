import torch
from models.budgeted_mini_main_v2 import BudgetedMiniMainViTV2


def test_mini_main_shapes_and_budget_counts():
    torch.manual_seed(1)
    m=BudgetedMiniMainViTV2(img_size=32,patch_size=4,embed_dim=192,depth=2,main_heads=8,mini_heads=4,mini_head_dim=16,direct_k=2,mode='mini_main')
    x=torch.randn(3,3,32,32); m.eval()
    for b in [0,2,4,6,8]:
        y,infos=m(x,budget=b,return_info=True); assert y.shape==(3,10)
        for info in infos:
            assert torch.all(info['active_main_mask'].sum(-1)==b)
            assert info['computed_sample_heads']==3*b


def test_sparse_dense_eval_match():
    torch.manual_seed(2)
    m=BudgetedMiniMainViTV2(img_size=32,patch_size=4,embed_dim=192,depth=2,main_heads=8,mini_heads=4,mini_head_dim=16,direct_k=2,mode='mini_main')
    x=torch.randn(2,3,32,32); m.eval()
    for b in [0,2,4,6,8]:
        with torch.no_grad():
            a=m(x,budget=b,force_dense_main=False); c=m(x,budget=b,force_dense_main=True)
        d=(a-c).abs().max().item(); assert d<1e-5,(b,d)


def test_main_only():
    torch.manual_seed(3)
    m=BudgetedMiniMainViTV2(img_size=32,patch_size=4,embed_dim=192,depth=2,main_heads=8,mini_heads=4,mini_head_dim=16,direct_k=2,mode='main_only')
    x=torch.randn(4,3,32,32); m.eval()
    for b in [2,4,6,8]:
        y,infos=m(x,budget=b,return_info=True); assert y.shape==(4,10)
        for info in infos:
            assert torch.all(info['active_main_mask'].sum(-1)==b)
            assert info['computed_sample_heads']==4*b

if __name__=='__main__':
    test_mini_main_shapes_and_budget_counts(); print('test_mini_main_shapes_and_budget_counts passed')
    test_sparse_dense_eval_match(); print('test_sparse_dense_eval_match passed')
    test_main_only(); print('test_main_only passed')
