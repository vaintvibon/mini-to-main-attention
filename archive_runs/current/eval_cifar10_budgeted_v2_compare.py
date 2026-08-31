# -*- coding: utf-8 -*-
"""Heldout comparison for 4-Mini/8-Main v2 vs matched Main-only baseline."""
import argparse, json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.budgeted_mini_main_v2 import BudgetedMiniMainViTV2


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',default='/content/cifar10')
    p.add_argument('--mini-main-checkpoint',required=True)
    p.add_argument('--main-only-checkpoint',required=True)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstrap-repeats',type=int,default=5000)
    p.add_argument('--latency-repeats',type=int,default=10)
    p.add_argument('--output',default='./outputs/budgeted_v2_compare.json')
    return p.parse_args()


def load_model(path,device):
    ck=torch.load(path,map_location=device,weights_only=False); c=ck['config']
    m=BudgetedMiniMainViTV2(
        img_size=c['img_size'],patch_size=c['patch_size'],num_classes=c['num_classes'],
        embed_dim=c['embed_dim'],depth=c['depth'],main_heads=c['main_heads'],
        mini_heads=c['mini_heads'],mini_head_dim=c['mini_head_dim'],direct_k=c['direct_k'],
        pool_ratio=c['pool_ratio'],mode=c['mode'])
    m.load_state_dict(ck['model'],strict=True); m.to(device).eval(); return m


def heldout_loader(args):
    tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    base=datasets.CIFAR10(args.data_dir,train=True,download=True,transform=tf)
    g=torch.Generator().manual_seed(args.seed); perm=torch.randperm(50000,generator=g).tolist()
    ds=Subset(base,perm[45000:50000])
    return DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=torch.cuda.is_available())


@torch.no_grad()
def collect(model,loader,budget,device):
    ls=[]; cs=[]
    for x,y in loader:
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        logits=model(x,budget=budget)
        ls.append(F.cross_entropy(logits,y,reduction='none').cpu())
        cs.append(logits.argmax(-1).eq(y).cpu())
    return {'losses':torch.cat(ls),'correct':torch.cat(cs)}


def summary(r): return {'ce':r['losses'].mean().item(),'acc':100*r['correct'].float().mean().item()}


def boot(delta,repeats,seed):
    d=delta.float().cpu(); n=d.numel(); g=torch.Generator().manual_seed(seed); vals=[]; done=0
    while done<repeats:
        r=min(200,repeats-done); idx=torch.randint(0,n,(r,n),generator=g); vals.append(d[idx].mean(1)); done+=r
    v=torch.cat(vals)
    return {'mean':d.mean().item(),'ci95':[torch.quantile(v,.025).item(),torch.quantile(v,.975).item()]}


@torch.no_grad()
def latency(model,budget,device,batch,repeats):
    x=torch.randn(batch,3,32,32,device=device)
    for _ in range(3): model(x,budget=budget)
    if device.type=='cuda': torch.cuda.synchronize()
    ts=[]
    for _ in range(repeats):
        t=time.perf_counter(); model(x,budget=budget)
        if device.type=='cuda': torch.cuda.synchronize()
        ts.append((time.perf_counter()-t)*1000)
    s=sorted(ts)
    return {'mean_ms':sum(ts)/len(ts),'median_ms':s[len(s)//2],'batch_size':batch}


def main():
    a=parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*100); print('4 MINI / 8 MAIN HELDOUT COMPARISON'); print('='*100)
    print('device:',device); print('Heldout: CIFAR10 train permutation [45000,50000), n=5000'); print('Official test: NOT USED')
    mm=load_model(a.mini_main_checkpoint,device); mo=load_model(a.main_only_checkpoint,device)
    if mm.mode!='mini_main' or mo.mode!='main_only': raise ValueError('wrong checkpoint modes')
    loader=heldout_loader(a); raw_mm={}; raw_mo={}; report={'mini_main':{},'main_only':{},'paired':{},'macs':{},'latency':{},'official_test_used':False}
    print('\n[MiniMain target]')
    for b in [0,2,4,6,8]:
        r=collect(mm,loader,b,device); raw_mm[b]=r; report['mini_main'][str(b)]=summary(r); s=summary(r); print(f"B={b}: CE={s['ce']:.6f} Acc={s['acc']:.2f}%")
    print('\n[MainOnly baseline]')
    for b in [2,4,6,8]:
        r=collect(mo,loader,b,device); raw_mo[b]=r; report['main_only'][str(b)]=summary(r); s=summary(r); print(f"B={b}: CE={s['ce']:.6f} Acc={s['acc']:.2f}%")
    print('\n[Same-budget MiniMain - MainOnly]')
    for i,b in enumerate([2,4,6,8]):
        d=boot(raw_mm[b]['losses']-raw_mo[b]['losses'],a.bootstrap_repeats,a.seed+100+i); report['paired'][f'B{b}_mini_minus_mainonly']=d
        print(f"B={b}: dCE={d['mean']:+.8f} 95%CI[{d['ci95'][0]:+.8f},{d['ci95'][1]:+.8f}]")
    d=boot(raw_mm[4]['losses']-raw_mm[8]['losses'],a.bootstrap_repeats,a.seed+500); report['paired']['target_B4_minus_B8']=d
    print(f"\nTarget B4-B8 dCE={d['mean']:+.8f} 95%CI[{d['ci95'][0]:+.8f},{d['ci95'][1]:+.8f}]")
    print('\n[Approx attention MAC/block]')
    report['macs']={'mini_main':{},'main_only':{}}
    for name,m,buds in [('mini_main',mm,[0,2,4,6,8]),('main_only',mo,[2,4,6,8])]:
        for b in buds:
            z=m.estimate_block_attention_macs(b); report['macs'][name][str(b)]=z
            print(f"{name:10s} B={b}: {z['attention_total_macs']/1e6:.3f}M")
    print('\n[Sparse latency]')
    report['latency']={'mini_main':{},'main_only':{}}
    for name,m,buds in [('mini_main',mm,[0,2,4,6,8]),('main_only',mo,[2,4,6,8])]:
        for b in buds:
            z=latency(m,b,device,a.batch_size,a.latency_repeats); report['latency'][name][str(b)]=z
            print(f"{name:10s} B={b}: mean={z['mean_ms']:.3f}ms median={z['median_ms']:.3f}ms")
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nCONSERVATIVE DECISION')
    b4=report['paired']['B4_mini_minus_mainonly']
    if b4['ci95'][1]<0: print('GOOD: B=4 MiniMain is significantly better CE than MainOnly.')
    elif b4['ci95'][0]>0: print('BAD: B=4 MiniMain is significantly worse CE than MainOnly.')
    else: print('AMBIGUOUS: B=4 difference is not statistically clear.')
    print('Also inspect B=0 Mini-only, B4->B8 gap, MACs, and latency.')
    print('Saved:',out)

if __name__=='__main__': main()
