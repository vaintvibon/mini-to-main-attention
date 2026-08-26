# -*- coding: utf-8 -*-
"""Train 4-Mini/8-Main budget model and matched Main-only baseline."""
import argparse, json, random, time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.budgeted_mini_main_v2 import BudgetedMiniMainViTV2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['mini_main','main_only'], required=True)
    p.add_argument('--data-dir', default='/content/cifar10')
    p.add_argument('--output-dir', default='./outputs/budgeted_v2')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--embed-dim', type=int, default=192)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--main-heads', type=int, default=8)
    p.add_argument('--mini-heads', type=int, default=4)
    p.add_argument('--mini-head-dim', type=int, default=16)
    p.add_argument('--direct-k', type=int, default=2)
    p.add_argument('--pool-ratio', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--kd-alpha', type=float, default=0.3)
    p.add_argument('--kd-temp', type=float, default=2.0)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--route-tau-start', type=float, default=1.5)
    p.add_argument('--route-tau-end', type=float, default=0.5)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--train-size', type=int, default=40000)
    p.add_argument('--val-size', type=int, default=5000)
    p.add_argument('--max-train-batches', type=int, default=0)
    p.add_argument('--max-val-batches', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=1)
    return p.parse_args()


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_data(args):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616)),
    ])
    train_base = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_tf)
    eval_base = datasets.CIFAR10(args.data_dir, train=True, download=False, transform=eval_tf)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(50000, generator=g).tolist()
    if args.train_size > 40000 or args.val_size > 5000:
        raise ValueError('Preserve [45000,50000) heldout: train<=40000, val<=5000.')
    train_ds = Subset(train_base, perm[:args.train_size])
    val_ds = Subset(eval_base, perm[40000:40000+args.val_size])
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed+100),
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def build_model(args):
    return BudgetedMiniMainViTV2(
        img_size=32, patch_size=4, num_classes=10,
        embed_dim=args.embed_dim, depth=args.depth,
        main_heads=args.main_heads, mini_heads=args.mini_heads,
        mini_head_dim=args.mini_head_dim, direct_k=args.direct_k,
        pool_ratio=args.pool_ratio, mode=args.mode,
        route_tau=args.route_tau_start,
    )


def kd_loss(student, teacher, temp):
    return F.kl_div(
        F.log_softmax(student/temp, dim=-1),
        F.softmax(teacher/temp, dim=-1),
        reduction='batchmean',
    ) * temp * temp


def autocast_ctx(device, enabled):
    dtype = torch.float16 if device.type == 'cuda' else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def choose_budgets(mode):
    if mode == 'mini_main':
        return 8, 0, random.choice([2,4,6])
    return 8, 2, random.choice([4,6])


def route_tau(epoch, epochs, start, end):
    if epochs <= 1: return end
    t = (epoch-1)/(epochs-1)
    return start * ((end/start)**t)


def train_epoch(model, loader, optimizer, scaler, device, args):
    model.train()
    sums = dict(loss=0., full_ce=0., low_ce=0., mid_ce=0.)
    corr = dict(full=0, low=0, mid=0); total = 0
    for bi,(x,y) in enumerate(loader):
        if args.max_train_batches and bi >= args.max_train_batches: break
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        full_b, low_b, mid_b = choose_budgets(args.mode)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, args.amp):
            lf = model(x, budget=full_b)
            ll = model(x, budget=low_b)
            lm = model(x, budget=mid_b)
            cf = F.cross_entropy(lf,y); cl=F.cross_entropy(ll,y); cm=F.cross_entropy(lm,y)
            loss = (cf+cl+cm)/3.0
            if args.kd_alpha > 0:
                teacher = lf.detach()
                kd = (kd_loss(ll,teacher,args.kd_temp)+kd_loss(lm,teacher,args.kd_temp))/2
                loss = loss + args.kd_alpha*kd
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer); scaler.update()
        n=y.numel(); total += n
        sums['loss'] += loss.item()*n; sums['full_ce'] += cf.item()*n
        sums['low_ce'] += cl.item()*n; sums['mid_ce'] += cm.item()*n
        corr['full'] += lf.argmax(-1).eq(y).sum().item()
        corr['low'] += ll.argmax(-1).eq(y).sum().item()
        corr['mid'] += lm.argmax(-1).eq(y).sum().item()
    out={k:v/max(total,1) for k,v in sums.items()}
    out.update({f'{k}_acc':100*v/max(total,1) for k,v in corr.items()})
    return out


@torch.no_grad()
def eval_budget(model, loader, budget, device, args):
    model.eval(); loss=0.; correct=0; total=0; active=0; block_samples=0
    for bi,(x,y) in enumerate(loader):
        if args.max_val_batches and bi >= args.max_val_batches: break
        x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
        with autocast_ctx(device, args.amp):
            logits, infos = model(x, budget=budget, return_info=True)
            loss += F.cross_entropy(logits,y,reduction='sum').item()
        correct += logits.argmax(-1).eq(y).sum().item(); total += y.numel()
        for info in infos:
            active += int(info['active_main_mask'].sum().item())
            block_samples += info['active_main_mask'].shape[0]
    return {'ce':loss/max(total,1), 'acc':100*correct/max(total,1),
            'avg_active_main_heads':active/max(block_samples,1)}


@torch.no_grad()
def eval_all(model, loader, device, args):
    budgets=[0,2,4,6,8] if args.mode=='mini_main' else [2,4,6,8]
    out={str(b):eval_budget(model,loader,b,device,args) for b in budgets}
    out['mean_ce']=sum(out[str(b)]['ce'] for b in budgets)/len(budgets)
    return out


def save_ckpt(path, model, args, epoch, val):
    torch.save({
        'model':model.state_dict(),
        'config':{
            'img_size':32,'patch_size':4,'num_classes':10,
            'embed_dim':args.embed_dim,'depth':args.depth,
            'main_heads':args.main_heads,'mini_heads':args.mini_heads,
            'mini_head_dim':args.mini_head_dim,'direct_k':args.direct_k,
            'pool_ratio':args.pool_ratio,'mode':args.mode,
        },
        'epoch':epoch,'val':val,'args':vars(args),
    }, path)


def main():
    args=parse_args(); seed_all(args.seed)
    if (args.main_heads,args.mini_heads)!=(8,4):
        raise ValueError('First experiment is fixed to Mini=4, Main=8.')
    if args.embed_dim % 8 != 0: raise ValueError('embed_dim must be divisible by 8')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('='*100); print('4 MINI / 8 MAIN BUDGET TRAIN'); print('='*100)
    print('mode:',args.mode,'device:',device)
    if device.type!='cuda': print('WARNING: GPU not enabled; full run will be slow.')
    print('Official CIFAR-10 test: NOT USED')
    print('Heldout permutation [45000,50000): NOT USED')
    train_loader,val_loader=build_data(args)
    model=build_model(args).to(device)
    print(f'params={sum(p.numel() for p in model.parameters())/1e6:.3f}M')
    for b in ([0,2,4,6,8] if args.mode=='mini_main' else [2,4,6,8]):
        m=model.estimate_block_attention_macs(b)
        print(f"B={b}: attention MAC/block={m['attention_total_macs']/1e6:.3f}M "
              f"(mini={m['mini_macs']/1e6:.3f}, main={m['main_macs']/1e6:.3f})")

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(args.epochs,1))
    scaler=torch.amp.GradScaler('cuda',enabled=args.amp and device.type=='cuda')
    outdir=Path(args.output_dir)/args.mode; outdir.mkdir(parents=True,exist_ok=True)
    best=float('inf'); best_epoch=-1
    metrics=outdir/'metrics.jsonl'
    if metrics.exists(): metrics.unlink()

    for epoch in range(1,args.epochs+1):
        tau=route_tau(epoch,args.epochs,args.route_tau_start,args.route_tau_end)
        model.set_route_temperature(tau); t0=time.time()
        tr=train_epoch(model,train_loader,opt,scaler,device,args)
        va=eval_all(model,val_loader,device,args) if (epoch%args.eval_every==0 or epoch==args.epochs) else None
        rec={'epoch':epoch,'tau':tau,'lr':opt.param_groups[0]['lr'],'sec':time.time()-t0,'train':tr,'val':va}
        with metrics.open('a',encoding='utf-8') as f: f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        print(f"\nEpoch {epoch:03d}/{args.epochs} tau={tau:.3f} time={rec['sec']:.1f}s")
        print(f"Train loss={tr['loss']:.4f} | full={tr['full_ce']:.4f}/{tr['full_acc']:.2f}% "
              f"low={tr['low_ce']:.4f}/{tr['low_acc']:.2f}% mid={tr['mid_ce']:.4f}/{tr['mid_acc']:.2f}%")
        if va is not None:
            for b in ([0,2,4,6,8] if args.mode=='mini_main' else [2,4,6,8]):
                m=va[str(b)]; print(f"  Val B={b}: CE={m['ce']:.4f} Acc={m['acc']:.2f}% active={m['avg_active_main_heads']:.2f}")
            print(f"  mean CE={va['mean_ce']:.6f}")
            if va['mean_ce'] < best:
                best=va['mean_ce']; best_epoch=epoch
                save_ckpt(outdir/'best.pt',model,args,epoch,va); print('  -> BEST')
        save_ckpt(outdir/'last.pt',model,args,epoch,va); sched.step()

    print('\nDONE best epoch=',best_epoch,'best mean CE=',best)
    print('best checkpoint:',outdir/'best.pt')
    print('Official CIFAR-10 test remains untouched.')

if __name__=='__main__': main()
