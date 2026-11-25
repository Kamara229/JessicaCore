"""Real Z73 cache head CPU/MPS parity. No synthetic representations or new parameters."""
import os,time,torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z120_dose_train import prepare,cache_rows,verify_z73,PARENT_SHA,short_metrics,NATIVE_SHA,FRESH_SHA,TERMINAL_SHARE,GT_SHA,partition,pack,PAIR_SHA,plan_pairs,digest,fresh,pair_penalty,logits_for,CE_ARMS
from research.calibri_nra3.s119q2i8z102_terminal_balance import weights
from research.calibri_nra3.s119a_hard_four_goals import check
ARMS=("CPU","MPS")
GUARDS={"CPU":(.25,.25),"MPS":(.25,.25)}
STEPS=28

def train_cached(child,cache,contrasts,arm):
    check(arm in ARMS and len(contrasts)==56,"matched real cache")
    check(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")=="0" and torch.backends.mps.is_available(),"actual GPU without fallback")
    dev=arm.lower()
    short,groups=partition(cache)
    authored={r["id"]:r for r in fresh()}
    pp=plan_pairs([dict(id=r["id"],family=authored[r["id"]]["family"],word=authored[r["id"]]["completion"],gold=r["gold"]) for r in short])
    check(len(pp)==264 and digest(pp)==PAIR_SHA,"exact pairs")
    pairs=torch.tensor(pp,dtype=torch.long,device=dev)
    def prepare_batch(rows):
        h,b,y=pack(rows)
        # FP64 intermediate is computed on CPU, then cast exactly as historical objective.
        wt=weights(y,CE_ARMS[1]).float()
        idx=(y!=2).nonzero().flatten()
        mask=torch.stack([m for e in rows for m in e["masks"]])
        check(bool(mask.gather(1,y[:,None]).all()),"gold legal")
        return tuple(t.to(dev) for t in (h,b,y,wt,idx,mask))
    sb=prepare_batch(short);rb=[prepare_batch(g) for g in groups]
    head=child.language_head
    for p in child.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    before={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    body={k:v.detach().cpu().clone() for k,v in child.state_dict().items() if not k.startswith("language_head.")}
    head.to(dev)
    opt=torch.optim.Adam([{"params":[head.down.weight,head.up.weight],"lr":.003},{"params":[head.output_bias],"lr":.001}],foreach=False)
    def sync():
        if dev=="mps":torch.mps.synchronize()
    def loss(batch,is_short):
        h,b,y,wt,idx,mask=batch
        logits=logits_for(head,h,b)
        check(logits.device.type==dev,"actual device logits")
        ce=(F.cross_entropy(logits,y,reduction="none")*wt).sum()
        if is_short:return .5*(ce+.25*pair_penalty(logits,pairs))
        sub=logits.index_select(0,idx);target=y.index_select(0,idx)
        eos=F.softplus(sub[:,2]-sub.gather(1,target[:,None]).squeeze(1)+1.).mean()
        invalid=F.softplus(torch.logsumexp(logits.masked_fill(mask,-torch.inf),dim=1)-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),dim=1)).mean()
        return .5*(ce+.25*eos+.25*invalid)
    def step(i):
        opt.zero_grad(set_to_none=True)
        a=loss(sb,True);a.backward()
        b=loss(rb[i%14],False);b.backward()
        check(all(p.grad is not None and p.grad.device.type==dev for p in head.parameters()),"actual device gradients")
        torch.nn.utils.clip_grad_norm_(list(head.parameters()),.6,error_if_nonfinite=True,foreach=False)
        opt.step()
        return a.detach()+b.detach()
    first=float(step(0).cpu())
    sync();t=time.perf_counter()
    for i in range(1,STEPS):v=step(i)
    sync();elapsed=time.perf_counter()-t
    final=float(v.cpu())
    head.to("cpu")
    changes={k:float((v.detach()-before[k]).norm()) for k,v in head.state_dict().items()}
    check(all(v>0 for v in changes.values()) and all(torch.equal(child.state_dict()[k],v) for k,v in body.items()),"real head updated, body frozen")
    check(all(torch.isfinite(v).all() for v in head.state_dict().values()),"finite weights")
    return {"arm":arm,"device":dev,"Adam_updates":STEPS,"timed_updates":STEPS-1,"total_cached_training_positions":STEPS*204+2*1618,"first_loss":first,"last_loss":final,"seconds":elapsed,"head_change_L2":changes,"new_parameters":0}
