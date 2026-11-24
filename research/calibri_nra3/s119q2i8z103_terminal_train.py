"""Z103 matched objective-only physical terminal balance. Same exact Z100 budget, data, parent, optimizer, and group order."""
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z88_short_word_train import prepare,cache_rows,verify_z73,PARENT_SHA,GT_SHA,short_metrics,NATIVE_SHA,FRESH_SHA
from research.calibri_nra3.s119q2i8z94_micro_overfit import metrics,logits_for
from research.calibri_nra3.s119q2i8z102_terminal_balance import ARMS,objective,TERMINAL_SHARE
STEPS=1764
GROUPS=14
def partition(cache):
    short=[r for r in cache if r["id"].startswith("S119Q2I8Z87_")]
    replay=[r for r in cache if not r["id"].startswith("S119Q2I8Z87_")]
    check(len(short)==48 and len(replay)==98 and len({r["id"] for r in cache})==146,"all exact TRAIN146 rows")
    check(sum(len(r["gold"]) for r in short)==204 and sum(len(r["gold"]) for r in replay)==1618,"all native targets")
    groups=[replay[i::GROUPS] for i in range(GROUPS)]
    check(all(len(g)==7 for g in groups),"every replay row equally exposed")
    return short,groups
def pack(rows):
    h=torch.cat([e["hidden"] for e in rows]);b=torch.cat([e["base"] for e in rows])
    y=torch.tensor([g for e in rows for g in e["gold"]],dtype=torch.long)
    check(h.shape==(len(y),1,320) and b.shape==(len(y),49152),"aligned full native cache")
    return h,b,y
def train_cached(child,cache,contrasts,arm):
    check(arm in ARMS,"fixed matched arm")
    check(len(contrasts)==56,"existing historical pair manifest retained as data provenance; CE-only objective")
    short,groups=partition(cache)
    h,b,y=pack(short);replay=[pack(g) for g in groups]
    head=child.language_head
    for p in child.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    check(sum(p.numel() for p in head.parameters())==247040,"same existing head")
    other={k:v.detach().cpu().clone() for k,v in child.state_dict().items() if not k.startswith("language_head.")}
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    opt=torch.optim.Adam([{"params":[head.down.weight,head.up.weight],"lr":.003},{"params":[head.output_bias],"lr":.001}])
    history=[];visits=[0]*GROUPS;replay_positions=0
    for i in range(STEPS+1):
        if i in (0,28,112,224,336,448,588,882,1176,1470,1764):
            with torch.no_grad():
                short_m=metrics(logits_for(head,h,b),y)
                replay_m=[metrics(logits_for(head,a,c),d) for a,c,d in replay]
            history.append({"step":i,"short":short_m,"replay":{"positions":1618,
                "NLL":sum(m["NLL"]*m["positions"] for m in replay_m)/1618,
                "top1":sum(m["top1"] for m in replay_m)}})
            print("Z103_PROGRESS "+arm+" "+str(history[-1]),flush=True)
            if i==588 and arm==ARMS[0]:
                check(abs(short_m["NLL"]-1.3017359972000122)<.04 and abs(short_m["top1"]-125)<=3
                      and abs(history[-1]["replay"]["NLL"]-4.05375626325902)<.04,
                      "reproduce Z97 before fixed-budget extension")
        if i==STEPS:break
        opt.zero_grad(set_to_none=True)
        loss=objective(logits_for(head,h,b),y,arm)*.5
        check(bool(torch.isfinite(loss)),"finite short CE")
        loss.backward()
        a,c,d=replay[i%GROUPS]
        loss=objective(logits_for(head,a,c),d,arm)*.5
        check(bool(torch.isfinite(loss)),"finite replay CE")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()),.6,error_if_nonfinite=True)
        opt.step()
        check(all(bool(torch.isfinite(p).all()) for p in head.parameters()),"finite learned head")
        visits[i%GROUPS]+=1;replay_positions+=len(d)
    check(visits==[126]*GROUPS and replay_positions==203868,"exact equal replay exposure")
    changed={k:float(torch.linalg.vector_norm(v.detach()-before[k])) for k,v in head.state_dict().items()}
    check(all(v>0 for v in changed.values()) and all(torch.equal(child.state_dict()[k].detach().cpu(),v) for k,v in other.items()),"actual learned head and frozen body")
    return {"arm":arm,"Adam_updates":STEPS,"short_cached_positions":STEPS*204,"replay_cached_positions":replay_positions,
        "total_cached_training_positions":STEPS*204+replay_positions,"history":history,"head_change_L2":changed,
        "replay_group_visits":visits,"short_and_replay_loss_weights":[.5,.5],"FP32_calls":0,"new_parameters":0}
