"""Z118 matched Z108 context head training with bundled historical EOS and UTF8 guards."""
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z88_short_word_train import prepare,cache_rows,verify_z73,PARENT_SHA,GT_SHA,short_metrics,NATIVE_SHA,FRESH_SHA
from research.calibri_nra3.s119q2i8z94_micro_overfit import metrics,logits_for
from research.calibri_nra3.s119q2i8z102_terminal_balance import objective,TERMINAL_SHARE,ARMS as CE_ARMS
from research.calibri_nra3.s119q2i8z107_context_discrimination import ARMS as CONTEXT_ARMS,plan_pairs,short_objective,pair_penalty
ARMS=("CONTEXT_CONTROL","CONTEXT_REPLAY_EOS_UTF8")
from research.calibri_nra3.s119q2i8z87_short_word_curriculum import fresh
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import digest
PAIR_SHA="3825abbee006ac504bdeb7f809cd7a8a24dd606ced03a4cb23ee3512b119b08e"
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
def replay_guard(logits,gold,mask):
    check(mask.shape==logits.shape and mask.dtype==torch.bool,"full legal reference transition masks")
    check(bool(mask.gather(1,gold[:,None]).all()),"all gold targets legal")
    nonterminal=gold!=2
    eos=F.softplus(logits[nonterminal,2]-logits[nonterminal].gather(1,gold[nonterminal,None]).squeeze(1)+1.).mean()
    invalid=F.softplus(torch.logsumexp(logits.masked_fill(mask,-torch.inf),dim=1)-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),dim=1)).mean()
    return eos+.25*invalid

def train_cached(child,cache,contrasts,arm):
    check(arm in ARMS,"fixed matched arm")
    check(len(contrasts)==56,"existing historical pair manifest retained as data provenance; CE-only objective")
    short,groups=partition(cache)
    authored={r["id"]:r for r in fresh()}
    pair_list=plan_pairs([dict(id=r["id"],family=authored[r["id"]]["family"],word=authored[r["id"]]["completion"],gold=r["gold"]) for r in short])
    check(len(pair_list)==264 and digest(pair_list)==PAIR_SHA,"exact precommitted Z107 native pairs")
    pairs=torch.tensor(pair_list,dtype=torch.long)
    h,b,y=pack(short);replay=[pack(g) for g in groups]
    masks=[torch.stack([m for e in g for m in e["masks"]]) for g in groups]
    check(all(m.shape==r[1].shape for m,r in zip(masks,replay)),"exact reference mask alignment")
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
                short_logits=logits_for(head,h,b)
                short_m=metrics(short_logits,y)
                short_m["pair_penalty"]=float(pair_penalty(short_logits,pairs))
                replay_m=[metrics(logits_for(head,a,c),d) for a,c,d in replay]
            history.append({"step":i,"short":short_m,"replay":{"positions":1618,
                "NLL":sum(m["NLL"]*m["positions"] for m in replay_m)/1618,
                "top1":sum(m["top1"] for m in replay_m)}})
            print("Z118_PROGRESS "+arm+" "+str(history[-1]),flush=True)
        if i==STEPS:break
        opt.zero_grad(set_to_none=True)
        loss=short_objective(logits_for(head,h,b),y,CONTEXT_ARMS[1],pairs)*.5
        check(bool(torch.isfinite(loss)),"finite short CE")
        loss.backward()
        a,c,d=replay[i%GROUPS]
        logits=logits_for(head,a,c)
        loss=objective(logits,d,CE_ARMS[1])*.5
        if arm==ARMS[1]:loss=loss+.5*replay_guard(logits,d,masks[i%GROUPS])
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
        "replay_group_visits":visits,"short_and_replay_loss_weights":[.5,.5],"FP32_calls":0,"new_parameters":0,"pair_plan_SHA256":PAIR_SHA,"pairs":264,"replay_EOS_margin_weight":1.0 if arm==ARMS[1] else 0.0,"replay_UTF8_invalid_mass_weight":.25 if arm==ARMS[1] else 0.0}
