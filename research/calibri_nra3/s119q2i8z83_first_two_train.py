"""Z83 matched head-only early gold margin on actual frozen Z73 representations."""
import torch
from research.calibri_nra3.s119q2i8z78_pending_gold_margin import ARMS as OLD_ARMS,EPOCHS,row_loss as baseline_loss,PARENT_SHA,GT_SHA
from torch.nn import functional as F
ARMS=("REVIVED_BASE","REVIVED_FIRST_TWO_MARGIN")
MARGIN=.5
WEIGHT=.5
def early_margin(logits,target,position):
    if position>=2:return logits.new_zeros(())
    if logits.ndim!=2 or logits.shape[0]!=1 or not 0<=target<logits.shape[1]:
        raise ValueError("full native vocabulary and gold token required")
    x=logits[0].float()
    wrong=torch.cat((x[:target],x[target+1:])).max()
    return WEIGHT*F.softplus(wrong-x[target]+MARGIN)/2
def row_loss(head,entry,contrast,arm):
    if arm not in ARMS:raise ValueError("matched arms")
    loss,metrics=baseline_loss(head,entry,contrast,OLD_ARMS[0])
    gold=entry["gold"]
    check(len(gold)>=2,"two actual reference positions")
    extra=loss.new_zeros(())
    if arm==ARMS[1]:
        logits=entry["base"][:2]+.25*head(entry["hidden"][:2])[:,0,:]
        extra=sum(early_margin(logits[t:t+1],gold[t],t) for t in range(2))
    return loss+extra,{**metrics,"margin_sum":float(extra.detach()),"early_positions":2}
def prefix_counts(records):
    check(len(records)==42,"exact TRAIN42 audit")
    counts={"first1":0,"first2":0,"first3":0}
    for row in records:
        p=row["positions"]
        for n in range(1,4):
            counts[f"first{n}"]+=int(len(p)>=n and all(x["target"]==x["top1"] for x in p[:n]))
    return counts

from research.calibri_nra3.s119q2i8z70_cached_head_train import cache_rows
from research.calibri_nra3.s119q2i8z73_head_revival import activation_summary
from research.calibri_nra3.s119q2i8z75_joint_after_head import verify_z73
from research.calibri_nra3.s119a_hard_four_goals import check
def train_cached(child,cache,contrasts,*,arm):
    check(arm in ARMS and len(cache)==98 and len(contrasts)==56,"fixed TRAIN98 matched arms")
    head=child.language_head
    for p in child.parameters():p.requires_grad_(False)
    for p in head.parameters():p.requires_grad_(True)
    check(sum(p.numel() for p in head.parameters())==247040,"fixed existing readout budget")
    other={k:v.detach().cpu().clone() for k,v in child.state_dict().items() if not k.startswith("language_head.")}
    before={k:v.detach().clone() for k,v in head.state_dict().items()}
    initial=activation_summary(head,cache)
    check(initial["exact_zero_units"]==[],"actual recovered four-unit parent")
    opt=torch.optim.Adam([{"params":[head.down.weight,head.up.weight],"lr":.0008},
                          {"params":[head.output_bias],"lr":.000025}])
    epochs=[];updates=0;positions=0
    for epoch in range(EPOCHS):
        totals={"CE_sum":0.,"existing_aux_sum":0.,"margin_sum":0.,"pending_positions":0,"early_positions":0}
        n=0
        for entry in cache:
            opt.zero_grad(set_to_none=True)
            loss,m=row_loss(head,entry,contrasts.get(entry["id"]),arm)
            check(bool(torch.isfinite(loss)),"finite cached loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(head.parameters()),.6,error_if_nonfinite=True)
            opt.step();updates+=1
            check(all(bool(torch.isfinite(p).all()) for p in head.parameters()),"finite actual weights")
            for k in totals:totals[k]+=m[k]
            n+=len(entry["gold"])
        check(n==1618 and totals["pending_positions"]==123 and totals["early_positions"]==196,"exact native manifest")
        check((totals["margin_sum"]==0) if arm==ARMS[0] else (totals["margin_sum"]>0),"matched treatment only")
        epochs.append({"epoch":epoch+1,"mean_online_NLL":totals["CE_sum"]/n,**totals})
        positions+=n
        print("Z83_EPOCH",arm,epochs[-1],flush=True)
    changed={k:float(torch.linalg.vector_norm(v.detach()-before[k])) for k,v in head.state_dict().items()}
    check(updates==392 and positions==6472 and all(v>0 for v in changed.values())
          and all(torch.equal(child.state_dict()[k].detach().cpu(),v) for k,v in other.items()),
          "actual learned head and untouched body")
    return {"arm":arm,"Adam_updates":updates,"cached_training_positions":positions,"trainable_parameters":247040,
            "head_change_L2":changed,"epochs":epochs,"activation_before":initial,"activation_after":activation_summary(head,cache),
            "parent_joint_SHA256":PARENT_SHA,"new_parameters":0,"FP32_source_calls":0}
