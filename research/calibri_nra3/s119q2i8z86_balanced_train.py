"""Z86 actual matched physical execution of preregistered Z85."""
import torch
from research.calibri_nra3.s119q2i8z85_balanced_margin import ARMS,row_loss,PARENT_SHA,GT_SHA,PENDING,COMPLETE_NONTERMINAL
from research.calibri_nra3.s119q2i8z78_pending_gold_margin import EPOCHS
from research.calibri_nra3.s119q2i8z83_first_two_train import prefix_counts
def validate_epoch(t,arm,n):
    if arm not in ARMS or n!=1618 or t["pending_positions"]!=PENDING or t["complete_nonterminal_positions"]!=COMPLETE_NONTERMINAL:
        raise ValueError("exact globally balanced native manifest")
    if not ((t["pending_margin"]==t["complete_margin"]==0) if arm==ARMS[0] else (t["pending_margin"]>0 and t["complete_margin"]>0)):
        raise ValueError("both state penalties active only in treatment")

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
        totals={"CE_sum":0.,"existing_aux_sum":0.,"pending_margin":0.,"complete_margin":0.,"pending_positions":0,"complete_nonterminal_positions":0}
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
        validate_epoch(totals,arm,n)
        epochs.append({"epoch":epoch+1,"mean_online_NLL":totals["CE_sum"]/n,**totals})
        positions+=n
        print("Z86_EPOCH",arm,epochs[-1],flush=True)
    changed={k:float(torch.linalg.vector_norm(v.detach()-before[k])) for k,v in head.state_dict().items()}
    check(updates==392 and positions==6472 and all(v>0 for v in changed.values())
          and all(torch.equal(child.state_dict()[k].detach().cpu(),v) for k,v in other.items()),
          "actual learned head and untouched body")
    return {"arm":arm,"Adam_updates":updates,"cached_training_positions":positions,"trainable_parameters":247040,
            "head_change_L2":changed,"epochs":epochs,"activation_before":initial,"activation_after":activation_summary(head,cache),
            "parent_joint_SHA256":PARENT_SHA,"new_parameters":0,"FP32_source_calls":0}
