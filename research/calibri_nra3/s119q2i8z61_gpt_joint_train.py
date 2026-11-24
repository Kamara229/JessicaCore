"""Z61 matched GPT-only full native CE, pair CE, pair CE+early EOS penalty.
Three copies of physically saved Z55, one epoch on the same frozen 98 TRAIN.
No external teacher interface or inference; no DEV labels or model selection.
"""
from collections import defaultdict,Counter
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z60_native_pair_objective import (
    pair_targets,token_loss,native_plan,OBJECTIVES as PAIR_OBJECTIVES,
)
from research.calibri_nra3.s119q2i8z59_gpt_compositional_curriculum import curriculum
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer,real_joint_native
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    make_joint_trainable,TOTAL_EXPECTED_TRAINABLE,scoped_reduced_native_head_gain,
)
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import (
    GATES,GT_GATE_LR,REGION_LR,PROJECTION_LR,HEAD_BIAS_LR,CLIP,VOCAB_SIZE,
)
from research.calibri_nra3.s119a_hard_four_goals import check
ARMS=("FULL_CE","FULL_CE_PLUS_GPT_PAIR","FULL_CE_PLUS_GPT_PAIR_EOS")
PLAN_SHA="2db9db9b5e4609e2619fa48e4046d4cd65f9257d2c8ee6d0eb093fb042d9f3cc"
CURRICULUM_SHA="fb68dde68942c9b4b686e88e64a7916d6152fd11e853de989495315fc6de93a5"
EOS_WEIGHT=1.
def prepare(tok):
    plan=native_plan(tok)
    check(plan["native_plan_SHA256"]==PLAN_SHA and plan["curriculum_SHA256"]==CURRICULUM_SHA
          and plan["native_target_positions"]==1618 and plan["contrast_pairs"]==28,
          "exact completed Z60 prereg")
    raw=curriculum()
    rows=[verified_native_answer(tok,r) for r in raw]
    by_id={r["id"]:r for r in rows}
    pairs=defaultdict(list)
    for r in raw:
        if "pair_id" in r:pairs[r["pair_id"]].append(r["id"])
    contrasts={}
    for a,b in pairs.values():
        i,x,y=pair_targets(by_id[a]["answer_native_with_eos"],by_id[b]["answer_native_with_eos"])
        contrasts[a]=(i,x,y);contrasts[b]=(i,y,x)
    check(len(rows)==98 and len(contrasts)==56 and
          sum(len(r["answer_native_with_eos"]) for r in rows)==1618 and
          Counter(r["split"] for r in rows)=={"TRAIN":98},"98 exact legal TRAIN")
    return rows,contrasts
def objective_loss(logits,gold,t,*,arm,contrast=None,vocab=49152,eos=2):
    check(arm in ARMS,"fixed three arms")
    total,ce,pair=token_loss(logits,gold,t,objective=PAIR_OBJECTIVES[0] if arm==ARMS[0] else PAIR_OBJECTIVES[1],
                              contrast=contrast,vocab=vocab,eos=eos)
    penalty=logits.new_zeros(())
    if arm==ARMS[2] and gold[t]!=eos:
        # -log(1-p(EOS)) = softplus(logit_EOS - logsumexp(nonEOS)).
        # Stable even when p(EOS) is very close to one.
        others=torch.cat((logits[0,:eos],logits[0,eos+1:]))
        penalty=EOS_WEIGHT*F.softplus(logits[0,eos]-torch.logsumexp(others,dim=0))/(len(gold)-1)
    return total+penalty,ce,pair,penalty
def region_views(child,full):
    return {"GT12":full.gates,"Core":child.child.child.shadow.unified.core_growth.up,
      "NRA3":child.child.child.shadow.unified.structured_interface.fuse.weight,
      "NPA1":child.child.child.shadow.unified.planning_action_area.state_update.weight_ih,
      "rank8_projection":child.child.up.weight,"language_head":child.language_head.up.weight}
def train(model,sources,bridge,full,expansion,child,rows,contrasts,*,arm):
    check(arm in ARMS and len(rows)==98 and Counter(r["split"] for r in rows)=={"TRAIN":98}
          and len(contrasts)==56,"fixed same TRAIN only")
    selected=make_joint_trainable(child)
    for name,p in full.named_parameters():p.requires_grad_(name=="gates")
    head=child.language_head
    special=[head.down.weight,head.up.weight,head.output_bias]
    nongate=[p for p in selected if id(p) not in {id(q) for q in special}]
    check(sum(p.numel() for p in selected)+full.gates.numel()==TOTAL_EXPECTED_TRAINABLE==368051
          and full.gates.numel()==GATES==12 and full.gates.requires_grad,"exact existing trainable weights")
    opt=torch.optim.Adam([{"params":[full.gates],"lr":GT_GATE_LR},
        {"params":nongate,"lr":REGION_LR},
        {"params":[head.down.weight,head.up.weight],"lr":PROJECTION_LR},
        {"params":[head.output_bias],"lr":HEAD_BIAS_LR}])
    before={k:v.detach().clone() for k,v in region_views(child,full).items()}
    protected={k:v.detach().cpu().clone() for k,v in child.child.child.shadow.unified.nra3_neural_weights.state_dict().items()}
    nll=0.;pair_sum=0.;eos_sum=0.;calls=0;updates=0;contrast_calls=0;early_calls=0
    for r in rows:
        gold=r["answer_native_with_eos"]
        opt.zero_grad(set_to_none=True)
        with scoped_reduced_native_head_gain(head) as witness:
            for t in range(len(gold)):
                logits=real_joint_native(model,sources,bridge,full,expansion,child,r["prefix"]+gold[:t],
                                          r["prompt_features"],contextual=True)
                check(full.last_forward_coverage==tuple(range(12)) and not full._active and full._carry is None,
                      "actual full native 43M all12GT")
                loss,ce,pair,early=objective_loss(logits,gold,t,arm=arm,contrast=contrasts.get(r["id"]))
                check(bool(torch.isfinite(loss)),"finite native training loss")
                loss.backward()
                nll+=float(ce.detach())*len(gold)
                pair_sum+=float(pair.detach());eos_sum+=float(early.detach());calls+=1
                contrast_calls+=int(arm!=ARMS[0] and r["id"] in contrasts and t==contrasts[r["id"]][0])
                early_calls+=int(arm==ARMS[2] and gold[t]!=2)
                del logits,loss,ce,pair,early
        check(witness[0]==len(gold),"exact full native calls")
        torch.nn.utils.clip_grad_norm_(selected+[full.gates],CLIP)
        opt.step();updates+=1
        check(all(bool(torch.isfinite(p).all()) for p in selected+[full.gates])
              and float(full.gates.detach().abs().max())<.5,"finite constrained physical update")
        if updates%14==0:print("Z61_TRAIN_PROGRESS",arm,updates,calls,flush=True)
    changed={k:float(torch.linalg.vector_norm(v.detach()-before[k])) for k,v in region_views(child,full).items()}
    check(all(v>1e-10 for v in changed.values()) and all(torch.equal(v.detach().cpu(),protected[k])
          for k,v in child.child.child.shadow.unified.nra3_neural_weights.state_dict().items()),
          "all intended regions changed, protected old neural weights unchanged")
    check(calls==1618 and updates==98 and contrast_calls==(0 if arm==ARMS[0] else 56)
          and early_calls==(1520 if arm==ARMS[2] else 0),"exact matched training and optional loss counts")
    return {"arm":arm,"native_training_forwards":calls,"Adam_updates":updates,"trainable_parameters":368051,
       "online_mean_native_NLL":nll/calls,"sum_pair_penalty":pair_sum,"sum_early_EOS_penalty":eos_sum,
       "contrast_target_positions":contrast_calls,"nonEOS_penalty_positions":early_calls,
       "actual_region_change_L2":changed,"FP32_calls":0,"terminal_EOS_CE_unchanged":True}
