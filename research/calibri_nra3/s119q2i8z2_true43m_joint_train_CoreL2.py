"""S119q2i8z2 actual original S142 43M Core+NRA3+NPA1 GT12 native49k
matched 27 source TRAIN Adam A/B on PREDECLARED novel Z2 author21 TRAIN
and never seen internal14 DEV (with previous old six TRAIN rehearsal).

A ordinary full native BPE human CE x3 second/third with normal first/EOS.
B same but λ=2.0 * tanh(existing learned Core onpath gate)^2, applied
EXACTLY ONCE per full TRAIN answer, never on fresh DEV or inference.
The gate is EXISTING learnable child.child.child.midlayer_gates[0];
all 368051 current neural+GT12 trainables identical in A/B, no added
modules/weights, no manual inference masking, no source135m teacher,
no protected 53751 S115/ original S142 pretrained mutation.
This directly tests spent Z/Z1 causal indication of weakly misaligned
Core contribution and will NOT promote any 27-step child based on
small synthetic DEV. Original GT/NRA3/NPA1 and learned native head retained.
"""
from __future__ import annotations
from collections import Counter
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    make_joint_trainable,TOTAL_EXPECTED_TRAINABLE,scoped_reduced_native_head_gain,HEAD_GAIN,
)
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import (
    CONTINUATION_WEIGHT,TOTAL_TRAIN,NEW_TRAIN,ROUNDS,CONDITIONS,
    GATES,GT_GATE_LR,REGION_LR,PROJECTION_LR,HEAD_BIAS_LR,CLIP,
    native_train_weights,VOCAB_SIZE,
)
from research.calibri_nra3.s119q2i8k_real43m_secondthird_teacherKL_train import (
    teacher_kl_only_secondthird_gold_target,
)
STAGE="CALIBRI_S119Q2I8Z2_TRUE43M_FULL_GT12_CORE_GATE_SHRINK_TRAINONLY_CE_AB"
EARLY23_MULTIPLIER=3.0
EARLY23_POSITIONS=(1,2)
CORE_GATE_SHRINK_LAMBDA=2.0

def actual_learned_Core_gate_tanh(child):
    """Gate0 is ONLY the existing on-path Core component, not old pretrained 43M."""
    gates=child.child.child.midlayer_gates
    check(isinstance(gates,torch.nn.Parameter)
          and gates.shape==(3,) and gates.requires_grad
          and bool(torch.isfinite(gates.detach()).all()),
          "existing fully jointly learnable [CORE,structured NRA3,NPA1] onpath gates")
    return torch.tanh(gates[0])

def trainonly_Core_regularizer(child,objective):
    check(objective in ("ALL_NATIVE_CE_EARLY23_X3",
                        "ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE")
          and CORE_GATE_SHRINK_LAMBDA==2.0,
          "predeclared only existing Core gate soft regularizer B")
    gate=actual_learned_Core_gate_tanh(child)
    strength=(CORE_GATE_SHRINK_LAMBDA
              if objective=="ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE"
              else 0.0)
    loss=strength*gate.square()
    check(loss.ndim==0 and bool(torch.isfinite(loss)),
          "genuine learned CORE component gate L2 shrink no inference edits")
    return loss

def train_ce_weights(gold,tok,*,objective):
    """Predeclared same-author non-EOS second/third gold BPE weighting only."""
    check(objective in ("ALL_NATIVE_CE_EARLY23_X3","ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE")
          and type(gold) is list and len(gold)>=3
          and int(gold[-1])==tok.eos_token_id==2
          and all(type(v) is int and 0<=v<VOCAB_SIZE and v!=2 for v in gold[:-1])
          and EARLY23_MULTIPLIER==3.0
          and EARLY23_POSITIONS==(1,2),
          "genuine entire human gold BPE and terminal EOS TRAIN only")
    values=[1.0]*len(gold)
    if objective in ("ALL_NATIVE_CE_EARLY23_X3","ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE"):
        for idx in EARLY23_POSITIONS:
            if idx<len(gold)-1:
                values[idx]=EARLY23_MULTIPLIER
    check(values[-1]==values[0]==1.0
          and all(x in (1.0,3.0) for x in values),
          "never manually mask or boost EOS logits, all positions remain CE")
    return values
def train_joint_true43m_shrink_Core_ab(model,sources,bridge,full,expansion,child,train,tok,
                     *,condition,continuation_weight=CONTINUATION_WEIGHT,
                     expected_total=TOTAL_TRAIN,expected_new=NEW_TRAIN,
                     new_id_prefix="S119Q2I6_",rounds=ROUNDS,
                     teacher_135m=None,teacher_KL_weight=0.0,objective="ALL_NATIVE_CE"):
    check(continuation_weight==1.0 and condition=="GT_RECURRENT"
          and teacher_135m is None and teacher_KL_weight==0.0
          and objective in ("ALL_NATIVE_CE_EARLY23_X3","ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE")
          and rounds==1 and expected_new==21 and expected_total==27
          and new_id_prefix=="S119Q2I8Z2_"
          and len(train)==expected_total
          and Counter(x["split"] for x in train)=={"TRAIN":expected_total}
          and sum(x["id"].startswith(new_id_prefix) for x in train)==expected_new,
          "ONLY Z2 new21 author TRAIN+six old TRAIN, no old or Z2 fresh DEV in gradients")
    selected=make_joint_trainable(child)
    check(sum(p.numel() for p in selected)==TOTAL_EXPECTED_TRAINABLE-GATES,
          "original q2i3 region+head full 368039 trainables")
    for name,p in full.named_parameters():
        p.requires_grad_(name=="gates")
    check(full.gates.requires_grad and full.gates.numel()==GATES
          and len(selected)==len({id(p) for p in selected}),
          "only 12 GT12 scalar gates train, original GT donor projections fixed")
    head=child.language_head
    special=[head.down.weight,head.up.weight,head.output_bias]
    nongate=[p for p in selected if id(p) not in {id(q) for q in special}]
    check(sum(p.numel() for p in nongate)+
          sum(p.numel() for p in special)+GATES==TOTAL_EXPECTED_TRAINABLE,
          "strict matched 368051 trainable region+readout+GT12 params")
    opt=torch.optim.Adam([
        {"params":[full.gates],"lr":GT_GATE_LR},
        {"params":nongate,"lr":REGION_LR},
        {"params":[head.down.weight,head.up.weight],"lr":PROJECTION_LR},
        {"params":[head.output_bias],"lr":HEAD_BIAS_LR},
    ])
    initial={
        "GT12":full.gates.detach().clone(),
        "Core":child.child.child.shadow.unified.core_growth.up.detach().clone(),
        "NRA3":child.child.child.shadow.unified.structured_interface.fuse.weight.detach().clone(),
        "NPA1":child.child.child.shadow.unified.planning_action_area.state_update.weight_ih.detach().clone(),
        "rank8_projection":child.child.up.weight.detach().clone(),
        "language_head":head.up.weight.detach().clone(),
    }
    protected={k:v.detach().cpu().half().clone() for k,v in
        child.child.child.shadow.unified.nra3_neural_weights.state_dict().items()}
    epoch_reports=[];steps=0;forwards=0;continuation_targets=0
    physical_teacher_forwards=0;teacher_KL_values=[]
    early23_weighted_calls=0
    early23_weighted_loss_values=[]
    core_penalty_calls=0
    core_penalty_values=[]
    initial_learned_Core_gate=float(actual_learned_Core_gate_tanh(child).detach())
    for epoch in range(rounds):
        all_ce=[];all_cont_ce=[];all_eos_ce=[];epoch_forwards=0
        for row in train:
            gold=row["answer_native_with_eos"]
            baseline_weights,statuses=native_train_weights(tok,gold,
                continuation_weight=continuation_weight)
            weights=train_ce_weights(gold,tok,objective=objective)
            check(len(weights)==len(baseline_weights)==len(gold)
                  and all(v==1. for v in baseline_weights),
                  "same unweighted all-native CE reference and only early23 W emphasis")
            opt.zero_grad(set_to_none=True)
            with scoped_reduced_native_head_gain(head) as witness:
                for t,target in enumerate(gold):
                    prefix=row["prefix"]+gold[:t]
                    check(prefix[len(row["prefix"]):]==gold[:t]
                          and prefix[:len(row["prefix"])]==row["prefix"]
                          and len(prefix)<160,
                          "causal ORIGINAL full-native decoder uses PREVIOUS gold only")
                    logits=real_joint_native(
                        model,sources,bridge,full,expansion,
                        child,prefix,row["prompt_features"],contextual=True)
                    check(full.last_forward_coverage==tuple(range(12))
                          and not full._active and full._carry is None
                          and logits.shape==(1,VOCAB_SIZE),
                          "REAL original 43M Core12 GT12 native BPE every position")
                    loss=F.cross_entropy(
                        logits,torch.tensor([int(target)],dtype=torch.long))
                    weighted=weights[t]*loss/sum(weights)
                    if t==0:
                        penalty=trainonly_Core_regularizer(child,objective)
                        weighted=weighted+penalty
                        core_penalty_values.append(float(penalty.detach()))
                        core_penalty_calls+=1
                    if weights[t]==3.0:
                        check(t in EARLY23_POSITIONS and int(target)!=int(tok.eos_token_id),
                              "true nonEOS correct second or third native gold target only")
                        early23_weighted_calls+=1
                        early23_weighted_loss_values.append(float(loss.detach()))
                    # PRE-DECLARED optional S119q2i8d SAME physical 49k native
                    # teacher-logit distillation ONLY at second and third
                    # native gold BPE positions. EOS stays human-target CE-only.
                    if (teacher_135m is not None and
                        teacher_kl_only_secondthird_gold_target(t,target,gold,tok.eos_token_id)):
                        check(int(target)!=tok.eos_token_id,
                              "NEVER teacher-distill EOS or teacher-generated QA")
                        from research.calibri_nra3.s119q2i8b_source135m_vs_original43m_native import source_teacher_step
                        soft=source_teacher_step(teacher_135m,prefix)
                        check(soft.shape==logits.shape==(1,VOCAB_SIZE),
                              "source135m and student43m actual native 49152 logits")
                        teacher_p=F.softmax(soft.detach().float()/2.0,dim=-1)
                        student_log=F.log_softmax(logits/2.0,dim=-1)
                        KL=F.kl_div(student_log,teacher_p,reduction="sum")*(2.0**2)
                        check(bool(torch.isfinite(KL)) and float(KL.detach())>=-0.00005,
                              "T2 full original native softteacher KL nonnegative")
                        weighted=weighted+teacher_KL_weight*KL/sum(weights)
                        physical_teacher_forwards+=1
                        teacher_KL_values.append(float(KL.detach()))
                    check(bool(torch.isfinite(weighted)),
                          "finite continuation-weighted real original native loss")
                    weighted.backward()
                    ce=float(loss.detach())
                    all_ce.append(ce)
                    if statuses[t]=="INCOMPLETE_UTF8_NEEDS_CONTINUATION":
                        all_cont_ce.append(ce)
                        continuation_targets+=1
                    if t==len(gold)-1:all_eos_ce.append(ce)
                    forwards+=1;epoch_forwards+=1
                    del logits,loss,weighted
            check(witness[0]==len(gold),
                  "identical predeclared readout scale for EVERY native BPE")
            torch.nn.utils.clip_grad_norm_(selected+[full.gates],CLIP)
            opt.step();steps+=1
            check(bool(torch.isfinite(full.gates).all())
                  and all(bool(torch.isfinite(x).all()) for x in selected)
                  and float(full.gates.detach().abs().max())<.5,
                  "true GT12 and full neural weights remain finite/contained")
        check(all_cont_ce and len(all_eos_ce)==expected_total,
              "actual trained Russian multibyte continuation and EOS targets")
        epoch_reports.append({
            "round":epoch+1,
            "actual_original_full_native_43M_decoder_calls":epoch_forwards,
            "optimizer_updates":expected_total,
            "train_mean_native_CE_ALL_target_BPE_EOS":
                sum(all_ce)/len(all_ce),
            "train_mean_native_CE_UTF8_continuation_target":
                sum(all_cont_ce)/len(all_cont_ce),
            "train_mean_native_EOS_CE":
                sum(all_eos_ce)/len(all_eos_ce),
            "continuation_target_count":len(all_cont_ce),
            "identical_full_native_49152_head_and_3x_CE_weights":True,
        })
    valid_early_teacher_positions=sum(
        teacher_kl_only_secondthird_gold_target(t,target,row["answer_native_with_eos"],
                                                tok.eos_token_id)
        for row in train
        for t,target in enumerate(row["answer_native_with_eos"]))
    check(physical_teacher_forwards==0 and not teacher_KL_values,
          "NO external 135M teacher, train on only original human native BPE+EOS")
    expected_early23=sum(
        sum(idx<len(row["answer_native_with_eos"])-1 for idx in EARLY23_POSITIONS)
        for row in train)*rounds
    check(early23_weighted_calls==expected_early23
          and core_penalty_calls==expected_total*rounds
          and len(core_penalty_values)==expected_total*rounds
          and (all(v==0. for v in core_penalty_values)
               if objective=="ALL_NATIVE_CE_EARLY23_X3" else
               any(v>0 for v in core_penalty_values)),
          "both A/B same human 3x second+third gold BPE CE; B ONLY Core learned gate shrink once per TRAIN answer")
    changed={}
    after={
        "GT12":full.gates,
        "Core":child.child.child.shadow.unified.core_growth.up,
        "NRA3":child.child.child.shadow.unified.structured_interface.fuse.weight,
        "NPA1":child.child.child.shadow.unified.planning_action_area.state_update.weight_ih,
        "rank8_projection":child.child.up.weight,
        "language_head":head.up.weight,
    }
    for key,v in after.items():
        delta=float(torch.linalg.vector_norm(v.detach()-initial[key]))
        changed[key]=delta
        check(delta>1e-10,
              "ALL GreenTunnel/NRA3/NPA1/CORE/rank8/head must learn: "+key)
    check(all(torch.equal(v.detach().cpu().half(),protected[k])
              for k,v in child.child.child.shadow.unified.nra3_neural_weights.state_dict().items())
          and steps==expected_total*rounds,
          "prior S115 53,751 original neural FP16 unchanged, 3 true rounds")
    return {
        "stage":STAGE,
        "objective":objective,
        "trained_full49k_head_gain":HEAD_GAIN,
        "train_only_true_SECOND_THIRD_native_gold_CE_weight":EARLY23_MULTIPLIER,
        "Core_gate_L2_lambda_PER_27TRAIN_answer":(
            CORE_GATE_SHRINK_LAMBDA if
            objective=="ALL_NATIVE_CE_EARLY23_X3_SHRINK_CORE" else 0.),
        "Core_gate_training_ONLY_regularizer_count":core_penalty_calls,
        "learned_existing_CORE_component_tanh_GATE_before_27TRAIN":
            initial_learned_Core_gate,
        "learned_existing_CORE_component_tanh_GATE_after_27TRAIN":
            float(actual_learned_Core_gate_tanh(child).detach()),
        "Core_trainonly_L2_average_27TRAIN":sum(core_penalty_values)/len(core_penalty_values),
        "Core_penalty_applied_to_OTHER_regions_or_native_vocabulary_or_OWN":False,
        "trained_true_SECOND_THIRD_BPE_nonEOS_targets_at_3x":early23_weighted_calls,
        "TRAIN_online_mean_gold_second_third_CE_not_DEV":(
            sum(early23_weighted_loss_values)/len(early23_weighted_loss_values)
            if early23_weighted_loss_values else None),
        "EOS_native_target_still_full_CE_and_never_masked":True,
        "all_nonEOS_native_vocab_logits_real49152_no_token_hacks":True,
        "condition":condition,
        "rounds":epoch_reports,
        "trained_original_decoder_calls":forwards,
        "native_byte_continuation_weight":continuation_weight,
        "count_weighted_native_byte_continuation_targets_across_all_epochs":
            continuation_targets,
        "real_all_neural_and_gt_Adam_updates":steps,
        "actual_gt12_core_NRA3_NPA1_native_head_change_l2":changed,
        "trainable_parameter_count":TOTAL_EXPECTED_TRAINABLE,
        "source135m_teacher_KL_fixed_T2_secondthird_gold_BPE_only":teacher_KL_weight,
        "physical_SOURCE135M_native_forward_calls_at_gold_secondthird_BPE_ONLY":physical_teacher_forwards,
        "mean_source135m_T2_4xKL_at_gold_secondthird_BPE":(
            sum(teacher_KL_values)/len(teacher_KL_values) if teacher_KL_values else None),
        "all_teacher_logits_softlabels_only_and_never_EOS":True,
        "source_135M_KL_ONLY_at_gold_secondthird_and_NEVER_EOS":True,
        "expected_per_epoch_early23_teacher_eligible_native_BPE":valid_early_teacher_positions,
        "actual_registered_new_TRAIN_rows":expected_new,
        "actual_registered_total_TRAIN_rows":expected_total,
        "actual_joint_epochs":rounds,
        "no_original_or_protected_weights_modified":True,
    }

