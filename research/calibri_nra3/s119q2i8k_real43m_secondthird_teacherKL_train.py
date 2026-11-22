"""S119q2i8k TRUE original43M joint BPE CE with selective source135M
teacher KL on human-gold second/third positions only; never EOS or DEV.
Research-only matched 368051 trainable Core/GT12/NRA3/NPA1/native head.
"""
from __future__ import annotations
from collections import Counter
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import make_joint_trainable,TOTAL_EXPECTED_TRAINABLE,scoped_reduced_native_head_gain
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import CONTINUATION_WEIGHT,TOTAL_TRAIN,NEW_TRAIN,ROUNDS,CONDITIONS,GATES,GT_GATE_LR,REGION_LR,PROJECTION_LR,HEAD_BIAS_LR,CLIP,native_train_weights,VOCAB_SIZE

def train_true_joint_early23(model,sources,bridge,full,expansion,child,train,tok,
                     *,condition,continuation_weight=CONTINUATION_WEIGHT,
                     expected_total=TOTAL_TRAIN,expected_new=NEW_TRAIN,
                     new_id_prefix="S119Q2I6_",rounds=ROUNDS,
                     teacher_135m=None,teacher_KL_weight=0.0):
    check(continuation_weight in (1.0,3.0) and condition in CONDITIONS and
          ((teacher_135m is None and teacher_KL_weight==0.0) or
           (teacher_135m is not None and teacher_KL_weight==0.25 and
            all(not p.requires_grad for p in teacher_135m.parameters()))) and
          rounds in (1,2,3) and expected_new>=NEW_TRAIN and
          expected_total>=expected_new and
          len(train)==expected_total and
          Counter(x["split"] for x in train)=={"TRAIN":expected_total}
          and sum(x["id"].startswith(new_id_prefix) for x in train)==expected_new,
          "ONLY registered fresh new and old TRAIN, never scored DEV labels")
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
    for epoch in range(rounds):
        all_ce=[];all_cont_ce=[];all_eos_ce=[];epoch_forwards=0
        for row in train:
            gold=row["answer_native_with_eos"]
            weights,statuses=native_train_weights(tok,gold,
                continuation_weight=continuation_weight)
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
                    # PRE-DECLARED optional S119q2i8d SAME physical 49k native
                    # teacher-logit distillation ONLY at second and third
                    # native gold BPE positions. EOS stays human-target CE-only.
                    if (teacher_135m is not None and
                        t in (1,2) and t<len(gold)-1):
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
    check(physical_teacher_forwards==(expected_total*2*rounds if teacher_135m is not None else 0),
          'exact two physical frozen 135M teacher forwards at gold second/third per TRAIN item')
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
        "actual_registered_new_TRAIN_rows":expected_new,
        "actual_registered_total_TRAIN_rows":expected_total,
        "actual_joint_epochs":rounds,
        "no_original_or_protected_weights_modified":True,
    }

