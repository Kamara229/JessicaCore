"""S119q2i8r physical true original43M language head ONLY neural learning.

After causal EOS overamplification proof #951–#953, keep architecture
Core12, 12 Green Tunnel gates, NRA3, NPA1 and LOWRANK=4 native 49152
head fixed. Starting SAME pretrained physical K early23-KL FP16 child,
head-only A=all-gold full native CE, B=CE + differentiable TRAIN-only
correct EOS vs gold ranking loss. 28 never-trained author E70 TRAIN + six
old TRAIN rehearsal, 34 full Adam optimizer updates per matched arm.
No original / protected learned weights / GT12 / nonhead modification.
Precommitted new 14 author DEV + old six are ONLY own-native inference,
never gradients, no independent benchmark, no acceptance / LIVE.
"""
from __future__ import annotations
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import torch

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import persist_half
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX,LOCAL_CKPT as Q2D_LOCAL,REPORT as Q2D_REPORT,VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import load_q2d_proof
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import load_q2d_trained_child
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX,REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import load_physical_q2f_child
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_ru_rows,provenance as old_ru_provenance,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    STAGE as Q2I1_STAGE,CONTEXT_CKPT as Q2I1_CTX,REPORT as Q2I1_REPORT,
    verified_native_answer,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,set_gt_route,safe_gt_gate_sidecar,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    STAGE as Q2I3_STAGE,REPORT as Q2I3_REPORT,
    RECURRENT_CKPT,TOTAL_EXPECTED_TRAINABLE,scoped_reduced_native_head_gain,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import check_trained_joint
from research.calibri_nra3.s119q2i8k_real43m_secondthird_teacherKL_train import train_true_joint_early23
from research.calibri_nra3.s119q2i7_real43m_utf8_weight_causal_ablation import (
    STAGE as Q2I7_STAGE,REPORT as Q2I7_REPORT,
)
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    STAGE as SOURCE_STAGE,KINDS,
    rows as previously_registered_rows,verify as source_verify,
)
from research.calibri_nra3.s119q2i8a_broader_russian_synthetic_source import (
    verify as q2i8a_source_verify,
)
from research.calibri_nra3.s119q2i8f_native_bpe_readiness import audit_tokenizer
from research.calibri_nra3.s119q2i8d_real43m_source_softkl_joint_train import (
    STAGE as PREV_STAGE,REPORT as PREV_REPORT,
    CE_FP16 as PREV_CE_FP16,CE_GATES as PREV_CE_GATES,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import OLD_PREFIX
from safetensors.torch import load_file
from research.calibri_nra3.s119q2g_fresh_own_token_source import normalized
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    native_bytelevel_full_utf8_status,first_divergence,
)

from research.calibri_nra3.s119q2i8a_real43m_broader_russian_joint import (
    STAGE as Q2I8A_STAGE,REPORT as Q2I8A_REPORT,
    RECURRENT_FP16 as Q2I8A_REC,RECURRENT_GT as Q2I8A_GT,
    verify_physical_1x_parent,score,
)
from research.calibri_nra3.s119q2i8b_source135m_vs_original43m_native import (
    physical_q2i8a_fitted_child,physical_q2i8a_gates,
)
from research.calibri_nra3.s119q2i8g_real43m_multiword_native_joint_ab import (
    STAGE as Q2I8G_PREV_STAGE,REPORT as Q2I8G_PREV_REPORT,
    CE_FP16 as Q2I8G_PREV_CE_FP16,CE_GATES as Q2I8G_PREV_CE_GATES,
)
from research.calibri_nra3.s119q2i8j_original43m_native_goldsecond_rank import (
    full_saved_child,full_saved_gt,
)
from research.calibri_nra3.s119q2i8k_new_ru_multiword_dev_source import (
    STAGE as D_SOURCE_STAGE,ORIGIN as D_ORIGIN,
    rows as new_holdout,verify as holdout_verify,
)


from research.calibri_nra3.s119q2i8r_new_ru_DEV_eos_source import (
    STAGE as R_DEV_STAGE,ORIGIN as R_DEV_ORIGIN,
    rows as r_dev_rows,verify as r_dev_verify,
)
from research.calibri_nra3.s119q2i8r_headonly_fullnative_eos_margin_train import (
    head_only_train,TRAINABLE_HEAD,EXPECTED_TRAIN,MARGIN_LAMBDA,GAIN,STAGE as HEAD_STAGE,
)
from research.calibri_nra3.s119q2i8o_native_head_gain_physical_ablation import (
    scoped_exact_native_gain,
)
from research.calibri_nra3.s119q2i8k_real43m_secondthird_joint_train_ab import (
    STAGE as K_STAGE,REPORT as K_REPORT,KL_FP16 as K_KL_FP16,
    KL_GATES as K_KL_GATES,native_new_train,own_token_dev_long,
)
from research.calibri_nra3.s119q2i8g_real43m_multiword_native_joint_ab import (
    physical_q2i8d_CE_warmstart,
)
from research.calibri_nra3.s119q2i8j_original43m_native_goldsecond_rank import (
    full_saved_child,full_saved_gt,
)

STAGE="CALIBRI_S119Q2I8R_REAL43M_34_UNUSED_TRAIN_RANK4_HEAD_EOS_MARGIN_CAUSAL_AB"
REPORT="S119Q2I8R_REAL43M_RANK4_HEAD_ONLY_GAIN1_EOS_CE_VS_MARGIN_G0.json"
CE_FP16="S119Q2I8R_REAL43M_NATIVE_HEAD_GAIN1_CE_ONLY_FP16_G0.safetensors"
MARGIN_FP16="S119Q2I8R_REAL43M_NATIVE_HEAD_GAIN1_CE_PLUS_EOS_MARGIN_FP16_G0.safetensors"
NEW_HELDOUT_DEV=14
OLD_SPENT_DEV=6
TRAIN_ROWS=34
NEW_TRAIN=28
NATIVE_VOCAB=49152
SEED=119157
ARMS=("BASELINE_CE_ONLY","TRAIN_ONLY_SUPERVISED_EOS_MARGIN")


def pack_r_author_dev(tok,row):
    check(row["split"]=="DEV"
          and row["origin"]==R_DEV_ORIGIN
          and row["id"].startswith("S119Q2I8R_")
          and row["kind"] in KINDS,
          "only new 14 predeclared author DEV before ANY optimizer")
    clean=dict(row)
    clean["origin"]="original_newly_authored_examples_no_third_party_corpus"
    item=verified_native_answer(tok,clean)
    check(item["split"]=="DEV"
          and tok.decode(item["answer_native_without_eos"],
                         skip_special_tokens=True)==row["completion"],
          "true original49152 native BPE target and EOS no corruption")
    item["tokenizer"]=tok
    item["eos_native_id"]=int(tok.eos_token_id)
    return item


def eval_own_gain(tok,model,sources,bridge,full,expansion,child,gt,dev,
                  *,gain,arm):
    check(gain in (0.25,1.0)
          and len(dev) in (NEW_HELDOUT_DEV,OLD_SPENT_DEV)
          and all(x["split"]=="DEV" for x in dev)
          and child.output_gate==0
          and child.candidate_accepted is False,
          "14 NEW or 6 OLD read-only original full-native self-generation")
    set_gt_route(full,"GT_RECURRENT",gt)
    output=[]
    for item in dev:
        with scoped_exact_native_gain(child.language_head,float(gain)) as counts:
            generated=own_token_dev_long(
                model,sources,bridge,full,expansion,child,item,
                condition="GT_RECURRENT",max_generate=40)
        check(counts[0]==generated["original_full_decoder_calls"],
              "each true OWN BPE requires exactly one full original43M 49152 head")
        ids=generated["generated_own_native_BPE_ids"]
        utf8=native_bytelevel_full_utf8_status(
            tok,ids)["entire_original_native_BPE_prefix_UTF8_status"]
        d=first_divergence(item["answer_native_with_eos"],ids)
        output.append({
            **generated,
            "eval_arm":arm,"head_gain":gain,
            "original_prompt_SHA256":item["prompt_sha256"],
            "entire_generated_UTF8_state":utf8,
            "valid_full_UTF8_bytes":utf8=="VALID_COMPLETE_UTF8",
            "true_native_full_EOS_correct":
                bool(generated["strict_finished_full_answer_exact"]),
            "first_gold_BPE_correct":bool(d["first_native_BPE_correct"]),
            "first_native_correct_but_wrong_second":bool(
                d["first_native_correct_but_wrong_OR_missing_second"]),
            "first_divergence_position_1based":
                d["first_wrong_or_missing_native_BPE_position_1based"],
            "source_internal_DEV_NOT_independent_blind":True,
        })
    return {
        "OWN_samples":output,
        "scores":{
            "DEV_n":len(output),
            "correct_full_native_ANSWER_and_EOS":sum(x["true_native_full_EOS_correct"] for x in output),
            "true_native_EOS_emitted":sum(x["native_EOS"] for x in output),
            "full_UTF8_valid_own_native":sum(x["valid_full_UTF8_bytes"] for x in output),
            "correct_first_native_BPE":sum(x["first_gold_BPE_correct"] for x in output),
            "first_correct_but_second_wrong":sum(x["first_native_correct_but_wrong_second"] for x in output),
            "real_original43M_OWN_forwards":sum(x["original_full_decoder_calls"] for x in output),
        }
    }


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,q2i7_dir,q2i8a_dir,
        q2i8d_dir,q2i8f_report,q2i8g_dir,q2i8k_dir,out):
    check(not out.exists(),"do not overwrite old research model or published result")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=source_verify()
    rdev=r_dev_verify()
    check(source["stage"]==SOURCE_STAGE
          and source["new_TRAIN_only"]==70
          and rdev["stage"]==R_DEV_STAGE
          and rdev["exact_14_new_source_DEV"]==NEW_HELDOUT_DEV,
          "only predeclared unseen internal author DEV and 70 TRAIN")
    old_rehearsal=[v for v in old_ru_rows("TRAIN")
                   if int(v["id"].rsplit("_",1)[-1])==0]
    old_dev=[v for v in old_ru_rows("DEV")
             if int(v["id"].rsplit("_",1)[-1])==6]
    check(len(old_rehearsal)==6 and len(old_dev)==6,
          "six old TRAIN rehearsal vs six old spent DEV never train on DEV")
    evidence=load_q2d_proof(q2d_dir/Q2D_REPORT,
        q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    q2i3=json.loads((q2i3_dir/Q2I3_REPORT).read_text(encoding="utf8"))
    q2i7=json.loads((q2i7_dir/Q2I7_REPORT).read_text(encoding="utf8"))
    q2i8a=json.loads((q2i8a_dir/Q2I8A_REPORT).read_text(encoding="utf8"))
    p7=q2i7_dir/Q2I7_REPORT
    p8=q2i8a_dir/Q2I8A_REPORT
    check(q2i8a["original_S93_sha256"]==sha(s93)
          and q2i8a["original_S142_real43M_GT_sha256"]==sha(s142)
          and q2i8a["prior_true_Q2I7_1X_control_physical_run_SHA256"]==sha(p7)
          and not q2i8a["candidate_accepted"]
          and q2i8a["output_gate"]==0,
          "true original prior 43M parent and saved protected physical provenance")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==NATIVE_VOCAB and tok.eos_token_id==2,
          "identical original SmolLM2 49152 bytelevel native BPE and EOS id2")
    readiness=json.loads(q2i8f_report.read_text(encoding="utf8"))
    check(readiness==audit_tokenizer(tok)
          and readiness["TRAIN_source_SHA256"]==source["TRAIN_sha256"]
          and len(readiness["selected_balanced_21_TRAIN_IDS"])==21,
          "strict original tokenizer audit and first 21 previously consumed TRAIN")
    prevg=json.loads((q2i8g_dir/Q2I8G_PREV_REPORT).read_text(encoding="utf8"))
    prevk=json.loads((q2i8k_dir/K_REPORT).read_text(encoding="utf8"))
    check(prevg["stage"]==Q2I8G_PREV_STAGE and prevk["stage"]==K_STAGE
          and not prevg["candidate_accepted"]
          and not prevk["candidate_accepted"]
          and prevg["output_gate"]==prevk["output_gate"]==0
          and prevk["original_S142_43M_sha256"]==sha(s142)
          and prevg["original_S142_43M_sha256"]==sha(s142)
          and prevk["original_S93_GT_sha256"]==sha(s93)
          and prevg["original_S93_GT_sha256"]==sha(s93)
          and prevk["registered_70_new_multisentence_TRAIN_source_sha"]==
              source["TRAIN_sha256"]
          and prevk["previous_physical_Q2I8g_parent_report_SHA256"]==
              sha(q2i8g_dir/Q2I8G_PREV_REPORT)
          and prevg["readiness_native_49152_pinned_report_SHA256"]==
              sha(q2i8f_report),
          "exact true completed physical A/B real 43M source hashes")
    usedg=set(readiness["selected_balanced_21_TRAIN_IDS"])
    usedk=set(prevk["never_previously_trained_NEXT21_ids"])
    check(len(usedg)==len(usedk)==21 and usedg.isdisjoint(usedk)
          and usedk==set(prevk["never_previously_trained_NEXT21_ids"])
          and set(prevk["previously_train_used_21_ids_excluded"])==usedg,
          "prior 42 original author TRAIN must be excluded from new 28")
    registered=previously_registered_rows()
    chosen=[x for x in registered if x["id"] not in usedg|usedk]
    check(len(chosen)==NEW_TRAIN
          and Counter(x["kind"] for x in chosen)=={k:4 for k in KINDS},
          "only LAST 28 never-trained author E70 TRAIN four per each domain")
    train_new=[native_new_train(tok,x) for x in chosen]
    train_old=[verified_native_answer(tok,x) for x in old_rehearsal]
    train=train_new+train_old
    newdev=[pack_r_author_dev(tok,x) for x in r_dev_rows()]
    olddev=[verified_native_answer(tok,x) for x in old_dev]
    for x in olddev:
        x["tokenizer"]=tok
        x["eos_native_id"]=int(tok.eos_token_id)
    check(len(train)==TRAIN_ROWS==EXPECTED_TRAIN and
          len(newdev)==NEW_HELDOUT_DEV and len(olddev)==OLD_SPENT_DEV and
          all(x["split"]=="TRAIN" for x in train) and
          all(x["split"]=="DEV" for x in newdev+olddev) and
          {x["prompt_sha256"] for x in train}.isdisjoint(
              {x["prompt_sha256"] for x in newdev+olddev}) and
          max(len(x["answer_native_with_eos"]) for x in train)<=40 and
          max(len(x["answer_native_with_eos"]) for x in newdev+olddev)<=40 and
          max(len(x["prefix"])+39 for x in newdev+olddev)<160,
          "new fresh 14 DEV NEVER enters Adam or model selection")
    d,_old=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,evidence["new_research_checkpoint_contextual"])
    f=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,q2f["projector_contextual_child_checkpoint"],d)
    i1=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],f)
    i3=check_trained_joint(
        q2i3_dir/RECURRENT_CKPT,
        q2i3["research_only_updated_FULL_neural_child_FP16"][
            "GT_RECURRENT"],i1)
    i7=verify_physical_1x_parent(
        q2i7_dir/"S119Q2I7_UNIFORM1X_GT12_NRA3_NPA1_NATIVEHEAD_FP16_G0.safetensors",
        q2i7["new_1x_physical_FP16_checkpoint_and_GT12_sidecar"][
            "trained_1x_neural_fp16"],i3)
    i8=physical_q2i8a_fitted_child(q2i8a_dir/Q2I8A_REC,
        q2i8a["reversible_research_only_FP16_neural_A_B_and_GT12_gates"][
            "GT_RECURRENT_neural"],i7)
    prevd=json.loads((q2i8d_dir/PREV_REPORT).read_text(encoding="utf8"))
    check(prevd["stage"]==PREV_STAGE and
          prevd["original_S142_43M_sha256"]==sha(s142) and
          prevd["original_S93_GT_sha256"]==sha(s93) and
          prevd["previous_fitted_parent_q2i8a_sha256"]==sha(p8) and
          not prevd["candidate_accepted"] and prevd["output_gate"]==0,
          "exact 43M training parent A/B provenance")
    parentd,gated=physical_q2i8d_CE_warmstart(
        i8,q2i8d_dir,prevd)
    gmeta=prevg["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    parentg=full_saved_child(
        q2i8g_dir/Q2I8G_PREV_CE_FP16,gmeta["CE_ONLY_neural"],parentd)
    gateg=full_saved_gt(
        q2i8g_dir/Q2I8G_PREV_CE_GATES,gmeta["CE_ONLY_GT12"])
    kmeta=prevk["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    original_fitted=full_saved_child(
        q2i8k_dir/K_KL_FP16,kmeta["SOURCE135M_KL_neural"],parentg)
    gt=full_saved_gt(
        q2i8k_dir/K_KL_GATES,kmeta["SOURCE135M_KL_GT12"])
    check(gated.shape==gateg.shape==gt.shape==(12,)
          and original_fitted.output_gate==0
          and original_fitted.candidate_accepted is False,
          "exact previous KL-fitted 43M Core GT12 NRA3 NPA1 native49k FP16 child")
    ce=copy.deepcopy(original_fitted)
    margin=copy.deepcopy(original_fitted)
    check(all(torch.equal(a,b) and torch.equal(a,c)
              for a,b,c in zip(original_fitted.state_dict().values(),
                               ce.state_dict().values(),
                               margin.state_dict().values())),
          "matched identical 421793 FP16 original fitted A/B start")
    model,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",ce)
    model.eval()
    orig_digest=old_original_param_digest(model)
    old_gt={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    initial_parent={k:v.detach().cpu().clone()
                    for k,v in original_fitted.state_dict().items()}
    native_full_gt_gate_contract(full,bridge)
    check(all(not p.requires_grad for name,p in model.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "original43M and 12 physical GT layer donor weights NEVER train")
    training={};results={};saved={}
    out.mkdir(parents=True)
    try:
        for label,child in (
            ("BASELINE_CE_ONLY",ce),
            ("TRAIN_ONLY_SUPERVISED_EOS_MARGIN",margin),
        ):
            set_gt_route(full,"GT_RECURRENT",gt)
            objective=(
                "FULL_NATIVE_CE" if label=="BASELINE_CE_ONLY"
                else "FULL_NATIVE_CE_PLUS_EOS_MARGIN")
            training[label]=head_only_train(
                model,sources,bridge,full,expansion,child,train,tok,
                objective=objective)
            check(old_original_param_digest(model)==orig_digest
                  and training[label]["actual_matched_native_rank4_head_only_Adam_updates"]==TRAIN_ROWS
                  and training[label]["trainable_parameters_original_native_head_ONLY"]==TRAINABLE_HEAD,
                  "only physical native rank4 head gradient, no other 43M regions")
        check(training["BASELINE_CE_ONLY"][
                "physical_43M_full_native49152_teacher_forced_TRAIN_forwards"]==
              training["TRAIN_ONLY_SUPERVISED_EOS_MARGIN"][
                "physical_43M_full_native49152_teacher_forced_TRAIN_forwards"],
              "matched A/B CE vs supervised EOS margin identical physical 43M training budget")
        saved={
            "CE_HEAD_ONLY_FP16":persist_half(ce,out/CE_FP16),
            "EOS_MARGIN_HEAD_ONLY_FP16":persist_half(margin,out/MARGIN_FP16),
        }
        controls={
            "OLD_KL_PREVIOUS_GAIN_0p25":(original_fitted,0.25),
            "OLD_KL_PREVIOUS_GAIN_1p0":(original_fitted,1.0),
            "CE_ONLY_FITTED_GAIN_1p0":(ce,1.0),
            "SUPERVISED_EOS_MARGIN_GAIN_1p0":(margin,1.0),
        }
        for label,(child,gain) in controls.items():
            results[label]={
                "NEW_DEV":eval_own_gain(
                    tok,model,sources,bridge,full,expansion,
                    child,gt,newdev,gain=gain,arm=label),
                "OLD_DEV":eval_own_gain(
                    tok,model,sources,bridge,full,expansion,
                    child,gt,olddev,gain=gain,arm=label),
            }
    finally:
        with torch.no_grad():
            for k,v in old_gt.items():
                full.state_dict()[k].copy_(v)
        for p in full.parameters():p.requires_grad_(False)
    check(old_original_param_digest(model)==orig_digest
          and all(torch.equal(full.state_dict()[k].detach().cpu(),v)
                  for k,v in old_gt.items())
          and all(torch.equal(original_fitted.state_dict()[k].detach().cpu(),v)
                  for k,v in initial_parent.items())
          and all(torch.equal(ce.state_dict()[k],original_fitted.state_dict()[k])
                  for k in initial_parent if not k.startswith("language_head."))
          and all(torch.equal(margin.state_dict()[k],original_fitted.state_dict()[k])
                  for k in initial_parent if not k.startswith("language_head.")),
          "fitted source original43M/GT12+all NRA3 NPA1 prior FP16 immutable")
    teacher_forwards=sum(v[
        "physical_43M_full_native49152_teacher_forced_TRAIN_forwards"]
        for v in training.values())
    own_forwards=sum(s["scores"]["real_original43M_OWN_forwards"]
                     for x in results.values() for s in x.values())
    report={
        "stage":STAGE,
        "original_S142_43M_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "previous_physical_S119q2i8g_SHA256":sha(q2i8g_dir/Q2I8G_PREV_REPORT),
        "physical_frozen_S119q2i8k_KL_source_report_SHA256":sha(q2i8k_dir/K_REPORT),
        "existing_S119q2i8k_fit_FP16_neural_421793_sha256":
            sha(q2i8k_dir/K_KL_FP16),
        "preregistered_fresh_internal_author14_DEV_sha256":rdev["source_sha256"],
        "original70_synthetic_TRAIN_source_sha256":source["TRAIN_sha256"],
        "previous_42E_TRAIN_ids_excluded":sorted(usedg|usedk),
        "LAST_28_never_before_used_E_TRAIN_rows":sorted(x["id"] for x in chosen),
        "predeclared_34_equal_TRAIN_rows_and_LR":{
            "28_fresh_E70_TRAIN":28,"6_old_rehearsal_TRAIN":6,
            "native_BPE_head_training_gain":GAIN,
            "MARGIN_LAMBDA_only_arm_B":MARGIN_LAMBDA,
        },
        "no_new_regions_no_architecture_change_native_rank4_49k":True,
        "matched_train_head_only_research_arms":training,
        "saved_FULL_FP16_neural_children_old_GT12_UNCHANGED_gate0":saved,
        "fresh_AUTHOR14_DEV_plus_old6_actual_OWN_native_self_generated":results,
        "physical_true43M_full_NATIVE_train_forwards":teacher_forwards,
        "physical_true43M_full_NATIVE_OWN_forwards":own_forwards,
        "physical_original43M_all_full_native_decoder_calls":
            teacher_forwards+own_forwards,
        "no_135M_or_DEV_teacher_or_gold_label_in_grad":True,
        "all_original_S142_GT12_NRA3_NPA1_and_S115_unchanged":True,
        "original_616J7_EOS_bpp_CPU_independent_blind_NOT_EVALUATED":True,
        "no_main_LIVE_Render_Supabase_APK_or_Jessica_chat":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"RESEARCH_ONLY_TRAINED_HEAD_EOS_CALIBRATION_CAUSAL_AB_NOT_PROMOTED",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                            encoding="utf8")
    print("S119Q2I8R_TRUE43M_FULL_NATIVE_EOS_MARGIN_HEAD_ONLY_AB "+json.dumps({
        "true_native43M_train_forwards":teacher_forwards,
        "true_native43M_own_DEV_forwards":own_forwards,
        "34_exact_head_Adam_updates_each":True,
        "matched_head_trainable_params":TRAINABLE_HEAD,
        "fresh14_new_DEV":{k:v["NEW_DEV"]["scores"] for k,v in results.items()},
        "old6_DEV":{k:v["OLD_DEV"]["scores"] for k,v in results.items()},
        "candidate_accepted":False,
    },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8f_report","q2i8g_dir","q2i8k_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,
        a.q2i8f_report,a.q2i8g_dir,a.q2i8k_dir,a.out)

if __name__=="__main__":
    main()
