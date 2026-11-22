"""S119q2i8j: actual pinned 43M/135M 2nd+3rd Russian BPE gold-rank diagnosis.

Only 14 Q2I8g ALREADY SPENT internally visible author DEV; NEVER train on
these labels or claim independent blind evaluation. For each source prompt,
at the already-known wrong 2nd and 3rd native 49,152 BPE positions, compare
full native next-token distributions from physically SHA-restored Q2I8d
CE-only parent, Q2I8g CE-only, Q2I8g 135M continuation-KL, and original
frozen SmolLM2-135M source. Gold-previous-BPE prefixes are DIAGNOSTIC
teacher-forcing ONLY, never mistaken for own-token generation.
Strict no backward, no optimizer, no production, no promotion.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
from collections import Counter
from pathlib import Path
import torch

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import old_original_param_digest
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX,LOCAL_CKPT as Q2D_LOCAL,REPORT as Q2D_REPORT,OLD_PREFIX,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import load_q2d_proof
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import load_q2d_trained_child
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX,REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import load_physical_q2f_child
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    CONTEXT_CKPT as Q2I1_CTX,REPORT as Q2I1_REPORT,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,set_gt_route,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    REPORT as Q2I3_REPORT,RECURRENT_CKPT,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import check_trained_joint
from research.calibri_nra3.s119q2i7_real43m_utf8_weight_causal_ablation import REPORT as Q2I7_REPORT
from research.calibri_nra3.s119q2i8a_real43m_broader_russian_joint import (
    REPORT as Q2I8A_REPORT,RECURRENT_FP16 as Q2I8A_REC,
    RECURRENT_GT as Q2I8A_GT,verify_physical_1x_parent,
)
from research.calibri_nra3.s119q2i8b_source135m_vs_original43m_native import (
    physical_q2i8a_fitted_child,physical_trained_step,
    source_teacher_step,
)
from research.calibri_nra3.s119q2i8g_real43m_multiword_native_joint_ab import (
    STAGE as TRAIN_STAGE,REPORT as TRAIN_REPORT,
    CE_FP16,KL_FP16,CE_GATES,KL_GATES,
    PREV_REPORT,pack_new_author_dev,physical_q2i8d_CE_warmstart,
)
from research.calibri_nra3.s119q2i8g_new_ru_multiword_dev_source import (
    STAGE as DEV_STAGE,rows as spent_dev,verify as dev_verify,
)
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    verify as train_verify,
)
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA

STAGE="CALIBRI_S119Q2I8J_REAL43M_VS_SOURCE135M_GOLD_SECOND_THIRD_NATIVE_RANK_DIAGNOSTIC"
REPORT="S119Q2I8J_ACTUAL_GOLD_SECOND_THIRD_NATIVE_49152_RANK_G0.json"
POSITIONS=(1,2) # zero-based gold answer native BPE: second and third
NATIVE_VOCAB=49152
LABELS=("PRIOR_Q2I8D_CE_PARENT","Q2I8G_CE_ONLY","Q2I8G_SOURCE135M_KL",
        "FROZEN_ORIGINAL135M_SOURCE")
CE_LABEL=LABELS[1]
KL_LABEL=LABELS[2]
PARENT_LABEL=LABELS[0]
TEACHER_LABEL=LABELS[3]
GATES_KEY="research.s119q2i2.fullGT12_gates_fp16"
SEED=119152


def real_native_rank(logits,gold):
    """True 49152-way rank and two discriminative *logit* margins."""
    x=logits.detach().float().cpu().reshape(-1)
    check(x.shape==(NATIVE_VOCAB,)
          and bool(torch.isfinite(x).all())
          and type(gold) is int and 0<=gold<NATIVE_VOCAB,
          "full original 49152 native finite logits, no vocabulary masking")
    top_ids=torch.topk(x,5,largest=True,sorted=True).indices
    pred=int(top_ids[0].item())
    rank=1+int((x>x[gold]).sum().item())
    logp=float((x[gold]-torch.logsumexp(x,dim=0)).item())
    return {
        "correct_gold_native_rank_full49152":rank,
        "correct_gold_rank_top5":rank<=5,
        "gold_native_top1":pred==gold,
        "gold_log_probability_native_49152":round(logp,7),
        "wrong_top1_minus_gold_logit_gap":round(float(x[pred]-x[gold]),7),
        "greedy_original_native_id":pred,
        "gold_native_id":gold,
        "top5_native_ids":[int(i) for i in top_ids.tolist()],
    }


def full_saved_child(path,metadata,parent):
    check(path.exists() and sha(path)==metadata["sha256"]
          and path.stat().st_size==metadata["file_bytes"],
          "SHA-pinned original physical fitted FP16 child")
    data=load_file(str(path),device="cpu")
    check(data and sum(v.numel() for v in data.values())==
          metadata["saved_total_child_FP16_params"]==421793
          and all(k.startswith(OLD_PREFIX) and v.dtype==torch.float16
                  for k,v in data.items()),
          "all full 421793 child parameters, original 49k head included")
    actual={k[len(OLD_PREFIX):]:v.float()
            for k,v in data.items()}
    child=copy.deepcopy(parent)
    check(set(actual)==set(child.state_dict()),"exact matching physical architecture")
    child.load_state_dict(actual,strict=True)
    child.eval()
    check(child.output_gate==0 and child.candidate_accepted is False
          and child.activation_allowed is False,"research-only gate0")
    return child


def full_saved_gt(path,metadata):
    check(path.exists() and sha(path)==metadata["sha256"]
          and path.stat().st_size==metadata["file_bytes"]==128
          and metadata["12_GT12_trainable_only"]
          and metadata["condition"]=="GT_RECURRENT",
          "original physical GT12 gates metadata and SHA")
    data=load_file(str(path),device="cpu")
    check(set(data)=={GATES_KEY}
          and data[GATES_KEY].shape==(12,)
          and data[GATES_KEY].dtype==torch.float16,
          "original 12 layer physically trained scalar gates sidecar")
    return data[GATES_KEY].float()


def score_by_position(records):
    """Aggregates only spent DEV; no model-selection threshold or p-hacking."""
    out={}
    for position in POSITIONS:
        rows=[x for x in records if x["native_gold_answer_index_0based"]==position]
        check(len(rows)==14,"exactly 14 spent Q2I8g author DEV per position")
        sub={}
        for label in LABELS:
            group=[r["four_source_original_native_results"][label] for r in rows]
            vals=[q["correct_gold_native_rank_full49152"] for q in group]
            gaps=[q["wrong_top1_minus_gold_logit_gap"] for q in group]
            sub[label]={
                "source_count":len(group),
                "gold_native_top1":sum(q["gold_native_top1"] for q in group),
                "gold_native_top5":sum(q["correct_gold_rank_top5"] for q in group),
                "mean_gold_rank":round(statistics.mean(vals),5),
                "median_gold_rank":statistics.median(vals),
                "mean_greedy_vs_gold_logit_gap":round(statistics.mean(gaps),6),
                "mean_true_gold_native_cross_entropy_nats":round(
                    -statistics.mean(q["gold_log_probability_native_49152"]
                                     for q in group),6),
            }
        ce=[r["four_source_original_native_results"][CE_LABEL][
            "correct_gold_native_rank_full49152"] for r in rows]
        kl=[r["four_source_original_native_results"][KL_LABEL][
            "correct_gold_native_rank_full49152"] for r in rows]
        sub["PAIRED_KL_rank_vs_CE"]={
            "KL_better_lower_gold_rank":sum(b<a for a,b in zip(ce,kl)),
            "KL_worse_higher_gold_rank":sum(b>a for a,b in zip(ce,kl)),
            "same_gold_rank":sum(a==b for a,b in zip(ce,kl)),
        }
        out["gold_position_"+str(position+1)]=sub
    return out


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,
        q2i7_dir,q2i8a_dir,q2i8d_dir,q2i8g_dir,out):
    check(not out.exists(),"research report never overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    old=dev_verify()
    train=train_verify()
    check(old["stage"]==DEV_STAGE
          and old["original_new_author_heldout_DEV_count"]==14
          and train["new_TRAIN_only"]==70,
          "14 previously scored author DEV only; no TRAIN/EVAL access")
    original_report=json.loads((q2i8g_dir/TRAIN_REPORT).read_text(encoding="utf8"))
    check(original_report["stage"]==TRAIN_STAGE
          and original_report["candidate_accepted"] is False
          and original_report["output_gate"]==0
          and original_report["registered_new_14_original_author_DEV_SHA"]==
              old["source_new_DEV_sha256"]
          and original_report["registered_70_new_multisentence_TRAIN_source_sha"]==
              train["TRAIN_sha256"]
          and original_report["original_S93_GT_sha256"]==sha(s93)
          and original_report["original_S142_43M_sha256"]==sha(s142)
          and original_report["all_trained_rows_are_SOURCE_SELECTED_21_NEW_TRAIN_plus_6_old_TRAIN"],
          "reused completed 21-new matched real 43M native A/B only")
    evidence=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
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
          and q2i8a["output_gate"]==0 and not q2i8a["candidate_accepted"],
          "original 43M provenance")
    d,_=load_q2d_trained_child(
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
    i8=physical_q2i8a_fitted_child(
        q2i8a_dir/Q2I8A_REC,
        q2i8a["reversible_research_only_FP16_neural_A_B_and_GT12_gates"][
            "GT_RECURRENT_neural"],i7)
    prev=json.loads((q2i8d_dir/PREV_REPORT).read_text(encoding="utf8"))
    check(original_report["previous_physical_Q2I8d_parent_report_SHA256"]==
          sha(q2i8d_dir/PREV_REPORT)
          and prev["original_S142_43M_sha256"]==sha(s142),
          "exact completed parent SHA")
    parent,parent_gates=physical_q2i8d_CE_warmstart(i8,q2i8d_dir,prev)
    meta=original_report["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    check(set(meta)=={"CE_ONLY_neural","SOURCE135M_KL_neural",
                      "CE_ONLY_GT12","SOURCE135M_KL_GT12"},
          "exact two matched fitted physical children and GT gates")
    ce=full_saved_child(q2i8g_dir/CE_FP16,meta["CE_ONLY_neural"],parent)
    kl=full_saved_child(q2i8g_dir/KL_FP16,meta["SOURCE135M_KL_neural"],parent)
    gt_ce=full_saved_gt(q2i8g_dir/CE_GATES,meta["CE_ONLY_GT12"])
    gt_kl=full_saved_gt(q2i8g_dir/KL_GATES,meta["SOURCE135M_KL_GT12"])
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==NATIVE_VOCAB and tok.eos_token_id==2,
          "exact same pinned 49152 original BPE and native EOS")
    dev=[pack_new_author_dev(tok,row) for row in spent_dev()]
    check(len(dev)==14 and
          len({x["prompt_sha256"] for x in dev})==14
          and all(x["split"]=="DEV" and
                  len(x["answer_native_with_eos"])>=4
                  for x in dev),
          "all 14 already spent author DEV two- and three-native targets")
    check(Counter(x["kind"] for x in dev)==
          {k:2 for k in {x["kind"] for x in dev}},
          "two original already spent internal DEV per category")
    model,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",ce)
    model.eval()
    frozen_orig=old_original_param_digest(model)
    frozen_full={k:v.detach().clone() for k,v in full.state_dict().items()}
    native_full_gt_gate_contract(full,bridge)
    all_children={PARENT_LABEL:parent,CE_LABEL:ce,KL_LABEL:kl}
    all_gates={PARENT_LABEL:parent_gates,CE_LABEL:gt_ce,KL_LABEL:gt_kl}
    unchanged={
        name:{k:v.detach().clone() for k,v in child.state_dict().items()}
        for name,child in all_children.items()
    }
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32
    ).eval().requires_grad_(False)
    count=sum(x.numel() for x in teacher.parameters())
    check(120_000_000<=count<=150_000_000
          and teacher.config.vocab_size==NATIVE_VOCAB,
          "true frozen pinned 135M language distribution donor only")
    rows=[];teacher_calls=0;physical43m=0
    try:
        for ex in dev:
            gold=ex["answer_native_with_eos"]
            prior=original_report["physical_OWN_sixdomain_new_and_old_DEV"]
            for label in ("CE_ONLY","SOURCE135M_UTF8_CONTINUATION_KL"):
                matches={x["source_id"]:x for x in prior[label]["NEW_DEV"]}
                check(ex["id"] in matches
                      and matches[ex["id"]]["first_gold_BPE_correct"]
                      and matches[ex["id"]]["first_native_correct_but_wrong_second"],
                      "spend ONLY exact already-known failed-next-native Q2I8g DEV")
            for index in POSITIONS:
                check(index<len(gold)-1
                      and len(ex["prefix"])+index<160,
                      "read-only gold BPE continuation excludes native EOS")
                prefix=ex["prefix"]+gold[:index]
                check(prefix[:len(ex["prefix"])]==ex["prefix"]
                      and prefix[len(ex["prefix"]):]==gold[:index],
                      "STRICT gold PREVIOUS tokens ONLY (forensic only, never self generation)")
                target=int(gold[index])
                entry={"source_id":ex["id"],"domain":ex["kind"],
                       "source_prompt_SHA256":ex["prompt_sha256"],
                       "native_gold_answer_index_0based":index,
                       "past_gold_BPE_count":index,
                       "teacher_forced_GOLD_PREFIX_DIAGNOSTIC_only":True,
                       "is_sealed_or_independent_blind_eval":False,
                       "four_source_original_native_results":{}}
                for label,child in all_children.items():
                    set_gt_route(full,"GT_RECURRENT",all_gates[label])
                    logits=physical_trained_step(
                        model,sources,bridge,full,expansion,
                        child,prefix,ex["prompt_features"])
                    entry["four_source_original_native_results"][label]=real_native_rank(
                        logits,target)
                    physical43m+=1
                teacher_logits=source_teacher_step(teacher,prefix)
                entry["four_source_original_native_results"][TEACHER_LABEL]=real_native_rank(
                    teacher_logits,target)
                teacher_calls+=1
                rows.append(entry)
    finally:
        with torch.no_grad():
            for k,v in frozen_full.items():
                full.state_dict()[k].copy_(v)
        for p in full.parameters():
            p.requires_grad_(False)
    check(old_original_param_digest(model)==frozen_orig
          and all(torch.equal(full.state_dict()[k],v)
                  for k,v in frozen_full.items())
          and all(
              torch.equal(child.state_dict()[kk],value)
              for label,child in all_children.items()
              for kk,value in unchanged[label].items()),
          "original 43M + GT12 and all physical children FROZEN no optimizer")
    check(physical43m==14*len(POSITIONS)*len(all_children)==84
          and teacher_calls==14*len(POSITIONS)==28,
          "exact 84 real original43M and 28 original source135M native forwards")
    report={
        "stage":STAGE,
        "reused_q2i8g_original_completed_training_JSON_sha256":
            sha(q2i8g_dir/TRAIN_REPORT),
        "source_14_spent_internal_DEV_sha256":old["source_new_DEV_sha256"],
        "original_S142_43M_sha256":sha(s142),
        "original_S93_full_GT_sha256":sha(s93),
        "original_135M_model_revision_PINNED":MODEL_SHA,
        "original_native_vocab":NATIVE_VOCAB,
        "number_of_previously_spent_DEV_questions":14,
        "gold_prefix_teacher_forced_diagnostic_NOT_OWN_generation":True,
        "no_new_model_learning_or_backward_or_optimizer":True,
        "original_43M_full_NATIVE_49152_forwards":physical43m,
        "source_135M_full_NATIVE_49152_forwards":teacher_calls,
        "four_real_physical_neural_state_native_gold_rank_comparison":
            score_by_position(rows),
        "already_spent_internal_DEV_each_target_BPE_diagnostics":rows,
        "new_language_capability_proven":False,
        "actual_OWN_14_DEV_still_all_0_complete_correct":True,
        "original_43M_and_saved_FP16_research_weights_unchanged":True,
        "full_protected_616_J7_and_0p7bpp_and_CPU90pct_not_assessed":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"GOLD_PREFIX_RANK_FORENSICS_ONLY_NO_PROMOTION",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I8J_GOLD_BPE_RANK_ORIGINAL43135M "+json.dumps({
        "second_native":report[
          "four_real_physical_neural_state_native_gold_rank_comparison"][
          "gold_position_2"],
        "third_native":report[
          "four_real_physical_neural_state_native_gold_rank_comparison"][
          "gold_position_3"],
        "true_43M_native_forward_calls":physical43m,
        "true_135M_native_forward_calls":teacher_calls,
        "candidate_accepted":False,
    },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8g_dir","out"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=parser.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,a.q2i8g_dir,a.out)

if __name__=="__main__":
    main()
