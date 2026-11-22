"""S119q2i8u: TRUE original43M full49152 gold second/third BPE rank
comparison of SAME parent K vs S119q2i8t jointly fitted CE and
CE+TRAIN-only native EOS-relative 27 updates. 14 already spent author
synthetic T DEV. Only teacher-forced previous gold BPE on posthoc
diagnostics; no new training, model selection, live, 616 J7/EVAL128.
54? No — exactly 14×2×3 = 84 REAL physical 43M forwards.
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
)
from research.calibri_nra3.s119q2i8k_real43m_secondthird_joint_train_ab import (
    STAGE as K_STAGE,REPORT as K_REPORT,
    KL_FP16 as K_KL_FP16,KL_GATES as K_KL_GATES,
)
from research.calibri_nra3.s119q2i8t_real43m_fulljoint_native_EOS_AB import (
    STAGE as T_STAGE,REPORT as T_REPORT,
    CE_FP16 as T_CE_FP16,MARGIN_FP16 as T_MARGIN_FP16,
    CE_GATES as T_CE_GATES,MARGIN_GATES as T_MARGIN_GATES,
    pack_t_author_native,
)
from research.calibri_nra3.s119q2i8g_real43m_multiword_native_joint_ab import (
    STAGE as Q2I8G_PREV_STAGE,REPORT as Q2I8G_PREV_REPORT,
    CE_FP16 as Q2I8G_PREV_CE_FP16,CE_GATES as Q2I8G_PREV_CE_GATES,
    PREV_REPORT,physical_q2i8d_CE_warmstart,
)
from research.calibri_nra3.s119q2i8t_new_ru_train_dev_source import (
    STAGE as DEV_STAGE,rows as t_rows,verify as dev_verify,
)
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    verify as train_verify,
)
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA

STAGE="CALIBRI_S119Q2I8U_READONLY_TRUE43M_K_VS_T_FULLJOINT_CE_EOS_NATIVE_BPE2BPE3_RANK"
REPORT="S119Q2I8U_TRUE43M_T_CE_VS_EOSMARGIN_GOLD_2ND3RD_BPE_NATIVE49K_G0.json"
POSITIONS=(1,2)
NATIVE_VOCAB=49152
LABELS=("K_FROZEN_EARLY23_KL_43M","T_FULLJOINT_CE_43M",
        "T_FULLJOINT_CE_PLUS_EOS_RELATIVE_43M")
PARENT_LABEL=LABELS[0]
CE_LABEL=LABELS[1]
KL_LABEL=LABELS[2]
GATES_KEY="research.s119q2i2.fullGT12_gates_fp16"
SEED=119159

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
        check(len(rows)==14,"exactly 14 now-spent T synthetic author DEV per position")
        sub={}
        for label in LABELS:
            group=[r["three_source_original43m_native_results"][label] for r in rows]
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
        ce=[r["three_source_original43m_native_results"][CE_LABEL][
            "correct_gold_native_rank_full49152"] for r in rows]
        kl=[r["three_source_original43m_native_results"][KL_LABEL][
            "correct_gold_native_rank_full49152"] for r in rows]
        sub["PAIRED_EOS_MARGIN_rank_vs_CE"]={
            "EOS_MARGIN_better_lower_gold_rank":sum(b<a for a,b in zip(ce,kl)),
            "EOS_MARGIN_worse_higher_gold_rank":sum(b>a for a,b in zip(ce,kl)),
            "same_gold_rank":sum(a==b for a,b in zip(ce,kl)),
        }
        out["gold_position_"+str(position+1)]=sub
    return out


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,
        q2i7_dir,q2i8a_dir,q2i8d_dir,q2i8g_dir,q2i8k_dir,q2i8t_dir,out):
    check(not out.exists(),"research report never overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    old=dev_verify()
    train=train_verify()
    check(old["stage"]==DEV_STAGE and old["new_dev_rows"]==14
          and train["new_TRAIN_only"]==70,
          "14 now-SPENT synthetic T author DEV only, no TRAIN or protected evaluation")
    original_report=json.loads((q2i8k_dir/K_REPORT).read_text(encoding="utf8"))
    t_report=json.loads((q2i8t_dir/T_REPORT).read_text(encoding="utf8"))
    check(original_report["stage"]==K_STAGE
          and t_report["stage"]==T_STAGE
          and original_report["candidate_accepted"] is False
          and t_report["candidate_accepted"] is False
          and original_report["output_gate"]==t_report["output_gate"]==0
          and original_report["registered_70_new_multisentence_TRAIN_source_sha"]==
              train["TRAIN_sha256"]
          and original_report["original_S93_GT_sha256"]==
              t_report["original_S93_GT_sha256"]==sha(s93)
          and original_report["original_S142_43M_sha256"]==
              t_report["original_S142_43M_sha256"]==sha(s142)
          and t_report["physical_frozen_S119q2i8k_KL_source_report_SHA256"]==
              sha(q2i8k_dir/K_REPORT)
          and t_report["new_T_14_DEV_SHA256"]==old["original_14_DEV_SHA256"]
          and t_report["matched_27TRAIN_SOURCE_ONLY_and_14DEV_NO_gradient"] is True,
          "only completed source SHA-specified original43M K/T native49k sidecars gate0")
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
          and original_report["previous_physical_Q2I8g_parent_report_SHA256"]==
          sha(q2i8g_dir/Q2I8G_PREV_REPORT)
          and prev["original_S142_43M_sha256"]==sha(s142),
          "exact completed parent SHA")
    parent_old,gates_old=physical_q2i8d_CE_warmstart(i8,q2i8d_dir,prev)
    g_report=json.loads((q2i8g_dir/Q2I8G_PREV_REPORT).read_text(encoding="utf8"))
    check(g_report["stage"]==Q2I8G_PREV_STAGE
          and g_report["candidate_accepted"] is False
          and g_report["output_gate"]==0
          and g_report["original_S93_GT_sha256"]==sha(s93)
          and g_report["original_S142_43M_sha256"]==sha(s142)
          and g_report["registered_70_new_multisentence_TRAIN_source_sha"]==
              train["TRAIN_sha256"],
          "exact immutable previous G CE trained parent")
    gmeta=g_report["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    parent=full_saved_child(q2i8g_dir/Q2I8G_PREV_CE_FP16,
        gmeta["CE_ONLY_neural"],parent_old)
    parent_gates=full_saved_gt(q2i8g_dir/Q2I8G_PREV_CE_GATES,
        gmeta["CE_ONLY_GT12"])
    check(gates_old.shape==parent_gates.shape==(12,),
          "original GT12 shape unchanged from old trained research state")
    meta=original_report["saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0"]
    check(set(meta)=={"CE_ONLY_neural","SOURCE135M_KL_neural",
                      "CE_ONLY_GT12","SOURCE135M_KL_GT12"},
          "exact fitted early23 original KL native49k previous K")
    original_k=full_saved_child(
        q2i8k_dir/K_KL_FP16,meta["SOURCE135M_KL_neural"],parent)
    gt_k=full_saved_gt(q2i8k_dir/K_KL_GATES,meta["SOURCE135M_KL_GT12"])
    newmeta=t_report["saved_FULL_FP16_joint_neural_and_GT12_research_only"]
    check(set(newmeta)=={"CE_joint_FP16","EOS_MARGIN_joint_FP16",
                         "CE_GT12","EOS_MARGIN_GT12"},
          "exact real original43M jointly trained existing neural and GT12 values")
    ce=full_saved_child(
        q2i8t_dir/T_CE_FP16,newmeta["CE_joint_FP16"],original_k)
    kl=full_saved_child(
        q2i8t_dir/T_MARGIN_FP16,newmeta["EOS_MARGIN_joint_FP16"],original_k)
    gt_ce=full_saved_gt(q2i8t_dir/T_CE_GATES,newmeta["CE_GT12"])
    gt_kl=full_saved_gt(q2i8t_dir/T_MARGIN_GATES,newmeta["EOS_MARGIN_GT12"])
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==NATIVE_VOCAB and tok.eos_token_id==2,
          "exact same pinned 49152 original BPE and native EOS")
    dev=[pack_t_author_native(tok,row,split="DEV") for row in t_rows("DEV")]
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
    all_children={PARENT_LABEL:original_k,CE_LABEL:ce,KL_LABEL:kl}
    all_gates={PARENT_LABEL:gt_k,CE_LABEL:gt_ce,KL_LABEL:gt_kl}
    unchanged={
        name:{k:v.detach().clone() for k,v in child.state_dict().items()}
        for name,child in all_children.items()
    }
    rows=[];physical43m=0
    try:
        for ex in dev:
            gold=ex["answer_native_with_eos"]
            prior=t_report["new_14_INTERNAL_DEV_and_old6_spent_OWN_native_BPE_by_arm"]
            for label in ("FULL_JOINT_CE_GAIN_0p25",
                          "FULL_JOINT_CE_PLUS_EOS_RELATIVE_GAIN_0p25"):
                matches={x["source_id"]:x for x in prior[label]["NEW_DEV"]["OWN_samples"]}
                check(ex["id"] in matches
                      and matches[ex["id"]]["first_gold_BPE_correct"]
                      and not matches[ex["id"]]["true_native_full_EOS_correct"],
                      "exact prior T 14 scored own DEV prompts all first BPE correct and full wrong")
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
                       "three_source_original43m_native_results":{}}
                for label,child in all_children.items():
                    set_gt_route(full,"GT_RECURRENT",all_gates[label])
                    logits=physical_trained_step(
                        model,sources,bridge,full,expansion,
                        child,prefix,ex["prompt_features"])
                    entry["three_source_original43m_native_results"][label]=real_native_rank(
                        logits,target)
                    physical43m+=1
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
    check(physical43m==14*len(POSITIONS)*len(all_children)==84,
          "exact 84 REAL original43M 12GT 49152 full native BPE rank forwards")
    diagnostic=score_by_position(rows)
    report={
        "stage":STAGE,
        "previous_original_K_43M_KL_REPORT_SHA256":sha(q2i8k_dir/K_REPORT),
        "completed_T_fulljoint_actual_43M_REPORT_SHA256":sha(q2i8t_dir/T_REPORT),
        "T_14_SYNTHETIC_ALREADY_SPENT_DEV_source_sha256":
            old["original_14_DEV_SHA256"],
        "original_S142_43M_sha256":sha(s142),
        "original_S93_full_GT_sha256":sha(s93),
        "true_original_49152_BPE_vocab":NATIVE_VOCAB,
        "num_spent_author_14_DEV_prompts":14,
        "true_source135M_teacher_calls":0,
        "teacher_forced_gold_PREVIOUS_tokens_ONLY_diagnostic_NOT_own":True,
        "no_training_or_backward_or_optimizer_or_promotion":True,
        "true_original43M_full_GT12_native49152_forwards":physical43m,
        "gold_position_2_3_spent_DEV_full_native49k_diagnostic":diagnostic,
        "per_sample_gold_prefix_only_SECOND_THIRD_native_BPE_posthoc":rows,
        "original_original43M_GT_and_K_T_saved_FP16_weights_unchanged":True,
        "no_protected_616_J7_EVAL128_GOLD_bpp_CPU_checked":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"RANK_DIAGNOSTICS_NOT_LANGUAGE_CORRECTION_OR_PROMOTION",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I8U_TRUE43M_GOLD_BPE2_BPE3_NATIVE_RANK "+json.dumps({
        "gold_second_49152":diagnostic["gold_position_2"],
        "gold_third_49152":diagnostic["gold_position_3"],
        "true_43M_native_forwards":physical43m,
        "candidate_accepted":False,
    },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8g_dir","q2i8k_dir","q2i8t_dir","out"):
        parser.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=parser.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,a.q2i8g_dir,a.q2i8k_dir,a.q2i8t_dir,a.out)

if __name__=="__main__":
    main()
