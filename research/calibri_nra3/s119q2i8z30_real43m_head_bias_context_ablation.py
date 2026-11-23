"""S119q2i8z30: physical forensic of Z29 rank4 native head bias collapse.

Previously successfully FITTED Z29 head-only training 2×84×5 original
S142 decoder passes yielded TRAIN CE progress, but NEW copied head
collapsed to just 2 distinct second-token IDs and 14/14 produced
a triple native BPE repeat within first6, versus old trained Z10
9 distinct second IDs and 10/14 triple repeat.

Causal ablation on EXACT SAME ALREADY-SPENT Z28 author DEV14:
  OLD Z10 full native head (checkpoint frozen)
  Z29 NEW entire 247040-param native head (checkpoint frozen)
  Z29 NEW low-rank down/up + OLD Z10 global native output bias
  Z29 NEW global native output bias + OLD Z10 low-rank down/up.
No learning/tuning/new DEV or weight updates. Every mode physically
runs true original S142 43M/GT12/Core/NRA3/NPA1 native49152 greedy
autonomous first6; replay old/new modes EXACT source Z29 artifact
native BPE IDs fail-closed. Human labels POSTHOC only. This tests
whether learned global native output_bias (49,152 dimensions) caused
mode collapse; no claim of new performance on independent holdout.
Never edit original models, prior fitted Z10 or fitted Z29;
do not promote GATE0; no J7/EVAL128/GOLD, packing or CPU gate.
"""
from __future__ import annotations
import argparse
import copy
import json
from collections import Counter
import time
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


from research.calibri_nra3.s119q2i8z28_precommit_84train14fresh_native_head_multistep import (
    STAGE as NEW_SOURCE_STAGE,ORIGIN as NEW_SOURCE_ORIGIN,
    rows as fresh_new_rows,verify as fresh_new_verify,
    TRAIN_EPOCHS,HEAD_GAIN,HEAD_LR,HEAD_WEIGHT_DECAY,EVAL_NATIVE_FIRST_K,
)
from research.calibri_nra3.s119q2i8z28_fullnative_trainonly_multitoken_head_loss import (
    select_train_positions,full_native_trainonly_pos_loss,
    verify_shadow_head_params,HEAD_TOTAL_PARAMS,
)
from research.calibri_nra3.s119q2i8z16_trainonly_lowrank_native64_reranker import (
    capture_existing_full43m_hidden,NativeBPE2CandidateReranker,
    HIDDEN,
)

from safetensors.torch import load_file
from research.calibri_nra3.s119q2i8z29_real43m_multitoken_headonly_train_eval import (
    STAGE as PRIOR_Z29_STAGE, REPORT as PRIOR_Z29_REPORT,
)
from research.calibri_nra3.s119q2i8z10_true43m_27TRAIN_contexttop8_AB import (
    STAGE as Z10_PREV_STAGE,REPORT as Z10_PREV_REPORT,
    CE_FP16 as Z10_CE_FP16,CONTEXT_FP16 as Z10_CONTEXT_FP16,
    CE_GATES as Z10_CE_GATES,CONTEXT_GATES as Z10_CONTEXT_GATES,
)
from research.calibri_nra3.s119q2i8z9_new_ru_train_dev_source import (
    STAGE as Z9_SOURCE_STAGE,ORIGIN as Z9_ORIGIN,
    rows as z9_rows,verify as z9_verify,
)
from research.calibri_nra3.s119q2i8z10_true43m_joint_train_contexttop8 import (
    train_joint_true43m_context_top8_ab,EARLY23_MULTIPLIER,
    CONTEXT_TOP8_MARGIN_WEIGHT,HEAD_GAIN,STAGE as JOINT_STAGE,
)
from research.calibri_nra3.s119q2i8w_true43m_joint_native_early23_AB import (
    STAGE as W_STAGE,REPORT as W_REPORT,
    EARLY_FP16 as W_EARLY_FP16,EARLY_GATES as W_EARLY_GATES,
)
from research.calibri_nra3.s119q2i8t_real43m_fulljoint_native_EOS_AB import (
    STAGE as T_STAGE,REPORT as T_REPORT,
    CE_FP16 as T_CE_FP16,CE_GATES as T_CE_GATES,
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

STAGE="CALIBRI_S119Q2I8Z30_PHYSICAL_S142_RANK4_NATIVE_HEAD_NEW_BIAS_VS_CONTEXT_FROZEN_SPENT14"
REPORT="S119Q2I8Z30_S142_RANK4_HEAD_BIAS_CONTEXT_CAUSAL_SPENT14_FIRST6_G0.json"
CE_FP16="S119Q2I8Z10_TRUE43M_FULLJOINT_NATIVE_CE_EARLY23_X3_FP16_G0.safetensors"
CONTEXT_FP16="S119Q2I8Z10_TRUE43M_FULLJOINT_NATIVE_CONTEXT_TOP8_BPE2_FP16_G0.safetensors"
CE_GATES="S119Q2I8Z10_TRUE43M_FULLJOINT_NATIVE_CE_EARLY23_X3_GT12_FP16_G0.safetensors"
CONTEXT_GATES="S119Q2I8Z10_TRUE43M_FULLJOINT_NATIVE_CONTEXT_TOP8_BPE2_GT12_FP16_G0.safetensors"
NEW_HELDOUT_DEV=14
OLD_SPENT_DEV=6
TRAIN_ROWS=27
NEW_TRAIN=21
NATIVE_VOCAB=49152
SEED=119163
ARMS=("ALL_NATIVE_CE_EARLY23_X3","ALL_NATIVE_CE_EARLY23_X3_CONTEXT_TOP8")


def pack_z9_author_native(tok,row,*,split):
    check(row["split"]==split
          and split in ("TRAIN","DEV")
          and row["origin"]==Z9_ORIGIN
          and row["id"].startswith("S119Q2I8Z9_")
          and row["kind"] in KINDS,
          "only fresh preregistered Z9 author 21 TRAIN and never-before-scored 14 DEV native49k")
    clean=dict(row)
    clean["origin"]="original_newly_authored_examples_no_third_party_corpus"
    item=verified_native_answer(tok,clean)
    check(item["split"]==split
          and tok.decode(item["answer_native_without_eos"],
                         skip_special_tokens=True)==row["completion"],
          "true original49152 human TRAIN or DEV native BPE target and EOS exact")
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
        two=bool(len(ids)>=2 and ids[:2]==item["answer_native_with_eos"][:2])
        three=bool(len(ids)>=3 and ids[:3]==item["answer_native_with_eos"][:3])
        output.append({
            "first_two_gold_BPE_correct":two,
            "first_three_gold_BPE_correct":three,
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
            "correct_first_TWOnative_BPE":sum(x["first_two_gold_BPE_correct"] for x in output),
            "correct_first_THREEnative_BPE":sum(x["first_three_gold_BPE_correct"] for x in output),
            "first_correct_but_second_wrong":sum(x["first_native_correct_but_wrong_second"] for x in output),
            "real_original43M_OWN_forwards":sum(x["original_full_decoder_calls"] for x in output),
        }
    }


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,q2i7_dir,q2i8a_dir,
        q2i8d_dir,q2i8f_report,q2i8g_dir,q2i8k_dir,q2i8t_dir,q2i8w_dir,z10_dir,z29_dir,out):
    check(not out.exists(),"do not overwrite old research model or published result")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=source_verify()
    z9_source=z9_verify()
    fresh=fresh_new_verify()
    check(source["stage"]==SOURCE_STAGE
          and source["new_TRAIN_only"]==70
          and z9_source["stage"]==Z9_SOURCE_STAGE
          and fresh["stage"]==NEW_SOURCE_STAGE
          and fresh["prior_legal_TRAIN_count"]==84
          and fresh["NEW_author_synthetic_DEV_count"]==14
          and fresh["fixed_training"]["train_epochs"]==2
          and fresh["fixed_training"]["head_only_lr"]==.006
          and fresh["fixed_training"]["free_autoregressive_native_dev_first_K"]==6
          and HEAD_GAIN==.25,
          "preregistered Z12 original author 14 synthetic DEV only")
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
    fresh_dev=[]
    for authored in fresh_new_rows("DEV"):
        check(authored["split"]=="DEV" and
              authored["origin"]==NEW_SOURCE_ORIGIN and
              authored["id"].startswith("S119Q2I8Z28_"),
              "read only spent Z28 original-author14 DEV, not TRAIN")
        item=dict(authored)
        item["origin"]="original_newly_authored_examples_no_third_party_corpus"
        prepared=verified_native_answer(tok,item)
        check(prepared["split"]=="DEV" and
              tok.decode(prepared["answer_native_without_eos"],
                         skip_special_tokens=True)==authored["completion"] and
              4<=len(prepared["answer_native_with_eos"])<=64 and
              len(prepared["prefix"])+len(prepared["answer_native_with_eos"])<160,
              "same exact original native49k previously SPENT Z28 human answer metadata")
        fresh_dev.append(prepared)
    check(len(fresh_dev)==14
          and len({r["prompt_sha256"] for r in fresh_dev})==14,
          "exact same previously SPENT Z28 fourteen DEV only")
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
    k_fitted=full_saved_child(
        q2i8k_dir/K_KL_FP16,kmeta["SOURCE135M_KL_neural"],parentg)
    k_gt=full_saved_gt(
        q2i8k_dir/K_KL_GATES,kmeta["SOURCE135M_KL_GT12"])
    t_report=json.loads((q2i8t_dir/T_REPORT).read_text(encoding="utf8"))
    check(t_report["stage"]==T_STAGE
          and not t_report["candidate_accepted"] and t_report["output_gate"]==0
          and t_report["physical_frozen_S119q2i8k_KL_source_report_SHA256"]==
              sha(q2i8k_dir/K_REPORT)
          and t_report["existing_S119q2i8k_fit_FP16_neural_421793_sha256"]==
              sha(q2i8k_dir/K_KL_FP16)
          and t_report["original_S142_43M_sha256"]==sha(s142)
          and t_report["original_S93_GT_sha256"]==sha(s93)
          and t_report["matched_27TRAIN_SOURCE_ONLY_and_14DEV_NO_gradient"] is True,
          "ONLY SHA verified completed S119q2i8t original physical joint trained CE")
    tmeta=t_report["saved_FULL_FP16_joint_neural_and_GT12_research_only"]
    check(set(tmeta)=={"CE_joint_FP16","EOS_MARGIN_joint_FP16",
                       "CE_GT12","EOS_MARGIN_GT12"},
          "exact original trained T full FP16 child and GT12 sidecar keys")
    t_fitted=full_saved_child(
        q2i8t_dir/T_CE_FP16,tmeta["CE_joint_FP16"],k_fitted)
    t_gt=full_saved_gt(q2i8t_dir/T_CE_GATES,tmeta["CE_GT12"])
    w_report=json.loads((q2i8w_dir/W_REPORT).read_text(encoding="utf8"))
    check(w_report["stage"]==W_STAGE
          and not w_report["candidate_accepted"] and w_report["output_gate"]==0
          and w_report["completed_T_FULLJOINT_CE_REPORT_SHA256"]==sha(q2i8t_dir/T_REPORT)
          and w_report["completed_T_FULLJOINT_CE_neural_FP16_SHA256"]==sha(q2i8t_dir/T_CE_FP16)
          and w_report["completed_T_FULLJOINT_CE_GT12_FP16_SHA256"]==sha(q2i8t_dir/T_CE_GATES)
          and w_report["original_S142_43M_sha256"]==sha(s142)
          and w_report["original_S93_GT_sha256"]==sha(s93)
          and w_report["match_full43M_27TRAIN_A_B_from_identical_T_CE_FP16_state"] is True,
          "verified successful original W 43M source joint early23 CE parent")
    wmeta=w_report["saved_full_FP16_original_joint_neural_plus_separate_GT12"]
    check(set(wmeta)=={"CE_joint_FP16","EARLY23_X3_joint_FP16",
                       "CE_GT12","EARLY23_X3_GT12"},
          "exact original completed prior W full neural early23x3 FP16 +GT12")
    original_fitted=full_saved_child(
        q2i8w_dir/W_EARLY_FP16,wmeta["EARLY23_X3_joint_FP16"],t_fitted)
    gt=full_saved_gt(q2i8w_dir/W_EARLY_GATES,wmeta["EARLY23_X3_GT12"])
    check(gated.shape==gateg.shape==k_gt.shape==t_gt.shape==gt.shape==(12,)
          and original_fitted.output_gate==0
          and original_fitted.candidate_accepted is False,
          "W true original43M+12GT parent with output gate0")
    prev_path=z10_dir/Z10_PREV_REPORT
    prev=json.loads(prev_path.read_text(encoding="utf8"))
    check(prev["stage"]==Z10_PREV_STAGE and
          prev["output_gate"]==0 and prev["candidate_accepted"] is False
          and prev["original_S142_43M_sha256"]==sha(s142)
          and prev["original_S93_GT_sha256"]==sha(s93)
          and prev["completed_W_421793_fp16_source_sha256"]==
              sha(q2i8w_dir/W_EARLY_FP16)
          and prev["completed_W_12GT_fit_source_sha256"]==
              sha(q2i8w_dir/W_EARLY_GATES)
          and prev["new_Z9_21_TRAIN_sha256"]==z9_source["Z9_21_TRAIN_sha256"]
          and prev["new_Z9_14_DEV_sha256"]==z9_source["Z9_14_DEV_sha256"]
          and prev["true43M_49k_full_native_TRAIN_forwards"]==1346
          and prev["true43M_49k_full_native_OWN_forwards"]==1707,
          "only actual completed Z10 matched real original43M saved A/B from W")
    saved=prev["neural_all421793_plus_separate12GT_FP16_saved"]
    check(set(saved)=={"CE_EARLY23_X3_FP16","CONTEXT_TOP8_BPE2_FP16",
                       "CE_EARLY23_X3_GT12","CONTEXT_TOP8_BPE2_GT12"},
          "both original fitted neural421793 and 12 GT12 physical sidecars")
    child_a=full_saved_child(
        z10_dir/Z10_CE_FP16,saved["CE_EARLY23_X3_FP16"],original_fitted)
    child_b=full_saved_child(
        z10_dir/Z10_CONTEXT_FP16,saved["CONTEXT_TOP8_BPE2_FP16"],original_fitted)
    gt_a=full_saved_gt(z10_dir/Z10_CE_GATES,saved["CE_EARLY23_X3_GT12"])
    gt_b=full_saved_gt(z10_dir/Z10_CONTEXT_GATES,saved["CONTEXT_TOP8_BPE2_GT12"])
    check(gt_a.shape==gt_b.shape==(12,)
          and all(x.output_gate==0 and x.candidate_accepted is False
                  for x in (child_a,child_b)),
          "research gate0, completed true saved fit A/B only")
    model,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",child_a)
    model.eval()
    original_digest=old_original_param_digest(model)
    full_gt_donor={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    child_frozen={
        key:{k:v.detach().cpu().clone() for k,v in child.state_dict().items()}
        for key,child in (("A",child_a),("B",child_b))
    }
    native_full_gt_gate_contract(full,bridge)
    outcomes={}
    out.mkdir(parents=True)
    try:
        # S142's compressed model exposes a custom TrainEmbedding, not
        # a bare nn.Embedding.weight. Read the REAL original table through
        # its frozen input-embedding forward, in native-ID chunks. This
        # does NOT run any decoder layers or train/change that module.
        embed_module=model.get_input_embeddings()
        native_pieces=[]
        with torch.no_grad():
            for i in range(0,NATIVE_VOCAB,1024):
                native_ids=torch.arange(i,min(i+1024,NATIVE_VOCAB),
                                        dtype=torch.long).view(1,-1)
                chunk=embed_module(native_ids)
                check(isinstance(chunk,torch.Tensor)
                      and tuple(chunk.shape)==(1,native_ids.numel(),HIDDEN)
                      and torch.isfinite(chunk).all(),
                      "original TRAINED compressed TrainEmbedding original49152 ID lookup")
                native_pieces.append(chunk[0].detach().float().cpu().clone())
        native_embedding=torch.cat(native_pieces,dim=0)
        check(native_embedding.shape==(NATIVE_VOCAB,HIDDEN)
              and not native_embedding.requires_grad
              and torch.isfinite(native_embedding).all(),
              "read-only original compressed native49152 embedding table h320")
        # Four actual physical autoregressive model paths on the SAME
        # already-SPENT Z28 DEV14. No TRAIN rows or gradients used.
        z29_path=z29_dir/PRIOR_Z29_REPORT
        previous=json.loads(z29_path.read_text(encoding="utf8"))
        FROZEN_NEW_HEAD_SHA={
            "Z10_FULLJOINT_EARLY23_X3_CE":
                "0a15e150038f15a8df6a4cfd8a3a9b4df38b1e718fd6792d30eab4c18244ea6d",
            "Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2":
                "05bd677da2c4b8487789289020e1842b4403d07e5b5d7d85d6c95456ec59c606",
        }
        check(previous["stage"]==PRIOR_Z29_STAGE
              and previous["Z28_84_previously_legal_TRAIN_SHA256"]==
                  fresh["TRAIN_SHA256"]
              and previous["Z28_brandNEW_author14DEV_SHA256"]==
                  fresh["DEV_SHA256"]
              and previous["new_Z28_14DEV_NOW_SPENT_internal_not_independently_blind"] is True
              and previous["true_original_S142_43M_all12GT_real_native49152_FORWARDS"]==2002
              and previous["candidate_accepted"] is False
              and previous["output_gate"]==0
              and set(previous["NEW_copied_head_only_TRAIN_and_first6_autonomous_DEV_A_B"])==
                  set(FROZEN_NEW_HEAD_SHA),
              "completed true Z29 new native head source SHA, consumed DEV14, 2002 physical forwards")
        MODES=(
            "ORIGINAL_Z10_FITTED_NATIVE_HEAD",
            "NEW_Z29_ALL_PARAMS",
            "NEW_Z29_LOW_RANK_OLD_GLOBAL_BIAS",
            "NEW_Z29_GLOBAL_BIAS_OLD_LOW_RANK",
        )
        for arm,original_child,gates in (
            ("Z10_FULLJOINT_EARLY23_X3_CE",child_a,gt_a),
            ("Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2",child_b,gt_b),
        ):
            start=time.monotonic()
            set_gt_route(full,"GT_RECURRENT",gates)
            prior=previous["NEW_copied_head_only_TRAIN_and_first6_autonomous_DEV_A_B"][arm]
            original_head={k:v.detach().float().cpu().clone()
                for k,v in original_child.language_head.state_dict().items()}
            tag="CE" if arm.endswith("_CE") else "CONTEXT_TOP8"
            name="S119Q2I8Z29_SHADOW_"+tag+"_ONLY_NATIVE_RANK4_HEAD247040_FP16_G0.safetensors"
            saved=prior["saved_only_copied_native_language_head_FP16"]
            path_fp16=z29_dir/name
            check(saved["filename"]==name
                  and saved["sha256"]==FROZEN_NEW_HEAD_SHA[arm]
                  and saved["candidate_accepted"] is False
                  and saved["output_gate"]==0
                  and saved["params"]==247040
                  and path_fp16.is_file()
                  and sha(path_fp16)==FROZEN_NEW_HEAD_SHA[arm],
                  "load only precisely pinned Z29 full head fp16 source")
            new_tensors=load_file(str(path_fp16),device="cpu")
            check(set(new_tensors)==set(original_head)==
                  {"down.weight","up.weight","output_bias"}
                  and all(v.dtype==torch.float16 for v in new_tensors.values())
                  and all(v.shape==original_head[k].shape
                          for k,v in new_tensors.items()),
                  "only existing physical Z29 rank4 native head weights, no original model")
            new_head={k:v.detach().float().clone() for k,v in new_tensors.items()}
            src={x["source_id"]:x for x in prior[
                "NEW_DEV14_own_native_autoregressive_source"]}
            check(len(src)==14,"real parent Z29 14 source-level autonomous sequences")
            mode_model={}
            for mode in MODES:
                if mode==MODES[0]:
                    mode_model[mode]=original_child
                    continue
                copy_child=copy.deepcopy(original_child)
                for p in copy_child.child.parameters():
                    p.requires_grad_(True)
                for p in copy_child.language_head.parameters():
                    p.requires_grad_(True)
                if mode==MODES[1]:
                    state=new_head
                elif mode==MODES[2]:
                    state={**new_head,"output_bias":original_head["output_bias"]}
                else:
                    state={**original_head,"output_bias":new_head["output_bias"]}
                copy_child.language_head.load_state_dict(state,strict=True)
                copy_child.eval()
                check(copy_child.output_gate==0
                      and copy_child.candidate_accepted is False
                      and all(torch.equal(copy_child.language_head.state_dict()[k].float(),v)
                              for k,v in state.items())
                      and all(torch.equal(copy_child.child.state_dict()[k].detach().cpu(),
                                          original_child.child.state_dict()[k].detach().cpu())
                              for k in original_child.child.state_dict()),
                      "strict hybrid: ONLY requested frozen Z29/old Z10 rank4 head components swapped")
                mode_model[mode]=copy_child
            observed=[]
            with torch.no_grad():
                for row in fresh_dev:
                    selected={}
                    for mode in MODES:
                        active_child=mode_model[mode]
                        own=[]
                        for step in range(EVAL_NATIVE_FIRST_K):
                            if own and own[-1]==2:break
                            with scoped_exact_native_gain(
                                  active_child.language_head,HEAD_GAIN) as witnessed:
                                logits=real_joint_native(
                                    model,sources,bridge,full,expansion,
                                    active_child,row["prefix"]+own,
                                    row["prompt_features"],contextual=True)
                            check(witnessed[0]==1
                                  and logits.shape==(1,NATIVE_VOCAB)
                                  and full.last_forward_coverage==tuple(range(12)),
                                  "one actual S142 43M full native49152 forward per autonomous first6 mode step")
                            own.append(int(logits[0].argmax().item()))
                        gold=row["answer_native_with_eos"]
                        count=0
                        for chosen,want in zip(own,gold):
                            if chosen!=want:break
                            count+=1
                        selected[mode]={
                            "true_original43M_autoregressive_first6_native_ids":own,
                            "first1_correct_POSTHOC":count>=1,
                            "first2_correct_POSTHOC":count>=2,
                            "first3_correct_POSTHOC":count>=3,
                            "first4_correct_POSTHOC":count>=4,
                            "first6_correct_POSTHOC":count>=6,
                            "contiguous_correct_BPE":count,
                            "original43M_native49152_forwards":len(own),
                            "BPE2_eq_BPE3_eq_BPE4":bool(
                                len(own)>=4 and own[1]==own[2]==own[3]),
                            "ANY_triple_native_repeat_first6":any(
                                own[i]==own[i+1]==own[i+2]
                                for i in range(max(0,len(own)-2))),
                            "native_EOS_within_first6":2 in own,
                        }
                    parent=src.get(row["id"])
                    check(parent is not None and
                          parent["source_prompt_SHA256"]==row["prompt_sha256"] and
                          selected[MODES[0]]["true_original43M_autoregressive_first6_native_ids"]==
                          parent["modes"]["ORIGINAL_Z10_FITTED_NATIVE_HEAD"][
                              "autonomous_first6_native_BPE_ids"] and
                          selected[MODES[1]]["true_original43M_autoregressive_first6_native_ids"]==
                          parent["modes"]["NEW_MULTIPOSITION_TRAINED_HEAD_ONLY_COPY"][
                              "autonomous_first6_native_BPE_ids"],
                          "CRITICAL Z30 S142 original and Z29 FP16 candidate exact archived 14 original autonomy parity")
                    observed.append({
                        "source_id":row["id"],"kind":row["kind"],
                        "prompt_SHA256":row["prompt_sha256"],
                        "four_targetblind_frozen_head_component_modes":selected,
                        "parent_original_and_new_head_predictions_replayed_exactly":True,
                        "new_Z28_author_DEV14_ALREADY_SPENT_no_training":True,
                    })
            check(len(observed)==14
                  and len({x["prompt_SHA256"] for x in observed})==14,
                  "all 14 original already SPENT author DEV scored once by all four fixed hybrid modes")
            scores={}
            for mode in MODES:
                rows=[x["four_targetblind_frozen_head_component_modes"][mode]
                      for x in observed]
                scores[mode]={
                    "first1_correct":sum(x["first1_correct_POSTHOC"] for x in rows),
                    "first2_correct":sum(x["first2_correct_POSTHOC"] for x in rows),
                    "first3_correct":sum(x["first3_correct_POSTHOC"] for x in rows),
                    "first4_correct":sum(x["first4_correct_POSTHOC"] for x in rows),
                    "first6_correct":sum(x["first6_correct_POSTHOC"] for x in rows),
                    "any_native_triple_repeat_first6":sum(
                        x["ANY_triple_native_repeat_first6"] for x in rows),
                    "BPE2_BPE3_BPE4_identical":sum(
                        x["BPE2_eq_BPE3_eq_BPE4"] for x in rows),
                    "native_EOS_first6":sum(
                        x["native_EOS_within_first6"] for x in rows),
                    "distinct_BPE2_native_ids":len(set(
                        x["true_original43M_autoregressive_first6_native_ids"][1]
                        for x in rows if len(
                        x["true_original43M_autoregressive_first6_native_ids"])>1)),
                    "true_S142_fullnative49152_forwards":sum(
                        x["original43M_native49152_forwards"] for x in rows),
                }
            outcomes[arm]={
                "same_previous_SPENT14_physical_mode_component_ablation":scores,
                "same_previous_SPENT14_source_raw_native_first6":observed,
                "copied_new_Z29_HEAD247040_SHA256":FROZEN_NEW_HEAD_SHA[arm],
                "wall_uncontrolled_CPU_seconds":round(time.monotonic()-start,3),
            }

    finally:
        with torch.no_grad():
            for k,v in full_gt_donor.items():
                full.state_dict()[k].copy_(v)
        for p in full.parameters():p.requires_grad_(False)
    check(old_original_param_digest(model)==original_digest
          and all(torch.equal(full.state_dict()[k].detach().cpu(),v)
                  for k,v in full_gt_donor.items())
          and all(torch.equal(child.state_dict()[k].detach().cpu(),v)
                  for tag,child in (("A",child_a),("B",child_b))
                  for k,v in child_frozen[tag].items()),
          "original S142/12GT donor and all saved Z10 A/B fitted params preserved")
    check(len(outcomes)==2
          and all(len(v["same_previous_SPENT14_source_raw_native_first6"])==14
                  and set(v["same_previous_SPENT14_physical_mode_component_ablation"])==
                      {"ORIGINAL_Z10_FITTED_NATIVE_HEAD",
                       "NEW_Z29_ALL_PARAMS",
                       "NEW_Z29_LOW_RANK_OLD_GLOBAL_BIAS",
                       "NEW_Z29_GLOBAL_BIAS_OLD_LOW_RANK"}
                  for v in outcomes.values()),
          "all true original S142 43M A/B and 4 read-only native head ablation modes")
    true_forward_calls=sum(
        m["true_S142_fullnative49152_forwards"]
        for v in outcomes.values()
        for m in v["same_previous_SPENT14_physical_mode_component_ablation"].values())
    check(56<=true_forward_calls<=672,
          "only original S142 real43M native49152 4 modes×14×6×A/B counted")
    report={
        "stage":STAGE,
        "source_completed_Z29_true43M_physical_report_SHA256":sha(z29_path),
        "previously_fitted_Z29_A_B_head247040_FP16_SHA256":FROZEN_NEW_HEAD_SHA,
        "original_Z28_brandNEW_author_synthetic_DEV14_NOW_SPENT_SHA256":fresh["DEV_SHA256"],
        "previous_legitimate_84_TRAIN_source_SHA256":fresh["TRAIN_SHA256"],
        "previous_physically_completed_Z29_true_original_S142_43M_forwards":2002,
        "new_Z30_true_original43M_native49152_forwards":true_forward_calls,
        "actual_S142_four_HEAD_component_modes_A_B":outcomes,
        "entire_old_native_rank4_vs_new_native_rank4_replayed_exact_source_Z29":True,
        "new_native_head_global_49152_bias_vs_lowrank_context_isolation":True,
        "all_four_state_combinations_copied_from_existing_training_only":True,
        "NO_new_TRAIN_or_DEV_supervision_or_hyperparameter_selection":True,
        "NO_new_independent_DEV_quality_validation":True,
        "original_frozen_S142_12GT_Core_NRA3_NPA1_Z10_A_B_and_Z29_weights_UNCHANGED":True,
        "no_616J7_EVAL128_GOLD_packed_bpp_CB59_CPU_certification":True,
        "no_MAIN_LIVE_Render_Supabase_APK_Jessica_changes":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"READONLY_FROZEN_NATIVE_HEAD_OUTPUT_BIAS_VS_LOWRANK_CAUSAL_DIAGNOSTIC_ONLY",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
                           sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z30_S142_HEAD_BIAS_VS_LOWRANK_CAUSAL_SPENT14 "+json.dumps({
        "source_SPENT_Z28_DEV_SHA256":fresh["DEV_SHA256"],
        "new_Z30_true_original_S142_native49152_forwards":true_forward_calls,
        "A_B":{"A" if arm.endswith("_CE") else "B":
            outarm["same_previous_SPENT14_physical_mode_component_ablation"]
            for arm,outarm in outcomes.items()},
        "candidate_accepted":False,"output_gate":0,
    },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8f_report","q2i8g_dir","q2i8k_dir","q2i8t_dir","q2i8w_dir","z10_dir","z29_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,
        a.q2i8f_report,a.q2i8g_dir,a.q2i8k_dir,a.q2i8t_dir,a.q2i8w_dir,a.z10_dir,a.z29_dir,a.out)

if __name__=="__main__":
    main()
