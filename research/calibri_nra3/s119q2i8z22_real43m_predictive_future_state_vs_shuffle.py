"""Z22 REAL ORIGINAL S142 43M + GT12 short-horizon native predictive proxy.

New Z21 synthetic 28 TRAIN/14 DEV. Actual native49152 BPE2 after OWN BPE1;
same-paid original last h320. Fixed 320->16->320 shadow predicts mean
human true BPE2..4 native embeddings; matched shuffled TRAIN control.
Target-free top64 and cosine rerank at DEV. No pretrained changes,
no 616J7/EVAL128/GOLD bpp CPU acceptance. output gate0.
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


from research.calibri_nra3.s119q2i8z21_fresh28train14dev_future_state_source import (
    STAGE as NEW_SOURCE_STAGE,ORIGIN as NEW_SOURCE_ORIGIN,
    rows as fresh_new_rows,verify as fresh_new_verify,TOP_K,LATENT_DIM,
)
from research.calibri_nra3.s119q2i8z16_trainonly_lowrank_native64_reranker import (
    capture_existing_full43m_hidden,NativeBPE2CandidateReranker,
    HIDDEN,
)
from research.calibri_nra3.s119q2i8z21_future_state_lowrank16_matched_contract import (
    target_next3_original_embeddings,train_matched_aligned_vs_permuted,STATES,
)
from safetensors.torch import save_file
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

STAGE="CALIBRI_S119Q2I8Z22_REAL43M_NEXT3_NATIVE_PREDICTIVE_STATE_ALIGNED_VS_PERMUTED_28TRAIN14DEV"
REPORT="S119Q2I8Z22_REAL43M_PREDICTIVE_NEXT3_VS_SHUFFLE_28TRAIN14DEV_FP16_G0.json"
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
        q2i8d_dir,q2i8f_report,q2i8g_dir,q2i8k_dir,q2i8t_dir,q2i8w_dir,z10_dir,out):
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
          and fresh["TRAIN_count"]==28 and fresh["DEV_count"]==14
          and fresh["fixed_topK"]==64 and fresh["latent_dim"]==16 and fresh["fixed_epochs"]==30
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
    fresh_train=[]
    fresh_dev=[]
    for split,folder,expected in (
        ("TRAIN",fresh_train,28),("DEV",fresh_dev,14)
    ):
        for authored in fresh_new_rows(split):
            check(authored["split"]==split
                  and authored["origin"]==NEW_SOURCE_ORIGIN
                  and authored["id"].startswith("S119Q2I8Z21_"),
                  "only new Z21 authored TRAIN or previously unseen DEV")
            item=dict(authored)
            item["origin"]="original_newly_authored_examples_no_third_party_corpus"
            prepared=verified_native_answer(tok,item)
            check(prepared["split"]==split and
                  tok.decode(prepared["answer_native_without_eos"],
                      skip_special_tokens=True)==authored["completion"] and
                  5<=len(prepared["answer_native_with_eos"])<=64 and
                  len(prepared["prefix"])+2<160,
                  "true original native49152 source target and live context bounds")
            folder.append(prepared)
        check(len(folder)==expected and
              len({r["prompt_sha256"] for r in folder})==expected,
              "strict preregistered 28 TRAIN or 14 DEV unique source SHA")
    check({r["prompt_sha256"] for r in fresh_train}.isdisjoint(
            r["prompt_sha256"] for r in fresh_dev),
          "no DEV source reused in shadow reranker human TRAIN")
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
        for name,child,gates in (
            ("Z10_FULLJOINT_EARLY23_X3_CE",child_a,gt_a),
            ("Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2",child_b,gt_b),
        ):
            set_gt_route(full,"GT_RECURRENT",gates)
            started=time.monotonic()
            train_cache=[]
            with torch.no_grad():
                for row in fresh_train:
                    with scoped_exact_native_gain(
                            child.language_head,.25) as witnessed:
                        first_logits=real_joint_native(
                            model,sources,bridge,full,expansion,child,
                            row["prefix"],row["prompt_features"],contextual=True)
                        check(first_logits.shape==(1,NATIVE_VOCAB)
                              and full.last_forward_coverage==tuple(range(12)),
                              "true original43M actual first-native OWN logit")
                        own_first=int(torch.argmax(first_logits[0]).item())
                        # Attach prehook to EXISTING native lm_head, not a
                        # second model forward; full h320 covers GT/NRA3/NPA1.
                        with capture_existing_full43m_hidden(model) as h:
                            second_logits=real_joint_native(
                                model,sources,bridge,full,expansion,child,
                                row["prefix"]+[own_first],row["prompt_features"],
                                contextual=True)
                        check(len(h)==1 and h[0].shape==(1,HIDDEN)
                              and second_logits.shape==(1,NATIVE_VOCAB)
                              and full.last_forward_coverage==tuple(range(12)),
                              "actual original43M h320 and full49k BPE2 same paid pass")
                    check(witnessed[0]==2,
                          "two physical original43M native forwards per TRAIN row")
                    gold=row["answer_native_with_eos"]
                    train_cache.append({
                        "source_id":row["id"],"split":"TRAIN",
                        "logits":second_logits.detach().float().cpu().clone(),
                        "hidden":h[0].detach().float().cpu().clone(),
                        "target_native_BPE2":int(gold[1]),
                        "own_first_correct":bool(own_first==gold[0]),
                    })
            check(len(train_cache)==28 and
                  all(r["split"]=="TRAIN" for r in train_cache),
                  "28 actual original43M TRAIN-only native BPE2 cached readouts")

            # All 25 fixed optimizer epochs affect ONLY separate
            # 1,280-param NEW shadow ranking weights. Neither original
            # model nor any DEV row is involved in training.
            reranker, train_record=exact_fixed_trainonly_reranker(
                train_cache,native_embedding)
            reranker.eval()
            check(train_record["TRAIN_only_eligible_original_TOP64_gold_second"]>=2
                  and train_record["original_model_gradient_steps"]==0
                  and train_record["reranker_trainable_params"]==1280,
                  "new28 real original TRAIN top64 only, fixed budget")
            old_trainable_model_count=368051
            observed=[]
            with torch.no_grad():
                for row in fresh_dev:
                    with scoped_exact_native_gain(
                            child.language_head,.25) as witnessed:
                        first_logits=real_joint_native(
                            model,sources,bridge,full,expansion,child,
                            row["prefix"],row["prompt_features"],contextual=True)
                        check(first_logits.shape==(1,NATIVE_VOCAB)
                              and full.last_forward_coverage==tuple(range(12)),
                              "true original43M first native OWN under original gates")
                        own_first=int(torch.argmax(first_logits[0]).item())
                        with capture_existing_full43m_hidden(model) as h:
                            second_logits=real_joint_native(
                                model,sources,bridge,full,expansion,child,
                                row["prefix"]+[own_first],
                                row["prompt_features"],contextual=True)
                        check(len(h)==1 and h[0].shape==(1,HIDDEN)
                              and second_logits.shape==(1,NATIVE_VOCAB)
                              and full.last_forward_coverage==tuple(range(12)),
                              "original43M output & captured same h320 GT12")
                    check(witnessed[0]==2,
                          "two real original43M native49k per DEV, no candidates")
                    # NO gold in candidate scoring or original shortlist.
                    ids,scores=reranker(
                        second_logits.detach(),h[0],native_embedding)
                    greedy=int(torch.argmax(second_logits[0]).item())
                    selected=int(ids[int(torch.argmax(scores).item())].item())
                    target=row["answer_native_with_eos"]
                    first_match=bool(own_first==target[0])
                    present=bool((ids==target[1]).any())
                    observed.append({
                        "source_id":row["id"],"kind":row["kind"],
                        "prompt_sha256":row["prompt_sha256"],
                        "own_first_native_correct":first_match,
                        "greedy_second_native_correct":
                            first_match and greedy==target[1],
                        "reranked_second_native_correct":
                            first_match and selected==target[1],
                        "gold_second_already_in_frozen_top64":
                            first_match and present,
                        "reranker_changed_second_native":selected!=greedy,
                        "actual_greedy_native_BPE2_ID":greedy,
                        "actual_reranked_native_BPE2_ID":selected,
                        "greedy_is_premature_native_EOS":greedy==2,
                        "reranked_is_premature_native_EOS":selected==2,
                        "candidate_native_top64_includes_EOS":bool((ids==2).any()),
                        "original_full43M_native_forwards":2,
                        "true_original_native_vocab_size":NATIVE_VOCAB,
                        "source_internal_DEV_NOW_SPENT_not_independent_blind":True,
                        "gold_only_posthoc_after_ranker_selection":True,
                    })
            check(len(observed)==14 and
                  len({r["prompt_sha256"] for r in observed})==14,
                  "ALL unseen Z16 DEV14 scored only after fixed TRAIN")
            stat={
                "n_NEW_Z16_DEV":14,
                "own_first_correct":sum(x["own_first_native_correct"] for x in observed),
                "correct_gold_second_in_native_top64":sum(
                    x["gold_second_already_in_frozen_top64"] for x in observed),
                "greedy_correct_second":sum(
                    x["greedy_second_native_correct"] for x in observed),
                "reranked_correct_second":sum(
                    x["reranked_second_native_correct"] for x in observed),
                "reranker_changed_second":sum(
                    x["reranker_changed_second_native"] for x in observed),
                "EOS_among_top64":sum(
                    x["candidate_native_top64_includes_EOS"] for x in observed),
                "greedy_premature_native_EOS":sum(
                    x["greedy_is_premature_native_EOS"] for x in observed),
                "reranked_premature_native_EOS":sum(
                    x["reranked_is_premature_native_EOS"] for x in observed),
                "unique_greedy_native_BPE2_IDs":len(set(
                    x["actual_greedy_native_BPE2_ID"] for x in observed)),
                "unique_reranked_native_BPE2_IDs":len(set(
                    x["actual_reranked_native_BPE2_ID"] for x in observed)),
                "true_original43M_full_native49k_forwards":84,
                "ranker_extra_original43M_forwards_per_sample":0,
                "saved_shadow_params":1280,
            }
            tag="CE" if name.endswith("_CE") else "CONTEXT_TOP8"
            path=out/("S119Q2I8Z17_SHADOW_NATIVE64_RANK2_"+tag+"_G0.safetensors")
            save_file({
                key:v.detach().cpu().half().contiguous()
                for key,v in reranker.state_dict().items()
            },str(path))
            check(path.is_file() and path.stat().st_size>2000,
                  "physical fitted 1280 shadow-only FP16, no pretrained edits")
            outcomes[name]={
                "train_only":train_record,
                "DEV_after_training_once":stat,
                "DEV_paired_source_details":observed,
                "fitted_SHADOW_rank2_FP16":{
                    "file":path.name,
                    "sha256":sha(path),
                    "bytes":path.stat().st_size,
                    "trainable_params":1280,
                    "candidate_accepted":False,
                    "output_gate":0,
                },
                "rough_uncontrolled_CPU_training_and_probe_seconds":
                    round(time.monotonic()-started,3),
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
    check(len(outcomes)==2 and
          all(v["DEV_after_training_once"]["true_original43M_full_native49k_forwards"]==84
              and v["train_only"]["new_Z16_TRAIN_records"]==28
              and v["train_only"]["TRAIN_only_eligible_original_TOP64_gold_second"]>=2
              for v in outcomes.values()),
          "real original43M A/B 28TRAIN+14DEV each exactly two original native passes")
    report={
        "stage":STAGE,
        "preregistered_new_Z16_28TRAIN_sha256":fresh["TRAIN_sha256"],
        "preregistered_new_Z16_14DEV_sha256":fresh["DEV_sha256"],
        "new_DEV_is_now_SPENT_not_independent_blind":True,
        "fixed_native_second_K":K,
        "fixed_reranker_rank":RANK,
        "fixed_25_train_epochs":TRAIN_EPOCHS,
        "original_full43M_native49152_forwards_A_B_total":168,
        "base_original43M_frozen_physical_Z10_report_sha256":sha(prev_path),
        "base_Z10_421793_FP16_and_GT12_per_arm_SHA256":{
            k:v["sha256"] for k,v in saved.items()},
        "original_S142_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "frozen_full43M_A_B_newTRAIN_only_lowrank64_and_NEWDEV_once":outcomes,
        "NO_DEV_gradient_or_gold_in_inference_candidate_generation":True,
        "only_new_1280_rank2_shadow_params_learned_per_arm":True,
        "source_stages_all_precommitted":True,
        "native_original43M_and_protected_S115_unchanged":True,
        "no_616J7_EVAL128_GOLD_allin_bpp_or_CPU_CB59_certification":True,
        "no_main_LIVE_Render_Supabase_APK_Jessica_changes":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"PHYSICAL_TINY_SHADOW_RANKER_RESEARCH_ONLY_NO_PROMOTION",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
                           sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z17_PHYSICAL_ORIGINAL43M_TRAINONLY_NATIVE64 "+json.dumps({
        "new28_TRAIN_sha256":fresh["TRAIN_sha256"],
        "new14_DEV_sha256":fresh["DEV_sha256"],
        "original43M_native_forwards_all":168,
        "actual_A_B":{"A" if k.endswith("_CE") else "B":{
            "trainonly":v["train_only"],
            "DEV14":v["DEV_after_training_once"],
            "shadow_rank2_sha256":v["fitted_SHADOW_rank2_FP16"]["sha256"],
        } for k,v in outcomes.items()},
        "candidate_accepted":False,"output_gate":0,
    },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8f_report","q2i8g_dir","q2i8k_dir","q2i8t_dir","q2i8w_dir","z10_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,
        a.q2i8f_report,a.q2i8g_dir,a.q2i8k_dir,a.q2i8t_dir,a.q2i8w_dir,a.z10_dir,a.out)

if __name__=="__main__":
    main()
