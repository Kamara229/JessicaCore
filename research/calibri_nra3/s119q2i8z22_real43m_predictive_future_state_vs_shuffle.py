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
    rows as fresh_new_rows,verify as fresh_new_verify,TOP_K,LATENT_DIM,TRAIN_EPOCHS,
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
                    with scoped_exact_native_gain(child.language_head,.25) as witnessed:
                        first_logits=real_joint_native(
                            model,sources,bridge,full,expansion,child,
                            row["prefix"],row["prompt_features"],contextual=True)
                        check(first_logits.shape==(1,NATIVE_VOCAB)
                              and full.last_forward_coverage==tuple(range(12)),
                              "real original43M all12GT first OWN native BPE")
                        own_first=int(first_logits[0].argmax().item())
                        with capture_existing_full43m_hidden(model) as h:
                            second_logits=real_joint_native(
                                model,sources,bridge,full,expansion,child,
                                row["prefix"]+[own_first],row["prompt_features"],
                                contextual=True)
                        check(len(h)==1 and h[0].shape==(1,HIDDEN)
                              and second_logits.shape==(1,NATIVE_VOCAB)
                              and full.last_forward_coverage==tuple(range(12)),
                              "actual S142 native49152 BPE2 and same already-paid h320")
                    check(witnessed[0]==2,"exactly two original43M TRAIN forwards")
                    gold=row["answer_native_with_eos"]
                    train_cache.append({
                        "source_id":row["id"],"split":"TRAIN",
                        "hidden":h[0].detach().float().cpu().clone(),
                        "original_native_logits":second_logits.detach().float().cpu().clone(),
                        "target_future_embedding":
                            target_next3_original_embeddings(native_embedding,gold),
                        "own_first_correct":bool(own_first==gold[0]),
                    })
            check(len(train_cache)==28 and
                  all(r["split"]=="TRAIN" for r in train_cache)
                  and len({r["source_id"] for r in train_cache})==28,
                  "only actual preregistered Z21 human 28 TRAIN")
            # Strictly matched lowrank16 future state shadows: ALIGN and
            # deranged human TRAIN target. No Z21 DEV in training.
            states,train_meta=train_matched_aligned_vs_permuted(
                train_cache,native_embedding)
            check(set(states)==set(STATES) and
                  all(x["trainable_shadow_state_params"]==10240
                      and x["original43M_gradient_updates"]==0
                      for x in train_meta.values()),
                  "new 10240-param native future BPE2..4 proxy vs shuffled control")
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
                              "actual first own native BPE Z21 unseen DEV")
                        own_first=int(first_logits[0].argmax().item())
                        with capture_existing_full43m_hidden(model) as h:
                            second_logits=real_joint_native(
                                model,sources,bridge,full,expansion,child,
                                row["prefix"]+[own_first],row["prompt_features"],
                                contextual=True)
                        check(len(h)==1 and h[0].shape==(1,HIDDEN)
                              and second_logits.shape==(1,NATIVE_VOCAB)
                              and full.last_forward_coverage==tuple(range(12)),
                              "same S142 paid forward h320 for both states")
                    check(witnessed[0]==2,
                          "two original S142 full native DEV forwards only")
                    # BEFORE reading human future gold, both state shadow
                    # rankers build their top64 from ORIGINAL BPE2 logits.
                    candidates=torch.topk(second_logits[0],TOP_K).indices
                    native_greedy=int(second_logits[0].argmax().item())
                    choices={}
                    for state_name in STATES:
                        ids,score,pred=states[state_name].predict_future_candidates(
                            second_logits.detach(),h[0],native_embedding)
                        check(torch.equal(ids,candidates)
                              and len(ids)==TOP_K,
                              "no human gold insertion to original native top64")
                        choice=int(ids[int(score.argmax().item())].item())
                        choices[state_name]={
                            "second_native_BPE_id":choice,
                            "changed_original_greedy":choice!=native_greedy,
                            "premature_EOS":choice==2,
                            "predicted_native_future_h320":
                                pred.detach().float().clone(),
                        }
                    # DEV gold enters ONLY after both choices are FROZEN;
                    # compute evaluation/summary, no DEV gradient or tuning.
                    gold=row["answer_native_with_eos"]
                    first_ok=bool(own_first==gold[0])
                    future_truth=target_next3_original_embeddings(
                        native_embedding,gold)
                    coverage=bool(first_ok and (candidates==gold[1]).any())
                    per_mode={}
                    for state_name in STATES:
                        pred=choices[state_name].pop("predicted_native_future_h320")
                        cosine=float(torch.nn.functional.cosine_similarity(
                            pred,future_truth,dim=-1).item())
                        per_mode[state_name]={
                            **choices[state_name],
                            "second_correct":bool(
                                first_ok and choices[state_name]["second_native_BPE_id"]==gold[1]),
                            "future_BPE2_3_4_embedding_cosine_POSTHOC":cosine,
                        }
                    observed.append({
                        "source_id":row["id"],"kind":row["kind"],
                        "prompt_SHA256":row["prompt_sha256"],
                        "own_FIRST_correct":first_ok,
                        "gold_SECOND_reachable_in_original_TOP64":coverage,
                        "original_greedy_SECOND_correct":bool(
                            first_ok and native_greedy==gold[1]),
                        "original_greedy_second_id":native_greedy,
                        "modes":per_mode,
                        "original43M_FULLnative49152_forwards":2,
                        "Z21_DEV_NOW_SPENT_not_independent_blind":True,
                        "human_gold_only_POSTHOC_after_selection":True,
                    })
            check(len(observed)==14 and
                  len({x["prompt_SHA256"] for x in observed})==14,
                  "new Z21 DEV14 scored once only after ALL 30 TRAIN epochs")
            score={
                "unseen_Z21_DEV14":14,
                "own_first_correct":sum(x["own_FIRST_correct"] for x in observed),
                "human_gold_second_in_original_top64":sum(
                    x["gold_SECOND_reachable_in_original_TOP64"] for x in observed),
                "original_greedy_second_correct":sum(
                    x["original_greedy_SECOND_correct"] for x in observed),
                "matched_future_states":{
                    state_name:{
                        "second_correct":sum(
                            x["modes"][state_name]["second_correct"] for x in observed),
                        "changed_original_greedy":sum(
                            x["modes"][state_name]["changed_original_greedy"]
                            for x in observed),
                        "premature_EOS":sum(
                            x["modes"][state_name]["premature_EOS"] for x in observed),
                        "distinct_second_ids":len(set(
                            x["modes"][state_name]["second_native_BPE_id"]
                            for x in observed)),
                        "mean_future_BPE2_3_4_embedding_cosine_POSTHOC":round(
                            sum(x["modes"][state_name][
                                "future_BPE2_3_4_embedding_cosine_POSTHOC"]
                                for x in observed)/14,6),
                    } for state_name in STATES
                },
                "true_original43M_all12GT_native49152_forwards":84,
                "extra_original43M_native_forwards_for_state":0,
            }
            tag="CE" if name.endswith("_CE") else "CONTEXT_TOP8"
            saved_states={}
            for state_name in STATES:
                label="ALIGNED" if state_name==STATES[0] else "PERMUTED_TRAIN"
                f=out/("S119Q2I8Z22_SHADOW_FUTURE16_"+tag+"_"+label+"_FP16_G0.safetensors")
                save_file({key:v.detach().cpu().half().contiguous()
                          for key,v in states[state_name].state_dict().items()},str(f))
                check(f.is_file() and f.stat().st_size>20000,
                      "saved only physical 10240-param FP16 proxy not original")
                saved_states[state_name]={
                    "filename":f.name,"sha256":sha(f),
                    "bytes":f.stat().st_size,"trainable_params":10240,
                    "candidate_accepted":False,"output_gate":0,
                }
            outcomes[name]={
                "new_Z21_TRAIN_only_per_state":train_meta,
                "new_Z21_DEV14_once":score,
                "new_Z21_DEV14_per_source":observed,
                "only_shadow_lowrank16_FP16_files":saved_states,
                "uncontrolled_CPU_seconds":round(time.monotonic()-started,3),
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
          and all(v["new_Z21_DEV14_once"][
              "true_original43M_all12GT_native49152_forwards"]==84
              and set(v["new_Z21_TRAIN_only_per_state"])==set(STATES)
              and all(v["new_Z21_TRAIN_only_per_state"][state][
                  "new_Z21_TRAIN_source_total"]==28 for state in STATES)
              for v in outcomes.values()),
          "same actual original S142 43M A/B 28TRAIN+14DEV and two matched states")
    report={
        "stage":STAGE,
        "previously_preregistered_Z21_TRAIN28_SHA256":fresh["TRAIN_SHA256"],
        "previously_preregistered_Z21_DEV14_SHA256":fresh["DEV_SHA256"],
        "Z21_DEV14_NOW_SPENT_NOT_independent_blind":True,
        "fixed_320_to16_to320_shadow_params":10240,
        "fixed_future_BPE_positions":[1,2,3],
        "fixed_native_top64_original49152":TOP_K,
        "fixed_30_TRAIN_epochs":TRAIN_EPOCHS,
        "original43M_A_B_total_real_native49152_forwards":168,
        "source_completed_Z10_original43M_report_sha256":sha(prev_path),
        "source_Z10_A_B_421793_FP16_and_12_GT_SHA256":{
            k:v["sha256"] for k,v in saved.items()},
        "original_S142_43M_SHA256":sha(s142),
        "original_S93_GT_SHA256":sha(s93),
        "actual_original43M_ALIGNED_vs_PERMUTED_future_embedding_state":outcomes,
        "no_DEV_gold_in_state_prediction_or_candidate_generation":True,
        "only_new_Z21_TRAIN_human_BPE2_BPE3_BPE4_embedding_supervision":True,
        "permuted_control_uses_ONLY_shuffled_Z21_TRAIN_targets":True,
        "original43M_all12GT_Core_NRA3_NPA1_native_head_unchanged":True,
        "no_claim_general_semantic_planning_or_full_sentences":True,
        "no_616J7_EVAL128_GOLD_packed_bpp_CB59_CPU_acceptance":True,
        "no_main_LIVE_Render_Supabase_APK_Jessica_changes":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"PREDICT_NEXT3_ORIGINAL_NATIVE_EMBEDDING_PROXY_RESEARCH_ONLY_GATE0",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
                           sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z22_REAL_ORIGINAL43M_FUTURE_STATE_VS_SHUFFLE "+json.dumps({
        "Z21_TRAIN28_SHA256":fresh["TRAIN_SHA256"],
        "Z21_DEV14_SHA256":fresh["DEV_SHA256"],
        "original43M_real_native49152_forwards":168,
        "A_B":{"A" if arm.endswith("_CE") else "B":{
            "DEV14":outcome["new_Z21_DEV14_once"],
            "train_only":outcome["new_Z21_TRAIN_only_per_state"],
            "shadow_SHA256":{
                state:fp["sha256"] for state,fp in
                outcome["only_shadow_lowrank16_FP16_files"].items()},
        } for arm,outcome in outcomes.items()},
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
