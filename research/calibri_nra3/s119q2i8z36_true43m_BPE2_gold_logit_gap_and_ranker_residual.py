"""S119q2i8z36 real S142 43M FROZEN native BPE2 logit margin forensics.

Previous REAL Z35 physical run #45259912788 SUCCESS, authentic
84 legal TRAIN + 14 now SPENT author Z34 DEV, separately fitted
four FP16 BPE2-only 30720param rank32 human-TRAIN and shifted17
label controls, no improved BPE2 token on independent NEW heldout.
WHY? Is original greedy vs gold2 native49k logit gap > mathematical
max +3.0 of bounded residual ±1.5, or does ranker fail to align
its score delta? EXACT original S142 43M/GT12/Core/NRA3/NPA1
real 2 native passes per previously SPENT human author14, replay
prior physical BPE1/BPE2 greedy baseline and each fitted ranker.
Read only original full pretrained + existing native FP16 heads,
score human BPE2 only POSTHOC. Zero new learning and no
new independent DEV; no hyperparameter adaptation on SPENT data.
Never change LIVE, main, Render, Supabase, APK, Jessica.
No J7/EVAL128/GOLD/bpp/CPU gate, output_gate0 research only.
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


from research.calibri_nra3.s119q2i8z34_fresh14dev_BPE2_only_source import (
    STAGE as NEW_SOURCE_STAGE,ORIGIN as NEW_SOURCE_ORIGIN,
    rows as fresh_new_rows,verify as fresh_new_verify,
    TOP_K,RANK,TRAIN_EPOCHS,EVAL_FIRST_NATIVE_BPE,
)
from research.calibri_nra3.s119q2i8z16_trainonly_lowrank_native64_reranker import (
    capture_existing_full43m_hidden,NativeBPE2CandidateReranker,
    HIDDEN,
)
from research.calibri_nra3.s119q2i8z34_position2_only_owncontext_top64_train import (
    Position2NativeCandidateComparator, MODES, SHADOW_PARAMS,
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

STAGE="CALIBRI_S119Q2I8Z36_READONLY_REAL43M_NATIVE49152_BPE2_LOGIT_MARGIN_SHADOW_RESIDUAL_SPENT14"
REPORT="S119Q2I8Z36_TRUE43M_SPENT14_BPE2_GREEDY_LOGIT_GAP_VS_BOUNDED_RANK32_RESIDUAL_G0.json"
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
        q2i8d_dir,q2i8f_report,q2i8g_dir,q2i8k_dir,q2i8t_dir,q2i8w_dir,z10_dir,z35_dir,out):
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
          and fresh["prior_legal_TRAIN84"]==84
          and fresh["brandNEW_author_synthetic_DEV14"]==14
          and fresh["fixed_second_native_BPE_only"]["native_full49152_candidate_topK"]==64
          and fresh["fixed_second_native_BPE_only"]["rank"]==32
          and fresh["fixed_second_native_BPE_only"]["autonomous_DEV_first_native_count"]==6
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
        ("TRAIN",fresh_train,84),("DEV",fresh_dev,14)
    ):
        for authored in fresh_new_rows(split):
            check(authored["split"]==split
                  and authored["id"].startswith("S119Q2I8Z34_")
                  and (authored["origin"]==NEW_SOURCE_ORIGIN if split=="DEV"
                       else authored["id"].startswith("S119Q2I8Z34_LEGAL_")),
                  "only 84 previous legal TRAIN or new14 original-author Z34 DEV")
            item=dict(authored)
            item["origin"]="original_newly_authored_examples_no_third_party_corpus"
            prepared=verified_native_answer(tok,item)
            check(prepared["split"]==split and
                  tok.decode(prepared["answer_native_without_eos"],
                      skip_special_tokens=True)==authored["completion"] and
                  4<=len(prepared["answer_native_with_eos"])<=48 and
                  len(prepared["prefix"])+2<160,
                  "true original native49152 source target and live context bounds")
            folder.append(prepared)
        check(len(folder)==expected and
              len({r["prompt_sha256"] for r in folder})==expected,
              "strict preregistered 84 past TRAIN or 14 brand-new DEV unique source SHA")
    check({r["prompt_sha256"] for r in fresh_train}.isdisjoint(
            r["prompt_sha256"] for r in fresh_dev),
          "no new DEV source or human target used in earlier training")
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
        # Entire previous Z35 source and fitted scores are SEALED.
        # Z34 original-author14 DEV was already physically scored,
        # may NEVER be used for retraining, model selection or tuning.
        prev_z35_path=z35_dir/(
            "S119Q2I8Z35_REAL43M_BPE2_ONLY_RANK32_84TRAIN14FRESH_DEV_FP16_G0.json")
        prev_z35=json.loads(prev_z35_path.read_text(encoding="utf8"))
        pinned_fits={
            "Z10_FULLJOINT_EARLY23_X3_CE":{
                "HUMAN_GOLD_BPE2":
                    "18546e46da099399164d5105c9d1968d23fd43ffbc4d57066131cc98677ff304",
                "WRONG_OFFSET17_LABEL_CONTROL":
                    "f594525154b102f006e2b92c73da773f8a7e388bd16e69b70465ba022917bb60",
            },
            "Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2":{
                "HUMAN_GOLD_BPE2":
                    "5b5895721c937318c255f6b42528e5b5f0bef5c3787f18efb659ca9471d7f2c1",
                "WRONG_OFFSET17_LABEL_CONTROL":
                    "f04d2d9cffffc0a0671f59130a6de011ed617f1f4b48fa832769e1da1db195aa",
            },
        }
        check(prev_z35["stage"]==
                "CALIBRI_S119Q2I8Z35_REAL43M_NATIVE49152_ONLY_BPE2_RANK32_84TRAIN14DEV"
              and prev_z35["Z34_PRECOMMITTED_84_LEGAL_TRAIN_SHA256"]==
                  fresh["TRAIN_SHA256"]==
                  "d3955fe5a39b265edaa241f67ce1b92741882799767d4e2adb23c9291ad184a6"
              and prev_z35["Z34_PRECOMMITTED_14_BRANDNEW_DEV_SHA256"]==
                  fresh["NEW_DEV_SHA256"]==
                  "8d934384abee646c34156df9dbd51b6e31af23d1136fa2ddda9cae6dd91ca2d5"
              and prev_z35["Z34_brandNEW_author_DEV14_NOW_SPENT_not_independently_blind"] is True
              and prev_z35["original43M_49k_native_FORWARDS"]==772
              and prev_z35["candidate_accepted"] is False and prev_z35["output_gate"]==0
              and set(prev_z35["actual_original43M_A_B_real_native49k_BPE2_only_rank32"])==
                  set(pinned_fits),
              "Z35 completed REAL original S142 43M original spent14/new 4 fits exactly pinned")

        for name,child,gates in (
            ("Z10_FULLJOINT_EARLY23_X3_CE",child_a,gt_a),
            ("Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2",child_b,gt_b),
        ):
            started=time.monotonic()
            set_gt_route(full,"GT_RECURRENT",gates)
            prev_arm=prev_z35[
                "actual_original43M_A_B_real_native49k_BPE2_only_rank32"][name]
            original_first6={
                row["source_id"]:row
                for row in prev_arm["NEW_DEV14_per_prompt_first6"]
            }
            check(len(original_first6)==14 and
                  set(prev_arm["fitted_shadow_only_FP16_sha"])==set(MODES),
                  "completed Z35 real first6 already scored on exactly 14")

            # Original S142 onpath embeddings are already restored
            # from the true frozen original TrainEmbedding forward.
            # Load precisely the human-label ranker and the matched
            # WRONG label trained negative control (FP16 original file).
            tag="CE" if name.endswith("_CE") else "CONTEXT_TOP8"
            fitted={}
            for mode in MODES:
                checkpoint=prev_arm["fitted_shadow_only_FP16_sha"][mode]
                label="HUMAN_BPE2_RANK32" if mode==MODES[0] else "WRONG_OFFSET17_CONTROL"
                fp=z35_dir/(
                    "S119Q2I8Z35_BPE2_ONLY_NATIVE64_"+tag+"_"+label+"_FP16_G0.safetensors")
                check(fp.exists() and checkpoint["filename"]==fp.name
                      and checkpoint["params"]==SHADOW_PARAMS==30720
                      and checkpoint["candidate_accepted"] is False
                      and checkpoint["output_gate"]==0
                      and sha(fp)==checkpoint["sha256"]==pinned_fits[name][mode],
                      "exact original fitted Z35 30720 rank32 shadow checkpoint SHA")
                weights=load_file(str(fp),device="cpu")
                shadow=Position2NativeCandidateComparator()
                check(set(shadow.state_dict())==set(weights)==
                      {"input_project","output_project"} and
                      all(v.dtype==torch.float16 and
                          v.shape==shadow.state_dict()[k].shape
                          for k,v in weights.items()),
                      "only physical Z35 FP16 trained BPE2 tiny neural shadow")
                shadow.load_state_dict(weights,strict=True)
                shadow.eval()
                check(shadow.output_gate==0 and
                      shadow.candidate_accepted is False,
                      "fitted BPE2-only output_gate0 immutable")
                fitted[mode]=shadow

            seen=[]
            native_forward_count=0
            with torch.no_grad():
                for row in fresh_dev:
                    previous=original_first6.get(row["id"])
                    check(previous is not None and
                          previous["prompt_SHA256"]==row["prompt_sha256"] and
                          set(previous["auto_generated_modes_no_teacher_forcing"])==
                              {"ORIGINAL_NATIVE_GREEDY",*MODES} and
                          previous["human_gold_native_first4_POSTHOC"]==
                              row["answer_native_with_eos"][:4],
                          "same originally scored original-author Z34 14 DEV, gold target posthoc only")
                    with scoped_exact_native_gain(child.language_head,.25) as count:
                        first_log=real_joint_native(
                            model,sources,bridge,full,expansion,child,
                            row["prefix"],row["prompt_features"],contextual=True)
                        check(first_log.shape==(1,NATIVE_VOCAB)
                              and full.last_forward_coverage==tuple(range(12)),
                              "one genuine frozen S142 43M native first BPE")
                        first=int(first_log[0].argmax().item())
                        with capture_existing_full43m_hidden(model) as hidden:
                            logits=real_joint_native(
                                model,sources,bridge,full,expansion,child,
                                row["prefix"]+[first],row["prompt_features"],
                                contextual=True)
                    native_forward_count+=2
                    check(count[0]==2 and len(hidden)==1 and
                          hidden[0].shape==(1,HIDDEN) and
                          logits.shape==(1,NATIVE_VOCAB)
                          and full.last_forward_coverage==tuple(range(12)),
                          "two actual original S142 43M native full49152 passes, same paid h320")
                    prevmode=previous["auto_generated_modes_no_teacher_forcing"]
                    baseline_native=prevmode["ORIGINAL_NATIVE_GREEDY"][
                        "autonomous_first_six_native_ids"]
                    check(len(baseline_native)>=2 and
                          first==baseline_native[0] and
                          int(logits[0].argmax())==baseline_native[1] and
                          all(prevmode[m]["autonomous_first_six_native_ids"][0]==first
                              for m in MODES),
                          "strict PREVIOUS Z35 real original S142 own BPE1 BPE2 parity (not new benchmark)")
                    gold1=int(row["answer_native_with_eos"][0])
                    gold2=int(row["answer_native_with_eos"][1])
                    native=logits.detach().float()[0]
                    original_greedy2=int(native.argmax())
                    gap=float(native[original_greedy2]-native[gold2])
                    targetblind_top64_ids=torch.topk(native,64,sorted=True).indices
                    gold_inside_original_top64=bool(
                        (targetblind_top64_ids==gold2).any())
                    correct_own_first=first==gold1
                    # Human gold after wrong OWN BPE1 is NOT a valid
                    # conditional training target: only infer true
                    # correctness gaps when own first BPE matches.
                    scored={}
                    for mode in MODES:
                        ids,scores,residual=fitted[mode](
                            logits.detach(),hidden[0].detach(),
                            native_embedding[first].detach().view(1,-1),
                            native_embedding,native_bpe_position=2)
                        check(torch.equal(ids,targetblind_top64_ids)
                              and bool((residual.abs()<=1.50001).all())
                              and len(scores)==64
                              and ids.unique().numel()==64,
                              "same ORIGINAL targetblind native top64, bounded only BPE2 residual")
                        learned_id=int(ids[int(scores.argmax())])
                        prior_selected=prevmode[mode]["autonomous_first_six_native_ids"][1]
                        check(learned_id==prior_selected,
                              "fitted Z35 FP16 exact original physical native BPE2 choice; fail-closed no tuning")
                        greedy_idx=int((ids==original_greedy2).nonzero().flatten().item())
                        if gold_inside_original_top64:
                            gold_idx=int((ids==gold2).nonzero().flatten().item())
                            net=float(residual[gold_idx]-residual[greedy_idx])
                            learned_gap=float(scores[greedy_idx]-scores[gold_idx])
                            gold_delta=float(residual[gold_idx])
                            greedy_delta=float(residual[greedy_idx])
                        else:
                            net=None
                            learned_gap=None
                            gold_delta=None
                            greedy_delta=float(residual[greedy_idx])
                        scored[mode]={
                            "native_output_from_saved_FP16_same_previous_Z35":learned_id,
                            "changed_original_native_greedy_BPE2":learned_id!=original_greedy2,
                            "learned_native_residual_max_abs":round(float(residual.abs().max()),6),
                            "learned_candidate_score_minmax_span":round(
                                float(residual.max()-residual.min()),6),
                            "greedy_original_residual":round(greedy_delta,6),
                            "gold_original_top64_residual":(
                                None if gold_delta is None else round(gold_delta,6)),
                            "gold_minus_greedy_learned_residual":(
                                None if net is None else round(net,6)),
                            "remaining_greedy_vs_gold_logit_gap":(
                                None if learned_gap is None else round(learned_gap,6)),
                            "native_gold_should_win_if_residual_exceeds_baseline_gap":
                                None if net is None else bool(net>gap),
                            "original_S142_43M_model_extra_forwards_for_candidate":0,
                        }
                    seen.append({
                        "source_id":row["id"],
                        "source_prompt_SHA256":row["prompt_sha256"],
                        "kind":row["kind"],
                        "human_first_BPE_matches_actual_OWN_first":correct_own_first,
                        "human_second_BPE_is_eligible_only_if_first_correct":correct_own_first,
                        "native_gold_BPE2_in_original_top64":gold_inside_original_top64,
                        "human_gold_BPE2_greedy_original_yes":original_greedy2==gold2,
                        "original_native_BPE2_gold_rank_full49152":int(
                            (native>native[gold2]).sum().item())+1,
                        "original_native49152_greedy_minus_gold_BPE2_logit_gap":
                            round(gap,6),
                        "original_gap_greater_than_THEORETICALLY_MAX_3_0_pairwise_correction":
                            bool(gap>3.0),
                        "only_2_true_original_S142_full49152_decoder_forwards":2,
                        "previous_Z35_NATIVE_own_first_and_second_replayed_exact":True,
                        "two_FROZEN_prior_Z35_rankers":scored,
                        "previously_SPENT_DEV14_not_training_or_validation":True,
                    })
            check(len(seen)==14
                  and len({x["source_prompt_SHA256"] for x in seen})==14
                  and native_forward_count==28,
                  "28 paid physical original 43M decoder forwards per arm 14 source*2")
            qualified=[row for row in seen if
                       row["human_first_BPE_matches_actual_OWN_first"]]
            qualified_top64=[row for row in qualified if
                             row["native_gold_BPE2_in_original_top64"]]
            stats={
                "previously_spent_DEV14_sources":14,
                "correct_own_native_BPE1_for_human_BPE2_valid":len(qualified),
                "correct_own_first_and_human_BPE2_in_original_top64":len(qualified_top64),
                "correct_first2_original_greedy":sum(
                    v["human_gold_BPE2_greedy_original_yes"] for v in qualified),
                "gold_BPE2_excluded_before_candidate":sum(
                    not v["native_gold_BPE2_in_original_top64"] for v in qualified),
                "original_fullnative49152_greedy_minus_gold_gap_GT3_0_correct_BPE1":sum(
                    v["original_gap_greater_than_THEORETICALLY_MAX_3_0_pairwise_correction"]
                    for v in qualified_top64),
                "original_fullnative49152_BPE2_gap_mean_correct_BPE1_top64":round(
                    sum(v["original_native49152_greedy_minus_gold_BPE2_logit_gap"]
                        for v in qualified_top64)/max(1,len(qualified_top64)),6),
                "absolute_max_possible_own_optimally_aligned_residual_pairwise":3.,
                "matched_shadow_results":{
                    m:{
                        "learned_ranker_changed_BPE2_on_correct_BPE1":sum(
                            v["two_FROZEN_prior_Z35_rankers"][m][
                                "changed_original_native_greedy_BPE2"]
                            for v in qualified),
                        "learned_ranker_realised_gold_vs_greedy_residual_GT_original_gap":sum(
                            v["two_FROZEN_prior_Z35_rankers"][m][
                                "native_gold_should_win_if_residual_exceeds_baseline_gap"]
                            for v in qualified_top64),
                        "learned_ranker_mean_gold_minus_greedy_native_residual":round(
                            sum(v["two_FROZEN_prior_Z35_rankers"][m][
                                "gold_minus_greedy_learned_residual"]
                                for v in qualified_top64)/max(1,len(qualified_top64)),6),
                        "learned_ranker_mean_max_abs_residual":round(
                            sum(v["two_FROZEN_prior_Z35_rankers"][m][
                                "learned_native_residual_max_abs"] for v in qualified)/max(1,len(qualified)),6),
                    }
                    for m in MODES
                },
                "real_original_S142_43M_native49152_forwards":native_forward_count,
            }
            outcomes[name]={
                "original_Z35_30720param_FP16_SHAs":pinned_fits[name],
                "previous_real_Z35_Z34_spent14_per_prompt_native_BPE2":seen,
                "physical_logit_gap_vs_native_rank32_bounded_score_causal_diagnostic":stats,
                "full_original_S142_native49152_decoder_forwards":native_forward_count,
                "uncontrolled_CPU_wall_seconds":round(time.monotonic()-started,3),
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
          and all(v["NEW_DEV14_full_autonomous_first6_native"][
                        "original43M_real_native_TRAIN_forwards"]==168
                  and v["NEW_DEV14_full_autonomous_first6_native"][
                        "brand_NEW_synthetic_DEV14"]==14
                  and set(v["TRAIN_only_shadow_optimizer"])==set(MODES)
                  and all(v["TRAIN_only_shadow_optimizer"][m]["legal_TRAIN84"]==84
                          and v["TRAIN_only_shadow_optimizer"][m]["shadow_rank32_params"]==30720
                          for m in MODES)
                  and len(v["NEW_DEV14_per_prompt_first6"])==14
                  for v in outcomes.values()),
          "real original S142 A/B native49152 BPE2-only rank32 with legal84 source only")
    total_forwards=sum(
        v["NEW_DEV14_full_autonomous_first6_native"][
            "original43M_real_native_TRAIN_forwards"]+
        v["NEW_DEV14_full_autonomous_first6_native"][
            "original43M_real_native_DEV_forwards"]
        for v in outcomes.values())
    check(350<=total_forwards<=900,
          "exact original S142 43M true native49152 two real full forwards per TRAIN and own first6 per DEV arm and mode")
    report={
        "stage":STAGE,
        "Z34_PRECOMMITTED_84_LEGAL_TRAIN_SHA256":fresh["TRAIN_SHA256"],
        "Z34_PRECOMMITTED_14_BRANDNEW_DEV_SHA256":fresh["NEW_DEV_SHA256"],
        "Z34_brandNEW_author_DEV14_NOW_SPENT_not_independently_blind":True,
        "fixed_rank32_candidate_K64_bound1_5_correct_vs_wrong_control_epoch25":True,
        "strict_ONLY_second_native_BPE_candidate_decoding_intervention":True,
        "original43M_49k_native_FORWARDS":total_forwards,
        "original_Z10_fitted_parent_SHA256":sha(prev_path),
        "saved_original_Z10_A_B_FPF16_and_12GT_SHA256":{
            k:v["sha256"] for k,v in saved.items()},
        "original_S142_43M_SHA256":sha(s142),
        "original_S93_GT_SHA256":sha(s93),
        "actual_original43M_A_B_real_native49k_BPE2_only_rank32":outcomes,
        "no_gold_in_candidate_selection_on_new_DEV":True,
        "no_previous_SPENT_DEV_used_in_TRAIN":True,
        "only_two_30720param_BPE2_gated_shadows_updated_per_arm":True,
        "original_S142_43M_GT12_Core_NRA3_NPA1_and_Z10_A_B_unchanged":True,
        "no_616J7_EVAL128_GOLD_bpp_CB59_CPU_certification":True,
        "no_MAIN_LIVE_Render_Supabase_APK_Jessica_changes":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"SECOND_NATIVE_BPE_ONLY_POSITION_GATED_TARGETBLIND_TOP64_RANK32_RESEARCH_NO_PROMOTION",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
                           sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z35_TRUE43M_NATIVE_BPE2_ONLY_RANK32_DEV_FIRST6 "+json.dumps({
        "Z34_TRAIN84_SHA256":fresh["TRAIN_SHA256"],
        "Z34_NEW_DEV14_SHA256":fresh["NEW_DEV_SHA256"],
        "REAL_original43M_fullnative49152_forwards":total_forwards,
        "A_B":{"A" if arm.endswith("_CE") else "B":{
            "DEV_first6":v["NEW_DEV14_full_autonomous_first6_native"],
            "fitted_TRAIN_only":v["TRAIN_only_shadow_optimizer"],
            "FP16_only_shadow_SHA256":{
                mode:rec["sha256"] for mode,rec in
                v["fitted_shadow_only_FP16_sha"].items()},
        } for arm,v in outcomes.items()},
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
