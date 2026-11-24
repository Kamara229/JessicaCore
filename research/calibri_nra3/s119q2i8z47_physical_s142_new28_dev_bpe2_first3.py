"""Z47 PHYSICAL precommitted Z46 FRESH28 unseen internal RU DEV, original43M.

One exact SHA-verified original S142 43M native49152 model+GT12+Z10 CE
frozen parent. Fit two 30720-param shadow BPE2 rank32 on SAME 84 legal
previous TRAIN, genuine OWN correct first tokens only, gold already in
original TOP64. TRAIN 8 epochs, human vs WRONG-offset17 matched control.
One new 28 DEV evaluation (this evaluation SPENDS them), original native
greedy vs HUMAN vs WRONG, naturally generated three BPE positions;
BPE1 and BPE3 remain full original model greedy, only BPE2 corrected
with fixed targetblind original TOP64 KL/entropy guard. Report all
preregistered diversity and exact-first2/3 thresholds regardless of
success. No model promotion, no Jessica, no sealed GOLD/J7/EVAL128.
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


from research.calibri_nra3.s119q2i8z40_new14dev_multiposition_entropy_prereg import (
    STAGE as NEW_SOURCE_STAGE,ORIGIN as NEW_SOURCE_ORIGIN,
    rows as fresh_new_rows,verify as fresh_new_verify,
    TOP_K,RANK,TRAIN_EPOCHS,SHADOW_PARAMS,POSITIONS,
    EVAL_OWN_NATIVE_TOKENS as EVAL_FIRST_NATIVE_BPE,
)
from research.calibri_nra3.s119q2i8z16_trainonly_lowrank_native64_reranker import (
    capture_existing_full43m_hidden,NativeBPE2CandidateReranker,
    HIDDEN,
)
from research.calibri_nra3.s119q2i8z40_causal_BPE234_entropy_guard import (
    CausalBPE234CandidateComparator,MODES,fit_controls,
)
from safetensors.torch import save_file
from research.calibri_nra3.s119q2i8z45_trainonly_onpolicy_bpe2_domain_holdout import (
    validate as validate_onpolicy, SOURCE_SHA as Z45_TRAIN_SHA,
)
from research.calibri_nra3.s119q2i8z46_fresh28_onpolicy_bpe2_preregister import (
    verify as z46_source_verify,rows as z46_rows,
)
from research.calibri_nra3.s119q2i8z47_fresh28_native_bpe2_control_eval import (
    fit_all_old84,pick_second,scores as score_newdev,MODES as Z47_MODES,
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

STAGE="S119Q2I8Z47_PHYSICAL_REAL43M_FRESH28_DEV_BPE2_ONPOLICY_AND_CROSSPROMPT_DIVERSITY"
REPORT="S119Q2I8Z47_REAL43M_FRESH28_DEV_BPE2_FIRST3_NATIVE_HUMAN_VS_WRONG_G0.json"
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
        q2i8d_dir,q2i8f_report,q2i8g_dir,q2i8k_dir,q2i8t_dir,q2i8w_dir,z10_dir,z41_report,z43_report,out):
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
          and fresh["previous_legal_TRAIN84"]==84
          and fresh["NEW_original_author_internal_DEV14"]==14
          and fresh["frozen_physical_config"]["topk_native49152"]==64
          and fresh["frozen_physical_config"]["rank"]==32
          and tuple(fresh["frozen_physical_config"]["allowed_BPE_positions"])==(2,3,4)
          and fresh["frozen_physical_config"]["shadow_params"]==31680
          and fresh["frozen_physical_config"]["own_native_generation_steps"]==6
          and HEAD_GAIN==.25,
          "preregistered Z12 original author 14 synthetic DEV only")
    prereg=z46_source_verify()
    check(prereg["brand_NEW_28_internal_DEV_SHA256"]==
          "c4489970fbb0868f47c15fe6cc47b348ec4c533e094e0775f2275e00bd4bc58c"
          and prereg["previous_84_legal_TRAIN_SHA256"]==Z45_TRAIN_SHA
          and prereg["source_disjoint_all_known_prior_spent_DEV_and_train"] is True
          and prereg["NEW_28_INTERNAL_DEV"]==28
          and prereg["candidate_accepted"] is False
          and prereg["output_gate"]==0
          and prereg["frozen_physical_config"]["eval_own_native_positions"]==3,
          "precommitted before scoring unused Z46 author DEV28 and fixed metrics")
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
    # The Z40 internal DEV14 is ALREADY spent. Do not prepare or forward it.
    # The old physical Z41 report is read to recover teacher-prefix coverage
    # already computed, never to optimize a new candidate.
    archived=json.loads(z41_report.read_text(encoding="utf8"))
    check(archived["stage"]==
          "CALIBRI_S119Q2I8Z41_REAL43M_NATIVE49152_CAUSAL_BPE234_ENTROPY_GUARDED_84TRAIN14DEV"
          and archived["Z40_PRECOMMITTED_84_LEGAL_TRAIN_SHA256"]==fresh["TRAIN_SHA256"]
          and archived["Z40_PRECOMMITTED_14_BRANDNEW_DEV_SHA256"]==fresh["DEV_SHA256"]
          and archived["original43M_49k_native_FORWARDS"]==913
          and archived["candidate_accepted"] is False and archived["output_gate"]==0
          and archived["original_S142_43M_SHA256"]==sha(s142)
          and archived["original_S93_GT_SHA256"]==sha(s93),
          "source-verified previous successful physical Z41 teacher-prefix TRAIN84 only")
    z43=json.loads(z43_report.read_text(encoding="utf8"))
    check(z43["stage"]==
          "CALIBRI_S119Q2I8Z43_REAL43M_LEGAL_TRAIN84_ONPOLICY_NATIVE_BPE2_3_4_G0"
          and z43["candidate_accepted"] is False and z43["output_gate"]==0
          and z43["Z40_previous_legal_TRAIN84_SHA256"]==Z45_TRAIN_SHA
          and z43["first_native_exact_on_84_TRAIN"]==77
          and z43["first_three_native_exact_on_84_TRAIN"]==0
          and z43["real_original43M_native_49152_decoder_forwards"]==332
          and z43["original_S142_43M_sha256"]==sha(s142)
          and z43["original_S93_GT_sha256"]==sha(s93)
          and z43["previous_Z40_DEV14_remains_SPENT_not_used"] is True,
          "same completed 43M S142 previous legal84 physically verified Z43 audit")
    fresh_train=[]
    for authored in fresh_new_rows("TRAIN"):
        check(authored["split"]=="TRAIN"
              and authored["id"].startswith("S119Q2I8Z40_LEGAL_")
              and authored["origin"]!=NEW_SOURCE_ORIGIN,
              "only Z40 previous legitimate TRAIN84; no Z40 DEV examples")
        item=dict(authored)
        item["origin"]="original_newly_authored_examples_no_third_party_corpus"
        row=verified_native_answer(tok,item)
        check(row["split"]=="TRAIN"
              and tok.decode(row["answer_native_without_eos"],
                  skip_special_tokens=True)==authored["completion"]
              and len(row["answer_native_with_eos"])>=4
              and len(row["prefix"])+4<160,
              "real native original49152 previous TRAIN only")
        fresh_train.append(row)
    check(len(fresh_train)==84
          and len({r["prompt_sha256"] for r in fresh_train})==84
          and fresh["TRAIN_SHA256"]==
              "a66833e8f4e83e4ba080dc3db0b3032a3315bcfc2fe7cee473e7e9ea7569023e",
          "exact frozen 84 legal TRAIN only and pre-existing SHA")

    new_dev=[]
    for authored in z46_rows("DEV"):
        check(authored["split"]=="DEV"
              and authored["id"].startswith("S119Q2I8Z46_")
              and authored["origin"]==
                  "author_internal_new_Z46_20261124_nativeBPE2_entropy_crossprompt_holdout",
              "Z46 prereg authored new28 only, no previously SPENT DEV")
        item=dict(authored)
        item["origin"]="original_newly_authored_examples_no_third_party_corpus"
        row=verified_native_answer(tok,item)
        check(row["split"]=="DEV"
              and tok.decode(row["answer_native_without_eos"],
                             skip_special_tokens=True)==authored["completion"]
              and len(row["answer_native_with_eos"])>=4
              and len(row["prefix"])+4<160,
              "real original native49152 new Z46 DEV complete answer BPE")
        new_dev.append(row)
    check(len(new_dev)==28
          and len({r["prompt_sha256"] for r in new_dev})==28
          and {r["prompt_sha256"] for r in new_dev}.isdisjoint({
              r["prompt_sha256"] for r in fresh_train}),
          "28 new pinned author DEV, original tokenizer no overlap with TRAIN")
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
    child=child_a
    set_gt_route(full,"GT_RECURRENT",gt_a)
    # Real S142's compressed TRAINED input TrainEmbedding is a custom module,
    # not a bare nn.Embedding.weight. Read the original native49,152 table
    # in detached 1024-ID chunks exactly as successful original Z41 did.
    embedding_module=model.get_input_embeddings()
    pieces=[]
    with torch.no_grad():
        for start in range(0,NATIVE_VOCAB,1024):
            idx=torch.arange(start,min(start+1024,NATIVE_VOCAB),
                             dtype=torch.long).view(1,-1)
            out_emb=embedding_module(idx)
            check(out_emb.shape==(1,idx.numel(),HIDDEN)
                  and torch.isfinite(out_emb).all(),
                  "original S142 pretrained compressed native TrainEmbedding")
            pieces.append(out_emb[0].detach().float().cpu().clone())
    native_embedding=torch.cat(pieces,dim=0)
    check(native_embedding.shape==(NATIVE_VOCAB,HIDDEN)
          and not native_embedding.requires_grad and
          torch.isfinite(native_embedding).all(),
          "read-only genuine original native49,152 embeddings")
    records=[]
    original_forwards=0
    started=time.monotonic()
    try:
        with torch.no_grad():
            for row in fresh_train:
                gold=row["answer_native_with_eos"]
                check(len(gold)>=4,"legal TRAIN84 original native gold BPE2")
                with scoped_exact_native_gain(child.language_head,.25) as called:
                    first_logits=real_joint_native(
                        model,sources,bridge,full,expansion,child,
                        row["prefix"],row["prompt_features"],contextual=True)
                    check(first_logits.shape==(1,NATIVE_VOCAB)
                          and full.last_forward_coverage==tuple(range(12)),
                          "S142 original43M all12GT true OWN first native")
                    first=int(first_logits[0].argmax().item())
                    check(first!=2,"Z43 original TRAIN84 no early EOS at BPE1")
                    with capture_existing_full43m_hidden(model) as h:
                        second_logits=real_joint_native(
                            model,sources,bridge,full,expansion,child,
                            row["prefix"]+[first],
                            row["prompt_features"],contextual=True)
                    check(second_logits.shape==(1,NATIVE_VOCAB)
                          and len(h)==1 and h[0].shape==(1,HIDDEN)
                          and full.last_forward_coverage==tuple(range(12)),
                          "paid actual frozen S142 true OWN second 49152 logits+h320")
                    check(called[0]==2,"exact 2 paid original43M native passes per legal TRAIN")
                    original_forwards+=called[0]
                    records.append({
                        "source_id":row["id"],"kind":row["kind"],
                        "split":"TRAIN",
                        "native_logits":second_logits.detach().float().cpu().clone(),
                        "h320":h[0].detach().float().cpu().clone(),
                        "prev_own_embed":native_embedding[first].view(1,-1).clone(),
                        "gold_second":int(gold[1]),
                        "own_first_correct":first==gold[0],
                    })
        check(original_forwards==168
              and len(records)==84
              and {r["source_id"] for r in records}==
                  {r["id"] for r in fresh_train},
              "genuine physical original43M 84 previous LEGAL TRAIN*2; no DEV")
        check(validate_onpolicy(records,native_embedding),
              "Z43 baseline 77 first/49 eligible exact source parity")
        # Precommitted ONLY 49 old legal TRAIN examples satisfy OWN
        # first correct + target-blind original-native gold BPE2 in top64.
        # Previous Z45 cross-fold model is not re-tuned on this new DEV.
        fitted={}
        fitted_immutable={}
        fit_meta={}
        for mode,wrong in (("HUMAN_OWNFIRST_TRAIN_ONLY",False),
                           ("WRONG17_OWNFIRST_TRAIN_CONTROL",True)):
            shadow,meta=fit_all_old84(records,native_embedding,wrong=wrong)
            fitted[mode]=shadow
            fit_meta[mode]=meta
            fitted_immutable[mode]={
                key:val.detach().cpu().clone()
                for key,val in shadow.state_dict().items()}
        observed=[]
        dev_forwards=0
        # Exactly one precommitted true autonomous DEV evaluation, and all
        # three modes share the EXACT original S142 BPE1 and BPE2 forwards.
        with torch.no_grad():
            for row in new_dev:
                gold=row["answer_native_with_eos"]
                with scoped_exact_native_gain(child.language_head,.25) as called:
                    first_logits=real_joint_native(
                        model,sources,bridge,full,expansion,child,
                        row["prefix"],row["prompt_features"],contextual=True)
                    check(first_logits.shape==(1,NATIVE_VOCAB)
                          and full.last_forward_coverage==tuple(range(12)),
                          "DEV true original S142 43M FIRST fullnative BPE")
                    first=int(first_logits[0].argmax().item())
                    native_second=None
                    second_h=None
                    if first!=2:
                        with capture_existing_full43m_hidden(model) as h:
                            second_logits=real_joint_native(
                                model,sources,bridge,full,expansion,child,
                                row["prefix"]+[first],
                                row["prompt_features"],contextual=True)
                        check(second_logits.shape==(1,NATIVE_VOCAB)
                              and len(h)==1 and h[0].shape==(1,HIDDEN)
                              and full.last_forward_coverage==tuple(range(12)),
                              "DEV true original OWN-prefix SECOND fullnative BPE")
                        native_second=second_logits.detach().float().clone()
                        second_h=h[0].detach().float().clone()
                    per_mode={}
                    for mode in Z47_MODES:
                        tokens=[first]
                        intervention={"alpha":0.,"entropy_delta_bits":0.,
                                      "changed":False,
                                      "candidate_original_top64":True}
                        if native_second is not None:
                            ranker=fitted.get(mode)
                            chosen,intervention=pick_second(
                                native_second,second_h,first,
                                native_embedding,ranker)
                            tokens.append(chosen)
                            # Third native token necessarily reads new
                            # original S142 43M forward with that mode's
                            # naturally OWN selected second native BPE.
                            if chosen!=2:
                                third=real_joint_native(
                                    model,sources,bridge,full,expansion,child,
                                    row["prefix"]+tokens,
                                    row["prompt_features"],contextual=True)
                                check(third.shape==(1,NATIVE_VOCAB)
                                      and full.last_forward_coverage==tuple(range(12)),
                                      "DEV original43M third always uses true own mode prefix")
                                tokens.append(int(third[0].argmax().item()))
                        first_correct=bool(tokens[0]==gold[0])
                        correct2=bool(len(tokens)>=2 and tokens[:2]==gold[:2])
                        correct3=bool(len(tokens)>=3 and tokens[:3]==gold[:3])
                        per_mode[mode]={
                            "native_first3_ids":tokens,
                            "correct_first":first_correct,
                            "correct_first2":correct2,
                            "correct_first3":correct3,
                            "premature_EOS_first3":bool(2 in tokens[:3]),
                            "second":intervention
                        }
                    expected=1+int(native_second is not None)+sum(
                        len(per_mode[m]["native_first3_ids"])>=3
                        for m in Z47_MODES)
                    check(called[0]==expected and 1<=expected<=5,
                          "actual original43M paid 1 shared BPE1 + 1 BPE2 + up to three third BPE")
                    dev_forwards+=called[0]
                    candidate_coverage=False
                    if native_second is not None:
                        rank=int((native_second[0]>native_second[0,gold[1]]).sum().item())+1
                        candidate_coverage=rank<=64
                    observed.append({
                        "source_id":row["id"],"kind":row["kind"],
                        "prompt_SHA256":row["prompt_sha256"],
                        "first_correct":first==gold[0],
                        "original_second_gold_in_original_TOP64":candidate_coverage,
                        "results":per_mode,
                        "paid_original43M_native_forwards":called[0],
                        "new_Z46_internal_DEV_now_SPENT":True,
                        "new_DEV_optimizer_steps":0,
                    })
        check(len(observed)==28 and
              dev_forwards==sum(r["paid_original43M_native_forwards"] for r in observed)
              and 28<=dev_forwards<=140
              and original_forwards==168,
              "one new 28 DEV original43M true onpolicy first3 per arm")
        metrics=score_newdev(observed)
        for mode,shadow in fitted.items():
            check(shadow.output_gate==0 and shadow.candidate_accepted is False
                  and all(torch.equal(shadow.state_dict()[k].detach().cpu(),v)
                          for k,v in fitted_immutable[mode].items()),
                  "no learned shadow DEV changes/optimizer/retuning")

    finally:
        with torch.no_grad():
            for k,v in full_gt_donor.items():
                full.state_dict()[k].copy_(v)
        for p in full.parameters(): p.requires_grad_(False)
    check(old_original_param_digest(model)==original_digest
          and all(torch.equal(full.state_dict()[k].detach().cpu(),v)
                  for k,v in full_gt_donor.items())
          and all(torch.equal(fit.state_dict()[k].detach().cpu(),v)
                  for tag,fit in (("A",child_a),("B",child_b))
                  for k,v in child_frozen[tag].items()),
          "S142 real original43M, all12GT and Z10 A/B fitted children bitwise unchanged")
    check(metrics["candidate_accepted"] is False
          and metrics["output_gate"]==0
          and metrics["NEW_DEV28_becomes_SPENT_once_scored"]
          and original_forwards==168,
          "Z46 source-only precommitted metrics reported NO promotion")
    report={
        "stage":STAGE,
        "Z46_NEW28_before_physical_SHA256":prereg["brand_NEW_28_internal_DEV_SHA256"],
        "previous_Z40_LEGAL_TRAIN84_SHA256":fresh["TRAIN_SHA256"],
        "frozen_prior_Z41_teacher_report_SHA256":sha(z41_report),
        "frozen_prior_Z43_onpolicy_report_SHA256":sha(z43_report),
        "original_S142_43M_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "trained_children_Z10_fitted_parent_SHA256":sha(prev_path),
        "true_original43M_prior_TRAIN_only_forwards":original_forwards,
        "true_original43M_new_DEV28_native_forwards":dev_forwards,
        "true_original43M_total_physically_paid_forwards":original_forwards+dev_forwards,
        "match_HUMAN_vs_WRONG17_training_controls":fit_meta,
        "new_internal_DEV28_posthoc_autonomous_FIRST3_with_original_BPE3":metrics,
        "new_internal_DEV28_per_prompt_ONLY":observed,
        "new_DEV28_now_SPENT_not_external_independent_blind":True,
        "no_old_SPENT_DEV_teacher_forcing_optimizer_or_scoring":True,
        "original43M_GT12_Core_NRA3_NPA1_Z10_and_all_shadow_weights_post_DEV_unchanged":True,
        "no_training_of_original_or_616J7_EVAL128_GOLD_bpp_CB59_CPU_certification":True,
        "no_MAIN_LIVE_Render_Supabase_APK_Jessica_changes":True,
        "rough_uncontrolled_CPU_elapsed_seconds":round(time.monotonic()-started,3),
        "candidate_accepted":False,"output_gate":0,
        "decision":"PREREG_Z46_FRESH28_SCORED_ONCE_GATE0_NO_PROMOTION"
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
                                 indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("S119Q2I8Z47_REAL43M_NEW28_TRUE_FIRST3_NATIVE_HUMAN_VS_WRONG "+json.dumps({
        "new_DEV_now_SPENT":28,
        "real_paid_original43M_native_forwards":original_forwards+dev_forwards,
        "modes":metrics["modes"],
        "precommitted_diagnostic_gates":metrics["precommitted_diagnostic_thresholds"],
        "all_nonpromoting_gates_pass":metrics["all_precommitted_nonpromoting_thresholds_pass"],
        "candidate_accepted":False,"output_gate":0,
    },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8f_report","q2i8g_dir","q2i8k_dir","q2i8t_dir","q2i8w_dir","z10_dir","z41_report","z43_report","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,
        a.q2i8f_report,a.q2i8g_dir,a.q2i8k_dir,a.q2i8t_dir,a.q2i8w_dir,a.z10_dir,a.z41_report,a.z43_report,a.out)

if __name__=="__main__":
    main()
