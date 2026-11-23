"""S119q2i8z10: TRUE original S142 43M 12GT/Core/NRA3/NPA1/native49k
matched A/B 27 new Z10 TRAIN joint updates from SAME completed W
early23×3 FP16 neural 421793 + fitted 12 Green Tunnel gates.
A: full native human all-BPE CE (second+third x3), normal EOS.
B: same plus differentiable 2.0*tanh(existing Core onpath gate)^2
once per TRAIN answer ONLY, no inference-time CORE mask or change.
New14 author Z10 internal synthetic DEV PRECOMMITTED and never in
optimizer, genuine real 43M OWN greedy 40 BPE and separate teacherforced
human previous gold BPE2 ranks AFTER training; 6 old spent DEV.
The existing Core component is NOT removed and is still learnable
alongside NRA3/NPA1/GT12/rank4 native readout 368051 current params.
No LIVE/main/Render/Supabase/APK/Jessica, no J7/EVAL128/GOLD.
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


from research.calibri_nra3.s119q2i8z12_parallel_native_new14_source import (
    STAGE as NEW_SOURCE_STAGE,ORIGIN as NEW_SOURCE_ORIGIN,
    rows as fresh_new_rows,verify as fresh_new_verify,
)
from research.calibri_nra3.s119q2i8z12_native_parallel_branch_contract import (
    native_lookahead_second_third,WIDTH,THIRD_WIDTH,HORIZON,
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

STAGE="CALIBRI_S119Q2I8Z13_TRUE43M_NATIVE49152_8X2_LOOKAHEAD_FRESH14_OWN"
REPORT="S119Q2I8Z13_TRUE43M_PARALLEL_TOP8XTOP2_VS_GREEDY_FRESH14_G0.json"
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
          and fresh["fresh_DEV_rows"]==14
          and fresh["BRANCH_WIDTH"]==8
          and fresh["THIRD_WIDTH"]==2
          and fresh["HORIZON"]==2
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
    freshdev=[]
    for authored in fresh_new_rows("DEV"):
        check(authored["split"]=="DEV"
              and authored["origin"]==NEW_SOURCE_ORIGIN
              and authored["id"].startswith("S119Q2I8Z12_"),
              "only previously unseen authored Z12 internal DEV")
        row=dict(authored)
        row["origin"]="original_newly_authored_examples_no_third_party_corpus"
        example=verified_native_answer(tok,row)
        check(example["split"]=="DEV"
              and tok.decode(example["answer_native_without_eos"],
                             skip_special_tokens=True)==authored["completion"]
              and 4<=len(example["answer_native_with_eos"])<=40
              and len(example["prefix"])+39<160,
              "exact native original tokenizer target, EOS, context limits")
        freshdev.append(example)
    check(len(freshdev)==14 and len({x["prompt_sha256"] for x in freshdev})==14,
          "14 unique original new DEV SHA records")
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
          "same exact completed W real original43M x3 CE fitted FP16/GT12 for Z10 matched arms")
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
    training={};gates={};saved={};own={};gold_ranks={}
    out.mkdir(parents=True)
    try:
        for label,child in (
            (ARMS[0],ce),
            (ARMS[1],margin),
        ):
            set_gt_route(full,"GT_RECURRENT",gt)
            training[label]=train_joint_true43m_context_top8_ab(
                model,sources,bridge,full,expansion,child,train,tok,
                condition="GT_RECURRENT",continuation_weight=1.0,
                expected_total=TRAIN_ROWS,expected_new=NEW_TRAIN,
                new_id_prefix="S119Q2I8Z9_",rounds=1,
                teacher_135m=None,teacher_KL_weight=0.0,
                objective=label)
            gates[label]=full.gates.detach().clone()
            check(old_original_param_digest(model)==orig_digest
                  and training[label]["real_all_neural_and_gt_Adam_updates"]==27
                  and training[label]["trainable_parameter_count"]==TOTAL_EXPECTED_TRAINABLE
                  and training[label]["second_BPE_context_trainonly_margin_count"]==27,
                  "27 actual one-pass true original43M 12GT Core/NRA3/NPA1+49k native all-joint Adam")
        check(training[ARMS[0]]["trained_original_decoder_calls"]==
              training[ARMS[1]]["trained_original_decoder_calls"]
              and training[ARMS[0]]["context_top8_aux_weight_if_B"]==0.
              and training[ARMS[1]]["context_top8_aux_weight_if_B"]==.20
              and training[ARMS[0]]["train_only_true_SECOND_THIRD_native_gold_CE_weight"]==
                  training[ARMS[1]]["train_only_true_SECOND_THIRD_native_gold_CE_weight"]==3.,
              "matched real 43M 27 TRAIN, same native x3 human second third CE, only B existing Core gate L2")
        saved={
            "CE_EARLY23_X3_FP16":persist_half(ce,out/CE_FP16),
            "CONTEXT_TOP8_BPE2_FP16":persist_half(margin,out/CONTEXT_FP16),
            "CE_EARLY23_X3_GT12":safe_gt_gate_sidecar(
                out/CE_GATES,gates[ARMS[0]],"GT_RECURRENT"),
            "CONTEXT_TOP8_BPE2_GT12":safe_gt_gate_sidecar(
                out/CONTEXT_GATES,gates[ARMS[1]],"GT_RECURRENT"),
        }
        arms={
            "W_PREVIOUS_EARLY23_X3_FROZEN":(original_fitted,gt,.25),
            "Z10_FULLJOINT_EARLY23_X3_CE":(ce,gates[ARMS[0]],.25),
            "Z10_FULLJOINT_EARLY23_X3_CONTEXT_TOP8_BPE2":(
                margin,gates[ARMS[1]],.25),
        }
        for label,(child,gt12,gain) in arms.items():
            own[label]={
                "NEW_DEV":eval_own_gain(
                    tok,model,sources,bridge,full,expansion,
                    child,gt12,newdev,gain=gain,arm=label),
                "OLD_DEV":eval_own_gain(
                    tok,model,sources,bridge,full,expansion,
                    child,gt12,olddev,gain=gain,arm=label),
            }
        # Gold-PREVIOUS BPE2 diagnostics are AFTER both optimizations
        # and separate from genuine self-native 40-token OWN generation.
        from research.calibri_nra3.s119q2i8n_original43m_early23_goldrank import real_native_rank
        from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native
        for label,(child,gt12,gain) in arms.items():
            set_gt_route(full,"GT_RECURRENT",gt12)
            records=[]
            for ex in newdev:
                target=int(ex["answer_native_with_eos"][1])
                check(ex["split"]=="DEV"
                      and ex["id"].startswith("S119Q2I8Z9_")
                      and len(ex["answer_native_with_eos"])>=4
                      and target!=tok.eos_token_id,
                      "Z9 author DEV gold-second human BPE only; gradients complete")
                prefix=ex["prefix"]+ex["answer_native_with_eos"][:1]
                with scoped_exact_native_gain(child.language_head,.25) as witness:
                    logits=real_joint_native(
                        model,sources,bridge,full,expansion,
                        child,prefix,ex["prompt_features"],contextual=True)
                check(witness[0]==1 and logits.shape==(1,NATIVE_VOCAB)
                      and full.last_forward_coverage==tuple(range(12)),
                      "one true original43M 12GT all regions native49152 gold-second source only")
                stat=real_native_rank(logits,target)
                records.append({
                    "source_id":ex["id"],"source_prompt_sha256":ex["prompt_sha256"],
                    "correct_gold_second_native_rank_full49152":stat[
                        "correct_gold_native_rank_full49152"],
                    "gold_second_TOP1":stat["gold_native_top1"],
                    "gold_second_TOP5":stat["correct_gold_rank_top5"],
                    "gold_second_NLL_nats":-stat["gold_log_probability_native_49152"],
                    "greedy_wrong_native_second_is_EOS":stat["greedy_original_native_id"]==2,
                    "given_gold_first_BPE_ONLY_for_readonly_diagnostic":True,
                })
            gold_ranks[label]={
                "new14_now_spent_internal_Z10_author_DEV":14,
                "correct_gold_second_top1":sum(x["gold_second_TOP1"] for x in records),
                "correct_gold_second_top5":sum(x["gold_second_TOP5"] for x in records),
                "mean_correct_gold_second_rank_full49152":
                    sum(x["correct_gold_second_native_rank_full49152"] for x in records)/14,
                "mean_gold_second_NLL_nats":sum(x["gold_second_NLL_nats"] for x in records)/14,
                "wrong_second_is_EOS":sum(x["greedy_wrong_native_second_is_EOS"] for x in records),
                "per_source_read_only_teacherforced_previous_gold_BPE":records,
            }
    finally:
        with torch.no_grad():
            for k,v in old_gt.items():full.state_dict()[k].copy_(v)
        for p in full.parameters():p.requires_grad_(False)
    check(old_original_param_digest(model)==orig_digest
          and all(torch.equal(full.state_dict()[k].detach().cpu(),v)
                  for k,v in old_gt.items())
          and all(torch.equal(original_fitted.state_dict()[k].detach().cpu(),v)
                  for k,v in initial_parent.items()),
          "original frozen S142 43M+GT12 donor, prior trained W 421793 FP16 source unchanged")
    training_calls=sum(v["trained_original_decoder_calls"] for v in training.values())
    own_calls=sum(x["scores"]["real_original43M_OWN_forwards"]
                  for arm in own.values() for x in arm.values())
    gold_calls=sum(r["new14_now_spent_internal_Z10_author_DEV"]
                   for r in gold_ranks.values())
    check(training_calls>TRAIN_ROWS*3*2 and own_calls>60
          and gold_calls==NEW_HELDOUT_DEV*3==42,
          "true source original43M 12GT 49k every TRAIN+OWN+gold BPE2 forensic")
    report={
        "stage":STAGE,
        "original_S142_43M_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "completed_W_fulljoint_43M_early23X3_source_report_sha256":
            sha(q2i8w_dir/W_REPORT),
        "completed_W_421793_fp16_source_sha256":sha(q2i8w_dir/W_EARLY_FP16),
        "completed_W_12GT_fit_source_sha256":sha(q2i8w_dir/W_EARLY_GATES),
        "new_Z9_21_TRAIN_sha256":z9_source["Z9_21_TRAIN_sha256"],
        "new_Z9_14_DEV_sha256":z9_source["Z9_14_DEV_sha256"],
        "matched_27_true43M_alljoint_neural_and_GT12_A_B_from_same_W_FP16":True,
        "second_gold_BPE2_trainonly_contextual_top8_margin_weight":CONTEXT_TOP8_MARGIN_WEIGHT,
        "same_human_BPE2andBPE3_CE3x_in_both_arms":True,
        "no_inference_manual_CORE_gate_overrides":True,
        "neural_all421793_plus_separate12GT_FP16_saved":saved,
        "true43M_full_GT12_Core_NRA3_NPA1_native49152_27TRAIN_AB":training,
        "true43M_SELF_OWN_newZ10_14DEV_plus_OLD6_each_three_states":own,
        "true43M_gold_SECOND_BPE_original49k_rank_only_POSTHOC":gold_ranks,
        "true43M_49k_full_native_TRAIN_forwards":training_calls,
        "true43M_49k_full_native_OWN_forwards":own_calls,
        "true43M_49k_gold_second_teacherforced_only_forwards":gold_calls,
        "frozen_original43M_S115_protected_W_parent_unchanged":True,
        "no_J7_616_EVAL128_GOLD_allin_bpp_CPU_tested":True,
        "no_main_LIVE_Render_Supabase_APK_Jessica_chat":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"RESEARCH_ONLY_MATCHED_TRUE43M_CONTEXT_TOP8_BPE2_TRAINONLY_VS_CE_NO_PROMOTION",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
                           sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z10_REAL43M_CONTEXT_TOP8_TRAINONLY_AB "+json.dumps({
        "true_original43M_train_forwards":training_calls,
        "true_original43M_OWN_forwards":own_calls,
        "true_original43M_posthoc_gold_second_forwards":gold_calls,
        "full_joint_trainable_params":TOTAL_EXPECTED_TRAINABLE,
        "TRAIN_only_learned_CORE_tanh_gate_per_arm":{
            k:{
                "before":v["learned_existing_CORE_component_tanh_GATE_before_27TRAIN"],
                "after":v["learned_existing_CORE_component_tanh_GATE_after_27TRAIN"],
                "lambda":v["context_top8_aux_weight_if_B"],
            } for k,v in training.items()},
        "new_Z9_14_actual_OWN":{
            k:v["NEW_DEV"]["scores"] for k,v in own.items()},
        "new_Z9_14_gold_second_posthoc_rank":{
            k:{j:v[j] for j in (
                "correct_gold_second_top1","correct_gold_second_top5",
                "mean_correct_gold_second_rank_full49152",
                "mean_gold_second_NLL_nats","wrong_second_is_EOS")}
            for k,v in gold_ranks.items()},
        "candidate_accepted":False,
    },ensure_ascii=False,sort_keys=True),flush=True)


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
