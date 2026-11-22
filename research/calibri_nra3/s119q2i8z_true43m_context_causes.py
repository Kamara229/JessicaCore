"""S119q2i8z causal read-only true original S142 43M 12GT Core NRA3 NPA1 and native49152 rank4 output context.

Three saved trained research FP16 physical models: W early23 parent, X CE, X strongest-wrong nonEOS.
Eight predeclared toggles on same now-SPENT X14 internal synthetic DEV at gold-previous-first-native BPE.
Exactly 336 real full 49152 native 43M forwards, no gradient/optimizer, irreversible model changes, sealed GOLD/J7/EVAL or live.
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


from research.calibri_nra3.s119q2i8x_new_ru_train_dev_source import (
    STAGE as X_SOURCE_STAGE,ORIGIN as X_ORIGIN,
    rows as x_rows,verify as x_verify,
)
from research.calibri_nra3.s119q2i8x_true43m_joint_native_hardneg_train import EARLY23_MULTIPLIER,HARDNEG_LAMBDA,HEAD_GAIN
from research.calibri_nra3.s119q2i8x_true43m_fulljoint_hardneg_AB import STAGE as X_STAGE,REPORT as X_REPORT,CE_FP16 as X_CE_FP16,HARDNEG_FP16 as X_HARDNEG_FP16,CE_GATES as X_CE_GATES,HARDNEG_GATES as X_HARDNEG_GATES
from research.calibri_nra3.s119q2i8z_context_ablation_contract import STAGE,CONDITIONS,MODELS,FULL,config,aggregate,EXPECTED_REAL_ORIGINAL_43M_FORWARDS
from research.calibri_nra3.s119p_real43m_onpath_component_ablation import temporary_gate_mask
from research.calibri_nra3.s119q2i8n_original43m_early23_goldrank import real_native_rank
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native
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

REPORT="S119Q2I8Z_TRUE43M_GOLD_SECOND_NATIVE49K_CONTEXT_CAUSAL_336_PROBES_G0.json"
NEW_HELDOUT_DEV=14
OLD_SPENT_DEV=6
TRAIN_ROWS=27
NEW_TRAIN=21
NATIVE_VOCAB=49152
SEED=119162


def pack_x_author_native(tok,row,*,split):
    check(row["split"]==split
          and split in ("TRAIN","DEV")
          and row["origin"]==X_ORIGIN
          and row["id"].startswith("S119Q2I8X_")
          and row["kind"] in KINDS,
          "only NEW X 21 author TRAIN and new14 author internal synthetic DEV")
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
        two=bool(len(ids)>=2 and
                 ids[:2]==item["answer_native_with_eos"][:2])
        three=bool(len(ids)>=3 and
                   ids[:3]==item["answer_native_with_eos"][:3])
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
        q2i8d_dir,q2i8f_report,q2i8g_dir,q2i8k_dir,q2i8t_dir,q2i8w_dir,q2i8x_dir,out):
    check(not out.exists(),"do not overwrite old research model or published result")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=source_verify()
    x_source=x_verify()
    check(source["stage"]==SOURCE_STAGE
          and source["new_TRAIN_only"]==70
          and x_source["stage"]==X_SOURCE_STAGE
          and x_source["fresh_X_TRAIN_rows"]==NEW_TRAIN
          and x_source["fresh_X_DEV_rows"]==NEW_HELDOUT_DEV
          and HEAD_GAIN==.25 and EARLY23_MULTIPLIER==3.0
          and HARDNEG_LAMBDA==.5,
          "21 new X 7-domain author TRAIN and 14 new X author DEV SHA precommitted")
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
    chosen=x_rows("TRAIN")
    train_new=[pack_x_author_native(tok,x,split="TRAIN") for x in chosen]
    train_old=[verified_native_answer(tok,x) for x in old_rehearsal]
    train=train_new+train_old
    newdev=[pack_x_author_native(tok,x,split="DEV") for x in x_rows("DEV")]
    olddev=[verified_native_answer(tok,x) for x in old_dev]
    for x in olddev:
        x["tokenizer"]=tok
        x["eos_native_id"]=int(tok.eos_token_id)
    check(len(train)==TRAIN_ROWS and len(newdev)==NEW_HELDOUT_DEV
          and len(olddev)==OLD_SPENT_DEV and
          all(x["split"]=="TRAIN" for x in train) and
          all(x["split"]=="DEV" for x in newdev+olddev)
          and Counter(x["kind"] for x in train_new)=={k:3 for k in KINDS}
          and {x["prompt_sha256"] for x in train}.isdisjoint(
              x["prompt_sha256"] for x in newdev+olddev)
          and max(len(x["answer_native_with_eos"]) for x in train)<=40
          and max(len(x["answer_native_with_eos"]) for x in newdev+olddev)<=40
          and max(len(x["prefix"])+39 for x in newdev+olddev)<160,
          "21 new author TRAIN+six old vs 14 fresh author DEV+six OLD no training leakage")
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
          and w_report["candidate_accepted"] is False
          and w_report["output_gate"]==0
          and w_report["completed_T_FULLJOINT_CE_REPORT_SHA256"]==
              sha(q2i8t_dir/T_REPORT)
          and w_report["completed_T_FULLJOINT_CE_neural_FP16_SHA256"]==
              sha(q2i8t_dir/T_CE_FP16)
          and w_report["completed_T_FULLJOINT_CE_GT12_FP16_SHA256"]==
              sha(q2i8t_dir/T_CE_GATES)
          and w_report["original_S142_43M_sha256"]==sha(s142)
          and w_report["original_S93_GT_sha256"]==sha(s93)
          and w_report["match_full43M_27TRAIN_A_B_from_identical_T_CE_FP16_state"] is True
          and w_report["predeclared_early23_gold_nonEOS_BPE2BPE3_multiplier"]==3.,
          "completed SAME original true S142+12GT fulljoint W CE early23 x3 research parent")
    wmeta=w_report["saved_full_FP16_original_joint_neural_plus_separate_GT12"]
    check(set(wmeta)=={"CE_joint_FP16","EARLY23_X3_joint_FP16",
                       "CE_GT12","EARLY23_X3_GT12"},
          "only original completed full physical FP16 W x3 fitted child and GT12")
    original_fitted=full_saved_child(
        q2i8w_dir/W_EARLY_FP16,wmeta["EARLY23_X3_joint_FP16"],t_fitted)
    gt=full_saved_gt(
        q2i8w_dir/W_EARLY_GATES,wmeta["EARLY23_X3_GT12"])
    check(gated.shape==gateg.shape==k_gt.shape==t_gt.shape==gt.shape==(12,)
          and original_fitted.output_gate==0
          and original_fitted.candidate_accepted is False,
          "both X arms continue from same W physically saved x3 true43M source")
    # S119q2i8z read-only true original43M models: body injected below.
    # PLACEHOLDER_Z_ACTUAL_FULL_NATIVE_CONTEXT_CAUSAL_STUDY


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8f_report","q2i8g_dir","q2i8k_dir","q2i8t_dir","q2i8w_dir","q2i8x_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,
        a.q2i8f_report,a.q2i8g_dir,a.q2i8k_dir,a.q2i8t_dir,a.q2i8w_dir,a.q2i8x_dir,a.out)

if __name__=="__main__":
    main()
