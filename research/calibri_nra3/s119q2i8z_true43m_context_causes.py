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
    # Precommit completed physical X original43M neural child, original 12GT
    # and every synthetic X14 prompt was already scored in #960.
    x_path=q2i8x_dir/X_REPORT
    x_report=json.loads(x_path.read_text(encoding="utf8"))
    check(x_report["stage"]==X_STAGE and
          x_report["candidate_accepted"] is False and
          x_report["output_gate"]==0 and
          x_report["completed_W_original43M_fulljoint_report_sha256"]==
              sha(q2i8w_dir/W_REPORT) and
          x_report["completed_W_421793_full_neural_FP16_sha256"]==
              sha(q2i8w_dir/W_EARLY_FP16) and
          x_report["completed_W_12GT_trained_FP16_sha256"]==
              sha(q2i8w_dir/W_EARLY_GATES) and
          x_report["new_X21_TRAIN_source_sha256"]==x_source["X21_TRAIN_sha256"] and
          x_report["new_X14_DEV_source_sha256"]==x_source["X14_DEV_sha256"] and
          x_report["original_S142_43M_sha256"]==sha(s142) and
          x_report["original_S93_GT_sha256"]==sha(s93) and
          x_report["matched_true43M_all_joint_368051_A_B_from_identical_W_FP16_and_GT12"] is True and
          x_report["actual_original_S142_43M_12GT_native_49152_training_forwards"]==1350 and
          x_report["actual_original_S142_43M_12GT_native_49152_OWN_forwards"]==1355 and
          x_report["prior_protected_S115_53751_unchanged"] is True,
          "exact completed research X full original43M/GT12 actual parent and X synthetic DEV SHA")
    meta=x_report["persisted_A_B_original43M_FULL_neural_and_12GT_FP16_gate0"]
    check(set(meta)=={"CE_EARLY23_X3_joint_FP16",
                       "EARLY23_X3_HARDNEG_joint_FP16",
                       "CE_EARLY23_X3_GT12",
                       "EARLY23_X3_HARDNEG_GT12"},
          "exact A/B original trained FP16 421793 and 12 GT scalars X gate0")
    ce=full_saved_child(q2i8x_dir/X_CE_FP16,
                         meta["CE_EARLY23_X3_joint_FP16"],original_fitted)
    hard=full_saved_child(q2i8x_dir/X_HARDNEG_FP16,
                           meta["EARLY23_X3_HARDNEG_joint_FP16"],original_fitted)
    gt_ce=full_saved_gt(q2i8x_dir/X_CE_GATES,meta["CE_EARLY23_X3_GT12"])
    gt_hard=full_saved_gt(q2i8x_dir/X_HARDNEG_GATES,
                           meta["EARLY23_X3_HARDNEG_GT12"])
    check(gt.shape==gt_ce.shape==gt_hard.shape==(12,) and
          all(child.output_gate==0 and child.candidate_accepted is False
              for child in (original_fitted,ce,hard)),
          "3 exact physical research full FP16 native models and GT12 sidecars gate0")
    arms={
        MODELS[0]:(original_fitted,gt),
        MODELS[1]:(ce,gt_ce),
        MODELS[2]:(hard,gt_hard),
    }
    scored=x_report["fresh_14X_INTERNAL_DEV_plus6_old_FULL_OWN_native_autoregressive"]
    check(set(arms)==set(MODELS) and
          len(newdev)==NEW_HELDOUT_DEV and
          set(z["id"] for z in newdev)==set(
              r["source_id"] for r in scored[MODELS[1]]["NEW_DEV"]["OWN_samples"]),
          "only same 14 X DEV already scored as own native BPE in successful 43M X")
    model,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",ce)
    model.eval()
    initial_original_digest=old_original_param_digest(model)
    full_start={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    children_start={
        name:{k:v.detach().cpu().clone() for k,v in child.state_dict().items()}
        for name,(child,_) in arms.items()
    }
    start_regions={
        name:child.child.child.midlayer_gates.detach().clone()
        for name,(child,_) in arms.items()
    }
    native_full_gt_gate_contract(full,bridge)
    check(all(v.shape==(3,) and bool(torch.isfinite(v).all())
              for v in start_regions.values()),
          "existing CORE NRA3 NPA1 midlayer original three gates present")
    check(all(not p.requires_grad for name,p in model.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "TRUE original 43M frozen and research-only X native child loaded")
    records=[]
    try:
        for arm,(child,saved_gates) in arms.items():
            check(saved_gates.shape==(12,) and
                  bool(torch.isfinite(saved_gates).all()),
                  "original GT12 full saved physical gates loaded")
            for example in newdev:
                gold=example["answer_native_with_eos"]
                target=int(gold[1])
                check(example["split"]=="DEV" and
                      example["id"].startswith("S119Q2I8X_") and
                      len(gold)>=4 and target!=tok.eos_token_id and
                      len(example["prefix"])+1<160,
                      "same already spent 14 X synthetic RU DEV gold second position only")
                prefix=example["prefix"]+gold[:1]
                baseline=None
                for label in CONDITIONS:
                    route,gt_scale,gain,mask=config(label)
                    set_gt_route(full,route,saved_gates*gt_scale)
                    with temporary_gate_mask(child.child.child,mask):
                        with scoped_exact_native_gain(child.language_head,float(gain)) as counted:
                            logits=real_joint_native(
                                model,sources,bridge,full,expansion,child,prefix,
                                example["prompt_features"],contextual=True)
                    check(counted[0]==1 and
                          full.last_forward_coverage==tuple(range(12)) and
                          not full._active and full._carry is None and
                          logits.shape==(1,NATIVE_VOCAB),
                          "TRUE one physical 43M 12GT all-region 49152 native BPE pass")
                    vec=logits.detach().float().cpu().reshape(-1)
                    check(bool(torch.isfinite(vec).all()),
                          "actual 49,152 native logits finite without posthoc masking")
                    result=real_native_rank(logits,target)
                    if label==FULL:
                        baseline=vec.clone()
                    check(baseline is not None and baseline.shape==(NATIVE_VOCAB,),
                          "same-model and same-X-synthetic-prompt full baseline precedes toggles")
                    linf=float((vec-baseline).abs().max().item())
                    records.append({
                        "model":arm,
                        "source_id":example["id"],
                        "source_prompt_SHA256":example["prompt_sha256"],
                        "condition":label,
                        "GT_route":route,
                        "GT_gate_scale":gt_scale,
                        "native_rank4_head_gain":gain,
                        "predeclared_Core_NRA3_NPA1_gate_mask":list(mask),
                        "gold_BPE_target_native_id":target,
                        "gold_49152_BPE_rank":result[
                            "correct_gold_native_rank_full49152"],
                        "gold_49152_BPE_NLL":-result[
                            "gold_log_probability_native_49152"],
                        "gold_top1":result["gold_native_top1"],
                        "gold_top5":result["correct_gold_rank_top5"],
                        "predicted_BPE_id":result["greedy_original_native_id"],
                        "winner_is_EOS":result["greedy_original_native_id"]==int(tok.eos_token_id),
                        "original_full49152_logit_Linf_delta_vs_FULL":linf,
                        "source_is_ALREADY_SPENT_synthetic_DEV":True,
                        "teacher_forced_PREVIOUS_gold_BPE_not_own":True,
                    })
                    del logits,vec,result
    finally:
        with torch.no_grad():
            for k,v in full_start.items():
                full.state_dict()[k].copy_(v)
        for p in full.parameters():
            p.requires_grad_(False)
    check(old_original_param_digest(model)==initial_original_digest and
          all(torch.equal(full.state_dict()[k].detach().cpu(),v)
              for k,v in full_start.items()) and
          all(torch.equal(child.state_dict()[k].detach().cpu(),value)
              for name,(child,_) in arms.items()
              for k,value in children_start[name].items()) and
          all(torch.equal(start_regions[name],child.child.child.midlayer_gates.detach())
              for name,(child,_) in arms.items()),
          "all S142 43M, full GT12 original donor and W/X complete trained child params bitwise restored")
    check(len(records)==EXPECTED_REAL_ORIGINAL_43M_FORWARDS==336,
          "exact 336 real original 43M 12GT native49152 second-BPE forwards")
    aggregate_results=aggregate(records)
    out.mkdir(parents=True)
    report={
        "stage":STAGE,
        "completed_X_real_original43M_report_SHA256":sha(x_path),
        "completed_W_original43M_parent_report_SHA256":sha(q2i8w_dir/W_REPORT),
        "new_X14_now_SPENT_source_SHA256":x_source["X14_DEV_sha256"],
        "original_S142_43M_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "original_frozen43M_and_all_three_research_FP16_model_and_GT12_weights_preserved":True,
        "three_completed_original43M_FULL_421793FP16_neural_and_real12GT_gates_verified":True,
        "original43M_GT12_layer_coverage_all12_each_forward":True,
        "real_original43M_full_native49152_gold_second_prefix_actual_physical_forwards":len(records),
        "registered_8_component_and_native_readout_conditions":list(CONDITIONS),
        "X14_synthetic_AUTHOR_DEV_already_spent_teacherforced_gold_PREVIOUS_BPE_NOT_OWN":True,
        "gain0_only_removes_learned_output_delta_NOT_internal_onpath_regions":True,
        "original_Core_NRA3_NPA1_existing_three_gates_reversible":True,
        "full_native49152_original43M_context_CE_rank_top1_and_ablation_deltas":aggregate_results,
        "per_source_14x3x8_actual_original43M_logit_rank_and_ce_records":records,
        "NO_optimizer_training_backward_or_new_135M_teacher":True,
        "NO_protected_616J7_EVAL128_GOLD_bpp_CPU_verified":True,
        "NO_LIVE_Render_Supabase_APK_main_Jessica_chat":True,
        "candidate_accepted":False,
        "output_gate":0,
        "decision":"CAUSAL_GT12_REGION_NATIVE_HEAD_SECOND_BPE_DIAG_ONLY_NO_PROMOTION",
    }
    (out/REPORT).write_text(json.dumps(
        report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8Z_TRUE43M_336_GOLD_SECOND_NATIVE_CONTEXT_CAUSAL "+json.dumps({
        "original43M_native49152_actual_forwards":len(records),
        "actual_three_trained_FP16_and_8_reversible_causal_controls":{
            name:{control:{k:vals[k] for k in (
                "gold_second_BPE_top1","gold_second_BPE_top5",
                "mean_gold_second_native_rank","mean_gold_second_NLL_nats",
                "wrong_gold_second_native_EOS_pred",
                "top1_winner_CHANGED_vs_FULL_same_source",
                "mean_abs_native_logit_Linf_CHANGE_vs_FULL")}
                for control,vals in state.items()}
            for name,state in aggregate_results.items()},
        "candidate_accepted":False,
    },sort_keys=True,ensure_ascii=False),flush=True)


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
