"""S119q2i8w: physical 43M 12 GreenTunnel joint native 49k language A/B.

START: exact real S119q2i8t FULLJOINT native plain CE FP16 neural
and 12 trained GT scalar gates, NOT earlier K pretrained. Human W
21 new TRAIN +6 old rehearsal, 27 matched all-architecture Adam updates.
A=all native BPE+EOS usual CE, B=early second/third nonEOS human
gold BPE native49k CE weighted3 with exact same model/data optimizer and
ordinary human EOS CE preserved. Predeclared original 14 new RU DEV for
genuine OWN40 BPE and separate gold-prefix second native read-only rank.
S115 original protected and S142 original43M frozen. No new regions,
no EOS mask, no LIVE, no sealed evaluations or promotion.
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


from research.calibri_nra3.s119q2i8w_new_ru_train_dev_source import (
    STAGE as W_SOURCE_STAGE,ORIGIN as W_ORIGIN,
    rows as w_rows,verify as w_verify,
)
from research.calibri_nra3.s119q2i8w_true43m_joint_native_early23_train import (
    train_joint_true43m_early23_ce_ab,EARLY23_MULTIPLIER,HEAD_GAIN,
    STAGE as JOINT_STAGE,
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

STAGE="CALIBRI_S119Q2I8W_TRUE43M_FRESH21TRAIN_FULL_GT12_JOINT_CE_VS_EARLY23X3_NATIVE49K"
REPORT="S119Q2I8W_TRUE43M_FULLJOINT_27TRAIN_STANDARD_CE_VS_SECONDTHIRD_X3_G025.json"
CE_FP16="S119Q2I8W_TRUE43M_FULLJOINT_NATIVE_CE_FP16_G0.safetensors"
EARLY_FP16="S119Q2I8W_TRUE43M_FULLJOINT_NATIVE_EARLY23X3_FP16_G0.safetensors"
CE_GATES="S119Q2I8W_TRUE43M_FULLJOINT_NATIVE_CE_GT12_FP16_G0.safetensors"
EARLY_GATES="S119Q2I8W_TRUE43M_FULLJOINT_NATIVE_EARLY23X3_GT12_FP16_G0.safetensors"
NEW_HELDOUT_DEV=14
OLD_SPENT_DEV=6
TRAIN_ROWS=27
NEW_TRAIN=21
NATIVE_VOCAB=49152
SEED=119160
ARMS=("FULL_JOINT_NATIVE_CE","FULL_JOINT_NATIVE_CE_EARLY23_X3")


def pack_w_author_native(tok,row,*,split):
    check(row["split"]==split
          and split in ("TRAIN","DEV")
          and row["origin"]==W_ORIGIN
          and row["id"].startswith("S119Q2I8W_")
          and row["kind"] in KINDS,
          "only NEW W 21 authored TRAIN / 14 authored unscored DEV native49k")
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
        output.append({
            "first_two_gold_BPE_correct":two,
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
            "first_correct_but_second_wrong":sum(x["first_native_correct_but_wrong_second"] for x in output),
            "real_original43M_OWN_forwards":sum(x["original_full_decoder_calls"] for x in output),
        }
    }


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,q2i7_dir,q2i8a_dir,
        q2i8d_dir,q2i8f_report,q2i8g_dir,q2i8k_dir,q2i8t_dir,out):
    check(not out.exists(),"do not overwrite old research model or published result")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=source_verify()
    w_source=w_verify()
    check(source["stage"]==SOURCE_STAGE
          and source["new_TRAIN_only"]==70
          and w_source["stage"]==W_SOURCE_STAGE
          and w_source["fresh_TRAIN"]==NEW_TRAIN
          and w_source["fresh_internal_DEV"]==NEW_HELDOUT_DEV
          and HEAD_GAIN==.25 and EARLY23_MULTIPLIER==3.0,
          "21 W new pre-registered TRAIN+14 fresh W synthetic internal DEV")
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
    chosen=w_rows("TRAIN")
    train_new=[pack_w_author_native(tok,x,split="TRAIN") for x in chosen]
    train_old=[verified_native_answer(tok,x) for x in old_rehearsal]
    train=train_new+train_old
    newdev=[pack_w_author_native(tok,x,split="DEV") for x in w_rows("DEV")]
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
    original_fitted=full_saved_child(
        q2i8t_dir/T_CE_FP16,tmeta["CE_joint_FP16"],k_fitted)
    gt=full_saved_gt(q2i8t_dir/T_CE_GATES,tmeta["CE_GT12"])
    check(gated.shape==gateg.shape==k_gt.shape==gt.shape==(12,)
          and original_fitted.output_gate==0
          and original_fitted.candidate_accepted is False,
          "continue from TRUE T original43M current architecture all-joint CE and 12GT")
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
    training={};results={};saved={};gates={}
    own={}
    out.mkdir(parents=True)
    try:
        for label,child in (
            ("FULL_JOINT_NATIVE_CE",ce),
            ("FULL_JOINT_NATIVE_CE_EARLY23_X3",margin),
        ):
            set_gt_route(full,"GT_RECURRENT",gt)
            objective=("ALL_NATIVE_CE" if label=="FULL_JOINT_NATIVE_CE"
                       else "ALL_NATIVE_CE_EARLY23_X3")
            training[label]=train_joint_true43m_early23_ce_ab(
                model,sources,bridge,full,expansion,child,train,tok,
                condition="GT_RECURRENT",continuation_weight=1.0,
                expected_total=TRAIN_ROWS,expected_new=NEW_TRAIN,
                new_id_prefix="S119Q2I8W_",rounds=1,
                teacher_135m=None,teacher_KL_weight=0.0,
                objective=objective)
            gates[label]=full.gates.detach().clone()
            check(old_original_param_digest(model)==orig_digest
                  and training[label]["real_all_neural_and_gt_Adam_updates"]==TRAIN_ROWS
                  and training[label]["trainable_parameter_count"]==
                      TOTAL_EXPECTED_TRAINABLE,
                  "true original43M ALL GT12+Core/NRA3/NPA1+rank8+native head matched 27 updates")
        check(training["FULL_JOINT_NATIVE_CE"]["trained_original_decoder_calls"]==
              training["FULL_JOINT_NATIVE_CE_EARLY23_X3"]["trained_original_decoder_calls"]
              and training["FULL_JOINT_NATIVE_CE"][
                  "trained_true_SECOND_THIRD_BPE_nonEOS_targets_at_3x"]==0
              and training["FULL_JOINT_NATIVE_CE_EARLY23_X3"][
                  "trained_true_SECOND_THIRD_BPE_nonEOS_targets_at_3x"]>=TRAIN_ROWS
              and all(x["EOS_native_target_still_full_CE_and_never_masked"]
                      for x in training.values()),
              "only TRAIN second and third correct nonEOS native CE positions are 3x")
        saved={
            "CE_joint_FP16":persist_half(ce,out/CE_FP16),
            "EARLY23_X3_joint_FP16":persist_half(margin,out/EARLY_FP16),
            "CE_GT12":safe_gt_gate_sidecar(
                out/CE_GATES,gates["FULL_JOINT_NATIVE_CE"],"GT_RECURRENT"),
            "EARLY23_X3_GT12":safe_gt_gate_sidecar(
                out/EARLY_GATES,gates["FULL_JOINT_NATIVE_CE_EARLY23_X3"],"GT_RECURRENT"),
        }
        arms={
            "PREVIOUS_T_FULLJOINT_CE_FROZEN":(original_fitted,gt,.25),
            "W_FULLJOINT_CE_1X":(
                ce,gates["FULL_JOINT_NATIVE_CE"],.25),
            "W_FULLJOINT_CE_EARLY23_3X":(
                margin,gates["FULL_JOINT_NATIVE_CE_EARLY23_X3"],.25),
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
        # Gold-previous-BPE rank forensic is strictly read-only AFTER
        # all TRAIN weights are frozen, distinct from actual OWN above.
        from research.calibri_nra3.s119q2i8n_original43m_early23_goldrank import real_native_rank
        from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import real_joint_native
        gold_ranks={}
        for label,(child,gt12,gain) in arms.items():
            set_gt_route(full,"GT_RECURRENT",gt12)
            rows=[]
            for ex in newdev:
                target=int(ex["answer_native_with_eos"][1])
                check(target!=tok.eos_token_id and ex["split"]=="DEV"
                      and len(ex["answer_native_with_eos"])>=4,
                      "W DEV second human nonEOS gold only, after TRAIN, no optimization")
                prefix=ex["prefix"]+ex["answer_native_with_eos"][:1]
                with scoped_exact_native_gain(child.language_head,.25) as witness:
                    logits=real_joint_native(
                        model,sources,bridge,full,expansion,child,prefix,
                        ex["prompt_features"],contextual=True)
                check(witness[0]==1 and logits.shape==(1,NATIVE_VOCAB)
                      and full.last_forward_coverage==tuple(range(12)),
                      "one true physical 43M full49k gold second forward per source and arm")
                val=real_native_rank(logits,target)
                rows.append({"source_id":ex["id"],"rank":val[
                    "correct_gold_native_rank_full49152"],
                    "gold_top1":val["gold_native_top1"],
                    "gold_top5":val["correct_gold_rank_top5"],
                    "gold_NLL_nats":-val["gold_log_probability_native_49152"],
                    "pred_is_EOS":val["greedy_original_native_id"]==tok.eos_token_id,
                    "predicted_native_BPE_id":val["greedy_original_native_id"],
                    "gold_native_BPE_id":target,
                    "only_gold_previous_BPE_not_self_generation":True})
            gold_ranks[label]={
                "original_W_spent_author_DEV_n":14,
                "correct_second_gold_top1":sum(r["gold_top1"] for r in rows),
                "correct_second_gold_top5":sum(r["gold_top5"] for r in rows),
                "mean_second_gold_rank":sum(r["rank"] for r in rows)/len(rows),
                "mean_second_gold_native_NLL_nats":sum(r["gold_NLL_nats"] for r in rows)/len(rows),
                "wrong_EOS_at_gold_SECOND":sum(r["pred_is_EOS"] for r in rows),
                "per_source_gold_prefix_rank_49152":rows,
            }
    finally:
        with torch.no_grad():
            for k,v in old_gt.items():
                full.state_dict()[k].copy_(v)
        for p in full.parameters():
            p.requires_grad_(False)
    check(old_original_param_digest(model)==orig_digest
          and all(torch.equal(full.state_dict()[k].detach().cpu(),v)
                  for k,v in old_gt.items())
          and all(torch.equal(original_fitted.state_dict()[k].detach().cpu(),v)
                  for k,v in initial_parent.items()),
          "original frozen 43M + original GT donor + prior T CE neural+GT research parent unmodified")
    trained_calls=sum(x["trained_original_decoder_calls"] for x in training.values())
    own_calls=sum(z["scores"]["real_original43M_OWN_forwards"]
                  for r in own.values() for z in r.values())
    second_gold_calls=sum(v["original_W_spent_author_DEV_n"] for v in gold_ranks.values())
    check(trained_calls>TRAIN_ROWS*3*2 and own_calls>=3*20
          and second_gold_calls==3*NEW_HELDOUT_DEV==42,
          "real original full43M actual train + OWN native + teacher-forced diagnostic 42")
    report={
        "stage":STAGE,
        "original_S142_43M_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "completed_T_FULLJOINT_CE_REPORT_SHA256":sha(q2i8t_dir/T_REPORT),
        "completed_T_FULLJOINT_CE_neural_FP16_SHA256":sha(q2i8t_dir/T_CE_FP16),
        "completed_T_FULLJOINT_CE_GT12_FP16_SHA256":sha(q2i8t_dir/T_CE_GATES),
        "new_W_21_TRAIN_sha256":w_source["new_21_TRAIN_sha256"],
        "new_W_14_DEV_sha256":w_source["new_14_DEV_sha256"],
        "match_full43M_27TRAIN_A_B_from_identical_T_CE_FP16_state":True,
        "predeclared_early23_gold_nonEOS_BPE2BPE3_multiplier":EARLY23_MULTIPLIER,
        "no_new_neural_regions":True,
        "true_original43M_full_GT12_Core_NRA3_NPA1_head_native49k_train_AB":training,
        "saved_full_FP16_original_joint_neural_plus_separate_GT12":saved,
        "real_OWN_native_40step_new_W14_DEV_and_OLD6":own,
        "read_only_TRUE43M_native49152_gold_second_BPE_position_rank_after_training":gold_ranks,
        "physical_true43M_full_NATIVE_training_calls":trained_calls,
        "physical_true43M_FULL_NATIVE_OWN_decoder_calls":own_calls,
        "physical_true43M_gold_second_ONLY_readonly_calls":second_gold_calls,
        "original_43M_prior_S115_53751_and_prior_T_CE_FP16_all_unchanged":True,
        "original_616J7_EVAL128_GOLD_allin_bpp_CPU_NOT_ASSESSED":True,
        "no_main_LIVE_Render_Supabase_APK_or_Jessica_chat":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"RESEARCH_ONLY_EARLY23_CE_3X_JOINT_CAUSAL_AB_NEW14_INTERNAL_DEV_NO_PROMOTION",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,
                           sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8W_TRUE43M_JOINT_CE_VS_EARLY23X3_27TRAIN "+json.dumps({
        "true43M_training_native49k_calls":trained_calls,
        "true43M_OWN_decoder_calls":own_calls,
        "true43M_second_gold_rank_forwards":second_gold_calls,
        "matched_trainable_params":TOTAL_EXPECTED_TRAINABLE,
        "fresh_NEW14_W_internal_OWN_DEV":{
            k:v["NEW_DEV"]["scores"] for k,v in own.items()},
        "fresh_NEW14_W_teacherforced_gold_second_BPE_top1_top5_rank":{
            k:{a:v[a] for a in (
                "correct_second_gold_top1","correct_second_gold_top5",
                "mean_second_gold_rank","mean_second_gold_native_NLL_nats",
                "wrong_EOS_at_gold_SECOND")}
            for k,v in gold_ranks.items()},
        "candidate_accepted":False,
    },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","q2i8d_dir",
                "q2i8f_report","q2i8g_dir","q2i8k_dir","q2i8t_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,a.q2i1_dir,
        a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.q2i8d_dir,
        a.q2i8f_report,a.q2i8g_dir,a.q2i8k_dir,a.q2i8t_dir,a.out)

if __name__=="__main__":
    main()
