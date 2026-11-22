"""S119q2i8d actual original43M joint-language training: CE vs CE+135M KL.

Two ORIGINAL physical 43M full-BPE native 49,152-candidate models
starting from EXACT same q2i8a fitted Core/NRA3/NPA1/GT12 FP16 weights
and native 12 GT12 gates. NO additional neural mini areas introduced.
A= CE-only; B= same CE + fixed 0.25*T² KL(source135M||student43M)
ONLY at teacher-forced TRAIN native BPE positions following an
INCOMPLETE UTF-8 byte prefix, NEVER at native EOS. Teacher is real
pinned SmolLM2-135M-Instruct frozen and never semantic QA oracle.
Train 70 Q2I8a original authored TRAIN + six q2i1 old TRAIN ONLY,
ONE epoch and exactly matched optimizer, native EOS, learning rates,
368051 trainable child neural+GT12 gates, no original 43M training.
New heldout is 14 wholly new authored DEV from Q2I8d; tiny internal
and visible (NOT independent blind), plus 6 previously spent old DEV
regression-only. No DEV ever participates in optimizer or KL.
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
    RECURRENT_CKPT,TOTAL_EXPECTED_TRAINABLE,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import check_trained_joint
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import train_true_joint
from research.calibri_nra3.s119q2i7_real43m_utf8_weight_causal_ablation import (
    STAGE as Q2I7_STAGE,REPORT as Q2I7_REPORT,
)
from research.calibri_nra3.s119q2i8a_broader_russian_synthetic_source import (
    STAGE as SOURCE_STAGE,KINDS,
    rows as previously_registered_rows,verify as source_verify,
)
from research.calibri_nra3.s119q2i8a_real43m_broader_russian_joint import (
    STAGE as Q2I8A_STAGE,REPORT as Q2I8A_REPORT,
    RECURRENT_FP16 as Q2I8A_REC,RECURRENT_GT as Q2I8A_GT,
    verify_physical_1x_parent,prepare_native_original,one_own_dev,score,
)
from research.calibri_nra3.s119q2i8b_source135m_vs_original43m_native import (
    physical_q2i8a_fitted_child,physical_q2i8a_gates,
)
from research.calibri_nra3.s119q2i8d_new_russian_heldout_source import (
    STAGE as D_SOURCE_STAGE,ORIGIN as D_ORIGIN,
    rows as new_holdout,verify as holdout_verify,
)

STAGE="CALIBRI_S119Q2I8D_REAL43M_EXISTING_GT12_NRA3_NPA1_SOURCE135M_KL_UTF8_AB"
REPORT="S119Q2I8D_REAL43M_NATIVE_BPE_TEACHERKL_UTF8_VS_CEONLY_REPORT.json"
CE_FP16="S119Q2I8D_CE_ONLY_ORIGINAL43M_JOINT_NEURAL_FP16_G0.safetensors"
KL_FP16="S119Q2I8D_SOFT135M_UTF8KL_ORIGINAL43M_JOINT_NEURAL_FP16_G0.safetensors"
CE_GATES="S119Q2I8D_CE_ONLY_ORIGINAL43M_GT12_FP16_G0.safetensors"
KL_GATES="S119Q2I8D_SOFT135M_UTF8KL_ORIGINAL43M_GT12_FP16_G0.safetensors"
SEED=119150
ROUNDS=1
ORIGINAL_NEW_TRAIN=70
OLD_TRAIN_REHEARSAL=6
TRAIN_ROWS=ORIGINAL_NEW_TRAIN+OLD_TRAIN_REHEARSAL
NEW_HELDOUT_DEV=14
OLD_SPENT_DEV=6
TEACHER_KL_CONT_WEIGHT=.25
TEMPERATURE=2.0
CONDITIONS=("CE_ONLY","SOURCE135M_UTF8_CONTINUATION_KL")
NATIVE_VOCAB=49152


def pack_new_author_dev(tok,row):
    check(row["split"]=="DEV" and row["origin"]==D_ORIGIN
          and row["id"].startswith("S119Q2I8D_") and row["kind"] in KINDS,
          "ONLY new q2i8d author-heldout DEV, NEVER optimizer")
    clean=dict(row)
    clean["origin"]="original_newly_authored_examples_no_third_party_corpus"
    item=verified_native_answer(tok,clean)
    check(item["split"]=="DEV"
          and tok.decode(item["answer_native_without_eos"],
                         skip_special_tokens=True)==row["completion"],
          "exact original49k native bytelevel EOS correct author target")
    item["tokenizer"]=tok
    item["eos_native_id"]=int(tok.eos_token_id)
    return item


def scores_for_one(tok,model,sources,bridge,full,expansion,child,
                   gt,dev,old,*,gt_condition):
    check(len(dev)==NEW_HELDOUT_DEV and len(old)==OLD_SPENT_DEV
          and gt_condition=="GT_RECURRENT",
          "paired new14 internal DEV and old6 regression only")
    set_gt_route(full,gt_condition,gt)
    new=one_own_dev(tok,model,sources,bridge,full,expansion,child,dev,
                    condition=gt_condition)
    old_rows=one_own_dev(tok,model,sources,bridge,full,expansion,child,old,
                         condition=gt_condition)
    return {"NEW_DEV":new,"OLD_DEV":old_rows,
            "NEW_DEV_scores":score(new),"OLD_DEV_scores":score(old_rows)}


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,q2i7_dir,q2i8a_dir,out):
    check(not out.exists(),"new isolated research weights; never overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    source=source_verify()
    holdout=holdout_verify()
    check(source["stage"]==SOURCE_STAGE and source["TRAIN"]==ORIGINAL_NEW_TRAIN
          and holdout["stage"]==D_SOURCE_STAGE
          and holdout["original_new_author_heldout_DEV_count"]==NEW_HELDOUT_DEV,
          "original70 TRAIN proven author source + registered NEW DEV only")
    old_rehearsal=[v for v in old_ru_rows("TRAIN")
                   if int(v["id"].rsplit("_",1)[-1])==0]
    old_dev=[v for v in old_ru_rows("DEV")
             if int(v["id"].rsplit("_",1)[-1])==6]
    check(len(old_rehearsal)==OLD_TRAIN_REHEARSAL
          and len(old_dev)==OLD_SPENT_DEV and
          set(v["prompt"] for v in old_rehearsal).isdisjoint(
              x["prompt"] for x in old_dev),
          "six old TRAIN-only rehearsal, no seen DEV answers")
    evidence=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    q2i3=json.loads((q2i3_dir/Q2I3_REPORT).read_text(encoding="utf8"))
    p7=q2i7_dir/Q2I7_REPORT
    q2i7=json.loads(p7.read_text(encoding="utf8"))
    p8=q2i8a_dir/Q2I8A_REPORT
    q2i8a=json.loads(p8.read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE
          and q2i3["stage"]==Q2I3_STAGE
          and q2i7["stage"]==Q2I7_STAGE
          and q2i8a["stage"]==Q2I8A_STAGE
          and q2i8a["original_S93_sha256"]==sha(s93)
          and q2i8a["original_S142_real43M_GT_sha256"]==sha(s142)
          and q2i8a["prior_true_Q2I7_1X_control_physical_run_SHA256"]==sha(p7)
          and q2i8a["pre_registered_NEW_original_synthetic_7domain_source"]==source
          and not q2i8a["candidate_accepted"] and q2i8a["output_gate"]==0,
          "completed real Q2I8a original fitted child SHA/provenance exact")
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==NATIVE_VOCAB and tok.eos_token_id is not None,
          "genuine original SmolLM2 full49152 native ByteLevel/EOS")
    train_new=[prepare_native_original(tok,x)
               for x in previously_registered_rows("TRAIN")]
    train_old=[verified_native_answer(tok,x) for x in old_rehearsal]
    train=train_new+train_old
    newdev=[pack_new_author_dev(tok,x) for x in new_holdout()]
    olddev=[verified_native_answer(tok,x) for x in old_dev]
    for x in olddev:
        x["tokenizer"]=tok
        x["eos_native_id"]=int(tok.eos_token_id)
    check(len(train)==TRAIN_ROWS and len(newdev)==NEW_HELDOUT_DEV
          and len(olddev)==OLD_SPENT_DEV and
          all(x["split"]=="TRAIN" for x in train)
          and all(x["split"]=="DEV" for x in newdev+olddev)
          and {x["prompt_sha256"] for x in train}.isdisjoint(
               x["prompt_sha256"] for x in newdev+olddev)
          and max(len(x["answer_native_with_eos"]) for x in newdev+olddev)<=18,
          "strict source-disjoint DEV never enters any 43M optimizer")
    d,old_child=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,evidence["new_research_checkpoint_contextual"])
    q2f_child=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,q2f["projector_contextual_child_checkpoint"],d)
    q2i1_child=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],q2f_child)
    q2i3_child=check_trained_joint(
        q2i3_dir/RECURRENT_CKPT,
        q2i3["research_only_updated_FULL_neural_child_FP16"][
            "GT_RECURRENT"],q2i1_child)
    q2i7_child=verify_physical_1x_parent(
        q2i7_dir/"S119Q2I7_UNIFORM1X_GT12_NRA3_NPA1_NATIVEHEAD_FP16_G0.safetensors",
        q2i7["new_1x_physical_FP16_checkpoint_and_GT12_sidecar"][
            "trained_1x_neural_fp16"],q2i3_child)
    parent=physical_q2i8a_fitted_child(
        q2i8a_dir/Q2I8A_REC,
        q2i8a["reversible_research_only_FP16_neural_A_B_and_GT12_gates"][
            "GT_RECURRENT_neural"],q2i7_child)
    gates=physical_q2i8a_gates(
        q2i8a_dir/Q2I8A_GT,
        q2i8a["reversible_research_only_FP16_neural_A_B_and_GT12_gates"][
            "GT_RECURRENT_12_gates"])
    ce=copy.deepcopy(parent)
    soft=copy.deepcopy(parent)
    check(all(torch.equal(a,b) and torch.equal(a,c)
              for a,b,c in zip(parent.state_dict().values(),
                               ce.state_dict().values(),
                               soft.state_dict().values())),
          "A/B exactly identical starting physical full FP16 Core NRA3 NPA1")
    model,sources,bridge,full,expansion=load_actual_original(s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",ce)
    model.eval()
    orig_digest=old_original_param_digest(model)
    old_gt={k:v.detach().cpu().clone() for k,v in full.state_dict().items()}
    parent_state={k:v.detach().cpu().clone() for k,v in parent.state_dict().items()}
    native_full_gt_gate_contract(full,bridge)
    check(all(not p.requires_grad for name,p in model.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "pretrained original Core12 + GT donors + native 49k head NEVER learn")
    teacher=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32
    ).eval().requires_grad_(False)
    teacher_count=sum(x.numel() for x in teacher.parameters())
    check(120_000_000<=teacher_count<=150_000_000
          and teacher.config.vocab_size==NATIVE_VOCAB
          and all(not p.requires_grad for p in teacher.parameters()),
          "frozen genuinely pinned 135M source soft logits teacher ONLY")
    training={};new_gates={};dev_results={};saved={}
    out.mkdir(parents=True)
    try:
        for label,child in (("CE_ONLY",ce),
                            ("SOURCE135M_UTF8_CONTINUATION_KL",soft)):
            set_gt_route(full,"GT_RECURRENT",gates)
            teacher_for_arm=(None if label=="CE_ONLY" else teacher)
            kl_weight=(0.0 if label=="CE_ONLY" else TEACHER_KL_CONT_WEIGHT)
            training[label]=train_true_joint(
                model,sources,bridge,full,expansion,child,train,tok,
                condition="GT_RECURRENT",continuation_weight=1.0,
                expected_total=TRAIN_ROWS,expected_new=ORIGINAL_NEW_TRAIN,
                new_id_prefix="S119Q2I8A_",rounds=ROUNDS,
                teacher_135m=teacher_for_arm,teacher_KL_weight=kl_weight)
            new_gates[label]=full.gates.detach().clone()
            check(training[label]["real_all_neural_and_gt_Adam_updates"]==TRAIN_ROWS
                  and training[label]["trainable_parameter_count"]==
                      TOTAL_EXPECTED_TRAINABLE
                  and old_original_param_digest(model)==orig_digest,
                  "original physical 43M full native 76 Adam updates/arm")
        check(training["CE_ONLY"]["trained_original_decoder_calls"]==
              training["SOURCE135M_UTF8_CONTINUATION_KL"]["trained_original_decoder_calls"]
              and training["CE_ONLY"][
                "physical_SOURCE135M_native_forward_calls_at_incomplete_prefix_only"]==0
              and training["SOURCE135M_UTF8_CONTINUATION_KL"][
                "physical_SOURCE135M_native_forward_calls_at_incomplete_prefix_only"]>0,
              "ONLY causal KL uses frozen original 135M teacher and same 43M budget")
        saved={
            "CE_ONLY_neural":persist_half(ce,out/CE_FP16),
            "SOURCE135M_KL_neural":persist_half(soft,out/KL_FP16),
            "CE_ONLY_GT12":safe_gt_gate_sidecar(
                out/CE_GATES,new_gates["CE_ONLY"],"GT_RECURRENT"),
            "SOURCE135M_KL_GT12":safe_gt_gate_sidecar(
                out/KL_GATES,new_gates["SOURCE135M_UTF8_CONTINUATION_KL"],
                "GT_RECURRENT"),
        }
        for name,child,gs in (
            ("PRIOR_Q2I8A_FROZEN_PARENT",parent,gates),
            ("CE_ONLY",ce,new_gates["CE_ONLY"]),
            ("SOURCE135M_UTF8_CONTINUATION_KL",
             soft,new_gates["SOURCE135M_UTF8_CONTINUATION_KL"]),
        ):
            dev_results[name]=scores_for_one(
                tok,model,sources,bridge,full,expansion,
                child,gs,newdev,olddev,gt_condition="GT_RECURRENT")
    finally:
        with torch.no_grad():
            full.gates.copy_(old_gt["gates"])
            full.mix_logits.copy_(old_gt["mix_logits"])
        for p in full.parameters():p.requires_grad_(False)
    check(old_original_param_digest(model)==orig_digest
          and all(torch.equal(v,full.state_dict()[k].detach().cpu())
                  for k,v in old_gt.items())
          and all(torch.equal(v,parent.state_dict()[k].detach().cpu())
                  for k,v in parent_state.items()),
          "frozen original S142 43M+GT12 and learned parent remain EXACT unchanged")
    train_phys=sum(x["trained_original_decoder_calls"] for x in training.values())
    teacher_phys=sum(x[
        "physical_SOURCE135M_native_forward_calls_at_incomplete_prefix_only"]
        for x in training.values())
    dev_phys=sum(x[split+"_scores"][
                   "original_43M_Core12_native_decoder_forwards"]
                 for x in dev_results.values() for split in ("NEW_DEV","OLD_DEV"))
    report={
        "stage":STAGE,
        "original_S142_43M_sha256":sha(s142),
        "original_S93_GT_sha256":sha(s93),
        "previous_fitted_parent_q2i8a_sha256":sha(p8),
        "frozen_teacher_huggingface_MODEL":MODEL,
        "frozen_teacher_revision_SHA":MODEL_SHA,
        "genuine_teacher_source135M_parameter_count":teacher_count,
        "registered_70_author_written_training_source_sha":source["TRAIN_SHA256"],
        "registered_new_14_original_author_DEV_SHA":holdout["source_new_DEV_sha256"],
        "all_trained_rows_are_PREVIOUSLY_registered_70_TRAIN_plus_6_old_TRAIN":True,
        "no_original_q2i8a_or_other_previous_DEV_in_teacher_or_Adam":True,
        "two_matched_368051_param_GT12_Core_NRA3_NPA1_languagehead_optimizer_arms":True,
        "matched_current_architecture_NO_new_neural_areas":True,
        "true_full_native_49152_ce_at_EVERY_BPE_including_correct_EOS":True,
        "teacher_temperature_T2":TEMPERATURE,
        "teacher_full_native_49152_KL_weight_at_UTF8_continuation_ONLY":
            TEACHER_KL_CONT_WEIGHT,
        "SOURCE135M_SOFT_KL_not_applied_to_EOS_or_other_native_positions":True,
        "original_43M_real_native_forward_train_calls":train_phys,
        "original_135M_real_native_teacher_forward_calls":teacher_phys,
        "original_43M_OWN_autoregressive_new_and_old_DEV_forward_calls":dev_phys,
        "all_original_full_native_43M_decoder_forwards":train_phys+dev_phys,
        "original_genuine_135M_and_43M_forward_calls":train_phys+dev_phys+teacher_phys,
        "physical_genuine_training_A_B":training,
        "physical_OWN_sixdomain_new_and_old_DEV":dev_results,
        "only_NEW_russian_DEV_never_train_size":NEW_HELDOUT_DEV,
        "saved_full_joints_and_native_gt_gate_sidecars_FP16_gate0":saved,
        "existing_old_S115_53751_neural_FP16_preserved":True,
        "original_S142_and_teacher135m_UNCHANGED":True,
        "previous_q2i8a_parent_neural_FP16_UNCHANGED":True,
        "source_teacher_IS_NOT_correct_QA_answer_ORACLE":True,
        "new_DEV_is_small_VISIBLE_internal_NOT_external_blind":True,
        "no_full_protected_616_J7_EOS_or_all_in_packed_bpp_KV_CPU":True,
        "no_S118_GOLD_EVAL128_J7_answers_or_Jessica_chat":True,
        "no_LIVE_main_Render_Supabase_APK_or_secrets":True,
        "candidate_accepted":False,
        "output_gate":0,
        "decision":"REAL_NATIVE_43M_SOURCE_KL_AB_ORIGINAL_ARCHITECTURE_ONLY_NOT_PROMOTED",
    }
    (out/REPORT).write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119Q2I8D_REAL43M_CONTINUATION_TEACHERKL_CAUSAL_AB "+
          json.dumps({
            "actual_43M_core12_native_train_forwards":train_phys,
            "actual_135M_source_teacher_forward_calls":teacher_phys,
            "actual_43M_native_DEV_OWN_forwards":dev_phys,
            "joint_Adam_updates_each":TRAIN_ROWS,
            "train_full_native_CE":{
                k:[z["rounds"][0]["train_mean_native_CE_ALL_target_BPE_EOS"]]
                for k,z in training.items()},
            "fresh_new_DEV_FULL_correct_answers":{
                k:v["NEW_DEV_scores"]["correct_full_native_ANSWER_and_EOS"]
                for k,v in dev_results.items()},
            "fresh_new_DEV_UTF8_valid":{
                k:v["NEW_DEV_scores"]["full_UTF8_valid_own_native"]
                for k,v in dev_results.items()},
            "fresh_new_DEV_NATIVE_EOS":{
                k:v["NEW_DEV_scores"]["true_native_EOS_emitted"]
                for k,v in dev_results.items()},
            "old6_DEV_FULL_correct_answers":{
                k:v["OLD_DEV_scores"]["correct_full_native_ANSWER_and_EOS"]
                for k,v in dev_results.items()},
            "original_S142_and_all_prior_FP16_unchanged":True,
            "candidate_accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir","q2i1_dir",
                "q2i3_dir","q2i7_dir","q2i8a_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,
        a.q2i1_dir,a.q2i3_dir,a.q2i7_dir,a.q2i8a_dir,a.out)


if __name__=="__main__":
    main()
