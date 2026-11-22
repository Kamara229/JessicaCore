"""S119q2i8a broader Russian source: REAL original 43M+GT12 joint training.

Independent new original short-answer synthetic corpus:
 70 TRAIN / 14 new DEV, 7 semantic / native-byte domains,
 plus six old q2i1 TRAIN-only rehearsal (not any spent DEV source).
 2 real full-native 49,152-head answer+EOS epochs per candidate.
 Two EXACT equal-param 368051 Core/NRA3/NPA1/GT12/native-head candidates,
 same fully-trained prior q2i7 1x FP16 start & GT12 gate values:
  - recurrent S126 original cross-layer GT carry sigmoid(0)=.5;
  - no-carry matched local S126 mix=-20.
 The ONLY architectural variable is recurrent cross-layer carry.
 1x native CE loss everywhere, no token 155/30 masks, no language
 heuristics/forced EOS. Pretrained 43M/GT donors/head and 53,751 old
 original FP16 remain unchanged. Only NEW research FP16 sidecars saved.

Eval truly own-generated full native 49,152 BPE+EOS, max 18 native
tokens, on 14 source-disjoint but visible internal DEV (never optimized),
plus 6 prior already-used q2i1 DEV (retention diagnostic only).
This is NOT independently blinded/externally licensed corpus. A larger
sample (~3.9x previous 18) is still tiny for language-model competence.
Do not promote without original full 616 J7+EOS, physical bpp<=0.7
and true efficient matched KV CPU>=90%, plus independent heldout QA.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import persist_half
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX,LOCAL_CKPT as Q2D_LOCAL,REPORT as Q2D_REPORT,
    OLD_PREFIX,VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX, REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_i1_rows,provenance as old_i1_provenance,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    STAGE as Q2I1_STAGE,CONTEXT_CKPT as Q2I1_CTX,REPORT as Q2I1_REPORT,
    verified_native_answer,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,set_gt_route,
    own_token_dev,safe_gt_gate_sidecar,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    STAGE as Q2I3_STAGE,REPORT as Q2I3_REPORT,
    RECURRENT_CKPT,RECURRENT_GATES,
    scoped_reduced_native_head_gain,TOTAL_EXPECTED_TRAINABLE,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import (
    check_trained_joint,verified_gt12_gates,
)
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    native_bytelevel_full_utf8_status,first_divergence,
)
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import (
    train_true_joint,
)
from research.calibri_nra3.s119q2i7_real43m_utf8_weight_causal_ablation import (
    STAGE as Q2I7_STAGE,REPORT as Q2I7_REPORT,
    CONTROL_FP16 as Q2I7_CONTROL_FP16,CONTROL_GATES as Q2I7_CONTROL_GATES,
)
from research.calibri_nra3.s119q2i8a_broader_russian_synthetic_source import (
    STAGE as SOURCE_STAGE,ORIGIN,KINDS,
    rows as new_rows,verify as source_verify,
)

STAGE="CALIBRI_S119Q2I8A_REAL43M_BROADER_RU_NATIVE_70TRAIN_14DEV_JOINT_2EPOCH"
REPORT="S119Q2I8A_ORIGINAL43M_RU70_FULL_NATIVE_EOS_GT12_JOINT_AB_REPORT.json"
RECURRENT_FP16="S119Q2I8A_RECURRENT_GT12_CORE_NRA3_NPA1_NATIVE_FP16_G0.safetensors"
LOCAL_FP16="S119Q2I8A_LOCAL_GT12_CORE_NRA3_NPA1_NATIVE_FP16_G0.safetensors"
RECURRENT_GT="S119Q2I8A_RECURRENT_GT12_12GATES_FP16_G0.safetensors"
LOCAL_GT="S119Q2I8A_LOCAL_GT12_12GATES_FP16_G0.safetensors"
SEED=119147
EPOCHS=2
NEW_TRAIN=70
OLD_TRAIN_REHEARSAL=6
TOTAL_TRAIN=76
NEW_DEV=14
OLD_DEV=6
MAX_GENERATED_NATIVE=18
VARIANTS=("GT_RECURRENT","GT_NOCARRY_LOCAL")
CONTINUATION_WEIGHT=1.0
NEURAL_PARAMS=368051
NATIVE_VOCAB=49152


def verify_physical_1x_parent(file,metadata,prior):
    """SHA-pin complete genuinely trained q2i7 1x neural FP16 checkpoint."""
    check(file.name==Q2I7_CONTROL_FP16
          and sha(file)==metadata["sha256"]
          and file.stat().st_size==metadata["file_bytes"],
          "original physical q2i7 1x entire trained neural FP16 child")
    stored=load_file(str(file),device="cpu")
    check(stored and
          sum(v.numel() for v in stored.values())==
              metadata["saved_total_child_FP16_params"]==421793
          and all(k.startswith(OLD_PREFIX) and v.dtype==torch.float16
                  for k,v in stored.items()),
          "complete 421793-parameter q2i7 original native 49k child")
    decoded={k[len(OLD_PREFIX):]:v.float() for k,v in stored.items()}
    child=copy.deepcopy(prior)
    check(set(child.state_dict())==set(decoded),
          "loaded complete real q2i7 CoreNRA3NPA1/projection+nativehead")
    child.load_state_dict(decoded,strict=True)
    child.eval()
    check(child.output_gate==0 and not child.candidate_accepted
          and not child.activation_allowed and child.child.output_gate==0,
          "fitted Q2I7 stage must never be promoted")
    return child


def verify_physical_1x_gt(file,metadata):
    check(file.name==Q2I7_CONTROL_GATES
          and sha(file)==metadata["sha256"]
          and file.stat().st_size==metadata["file_bytes"]==128
          and metadata["12_GT12_trainable_only"]
          and metadata["condition"]=="GT_RECURRENT",
          "actual q2i7 trained RECURRENT GT12 separate 12-gate SHA")
    data=load_file(str(file),device="cpu")
    check(set(data)=={"research.s119q2i2.fullGT12_gates_fp16"}
          and data["research.s119q2i2.fullGT12_gates_fp16"].shape==(12,)
          and data["research.s119q2i2.fullGT12_gates_fp16"].dtype==
              torch.float16,
          "only new physical S126 GT12 12 scalars in prior checkpoint")
    return data["research.s119q2i2.fullGT12_gates_fp16"].float()


def prepare_native_original(tok,row):
    check(row["split"] in ("TRAIN","DEV")
          and row["origin"]==ORIGIN
          and row["kind"] in KINDS
          and row["id"].startswith("S119Q2I8A_"),
          "only registered 70TRAIN/14DEV synthetic-original new source")
    # Reuse q2i1 exact original tokenizer-prefix and EOS provenance gate,
    # by translating ONLY the local author-owned source marker for API.
    imported=dict(row)
    imported["origin"]="original_newly_authored_examples_no_third_party_corpus"
    packed=verified_native_answer(tok,imported)
    check(packed["id"]==row["id"] and
          packed["split"]==row["split"] and
          packed["kind"]==row["kind"] and
          tok.decode(packed["answer_native_without_eos"],
                     skip_special_tokens=True)==row["completion"],
          "entire actual native BPE answer and original EOS exact unchanged")
    return packed


def one_own_dev(tok,model,sources,bridge,full,expansion,child,items,
                *,condition):
    check(condition in VARIANTS and
          len(items) in (NEW_DEV,OLD_DEV)
          and all(x["split"]=="DEV" for x in items),
          "no gold DEV in 43M optimizer or previous benchmark")
    result=[]
    for x in items:
        with scoped_reduced_native_head_gain(
            child.language_head) as native_hook_count:
            got=own_token_dev(
                model,sources,bridge,full,expansion,child,x,
                condition=condition,max_generate=MAX_GENERATED_NATIVE)
        check(native_hook_count[0]==got["original_full_decoder_calls"],
              "same 0.25 original 49k native head and ONLY own tokens")
        gold=x["answer_native_with_eos"]
        ids=list(got["generated_own_native_BPE_ids"])
        validity=native_bytelevel_full_utf8_status(tok,ids)
        divergence=first_divergence(gold,ids)
        result.append({
            **got,
            "true_native_full_EOS_correct":got[
                "strict_finished_full_answer_exact"],
            "entire_generated_UTF8_state":
                validity["entire_original_native_BPE_prefix_UTF8_status"],
            "valid_full_UTF8_bytes":
                validity["entire_original_native_BPE_prefix_UTF8_status"]==
                    "VALID_COMPLETE_UTF8",
            "real_EOS_but_UNFINISHED_or_INVALID_UTF8":
                bool(got["native_EOS"]) and
                validity["entire_original_native_BPE_prefix_UTF8_status"]!=
                    "VALID_COMPLETE_UTF8",
            "first_gold_BPE_correct":divergence[
                "first_native_BPE_correct"],
            "first_native_correct_but_wrong_second":divergence[
                "first_native_correct_but_wrong_OR_missing_second"],
            "first_divergence_position_1based":divergence[
                "first_wrong_or_missing_native_BPE_position_1based"],
            "native_full_answer_target_len_including_EOS":len(gold),
            "source_internal_DEV_not_blind":True,
        })
    return result


def score(records):
    check(len(records) in (NEW_DEV,OLD_DEV),
          "read-only 14 NEWDEV or 6 old-source DEV original own responses")
    return {
        "source_count":len(records),
        "correct_full_native_ANSWER_and_EOS":sum(
            x["true_native_full_EOS_correct"] for x in records),
        "visible_short_text_exact_even_if_EOS_missing":sum(
            x["text_exact_even_if_not_finished"] for x in records),
        "true_native_EOS_emitted":sum(x["native_EOS"] for x in records),
        "native_EOS_after_incomplete_or_INVALID_UTF8":sum(
            x["real_EOS_but_UNFINISHED_or_INVALID_UTF8"] for x in records),
        "full_UTF8_valid_own_native":sum(
            x["valid_full_UTF8_bytes"] for x in records),
        "first_native_gold_BPE_correct":sum(
            x["first_gold_BPE_correct"] for x in records),
        "first_native_correct_but_wrong_second":sum(
            x["first_native_correct_but_wrong_second"] for x in records),
        "original_43M_Core12_native_decoder_forwards":sum(
            x["original_full_decoder_calls"] for x in records),
        "per_kind_complete_answers":{
            k:sum(x["true_native_full_EOS_correct"]
                  for x in records if x["domain"]==k)
            for k in sorted({x["domain"] for x in records}),
        },
    }


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,q2i7_dir,out):
    check(not out.exists(),"new stage research output, no overwrites")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    new_source=source_verify()
    check(new_source["stage"]==SOURCE_STAGE
          and new_source["TRAIN"]==NEW_TRAIN
          and new_source["DEV"]==NEW_DEV
          and new_source["no_overlap_prior_or_spent_Q2G_I1_I6"],
          "all source SHA/provenance registered before physical original load")
    older_meta=old_i1_provenance()
    old_rehearsal=[x for x in old_i1_rows("TRAIN")
                   if int(x["id"].rsplit("_",1)[-1])==0]
    old_dev=[x for x in old_i1_rows("DEV")
             if int(x["id"].rsplit("_",1)[-1])==6]
    check(len(old_rehearsal)==OLD_TRAIN_REHEARSAL and
          len(old_dev)==OLD_DEV and
          not ({x["prompt"] for x in old_rehearsal}&
               {x["prompt"] for x in old_dev}),
          "six old q2i1 TRAIN ONLY; never scored old DEV optimizer")
    q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    q2i3=json.loads((q2i3_dir/Q2I3_REPORT).read_text(encoding="utf8"))
    q2i7path=q2i7_dir/Q2I7_REPORT
    q2i7=json.loads(q2i7path.read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE and
          q2i3["stage"]==Q2I3_STAGE and
          q2i7["stage"]==Q2I7_STAGE and
          q2i7["original_S93_SHA256"]==sha(s93)
          and q2i7["original_S142_43M_GT_SHA256"]==sha(s142)
          and q2i7["previous_original_q2i3_988_run_SHA256"]==
              sha(q2i3_dir/Q2I3_REPORT)
          and q2i7["matched_RU_TRAIN_source_examples_24_per_epoch"]==24
          and q2i7["new_exact_matched_train_1x_native_continuation_weight"]==
              CONTINUATION_WEIGHT
          and q2i7["actual_train_param_count_each_368051"]==
              TOTAL_EXPECTED_TRAINABLE
          and not q2i7["candidate_accepted"]
          and q2i7["output_gate"]==0,
          "exact previously finished original physical q2i7 1x trained parent")
    q2d_parent,_=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,q2d["new_research_checkpoint_contextual"])
    q2f_parent=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,
        q2f["projector_contextual_child_checkpoint"],q2d_parent)
    q2i1_child=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],q2f_parent)
    q2i3_parent=check_trained_joint(
        q2i3_dir/RECURRENT_CKPT,
        q2i3["research_only_updated_FULL_neural_child_FP16"][
            "GT_RECURRENT"],q2i1_child)
    fitted_parent=verify_physical_1x_parent(
        q2i7_dir/Q2I7_CONTROL_FP16,
        q2i7["new_1x_physical_FP16_checkpoint_and_GT12_sidecar"][
            "trained_1x_neural_fp16"],q2i3_parent)
    starting_gates=verify_physical_1x_gt(
        q2i7_dir/Q2I7_CONTROL_GATES,
        q2i7["new_1x_physical_FP16_checkpoint_and_GT12_sidecar"][
            "trained_1x_original_GT12_gate_fp16"])
    baseline=copy.deepcopy(fitted_parent)
    recurrent=copy.deepcopy(fitted_parent)
    local=copy.deepcopy(fitted_parent)
    check(all(torch.equal(a,b) and torch.equal(a,c)
              for a,b,c in zip(
                  baseline.state_dict().values(),
                  recurrent.state_dict().values(),
                  local.state_dict().values())),
          "both 70TRAIN 2-epoch variants start EXACT SAME actual 1x FP16 child")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE==NATIVE_VOCAB
          and tok.eos_token_id is not None,
          "original S142 49152 GPT ByteLevel native BPE EOS only")
    train_new=[prepare_native_original(tok,r)
               for r in new_rows("TRAIN")]
    new_dev=[prepare_native_original(tok,r)
             for r in new_rows("DEV")]
    rehearsal=[verified_native_answer(tok,r) for r in old_rehearsal]
    old_dev_packed=[verified_native_answer(tok,r) for r in old_dev]
    train=train_new+rehearsal
    check(len(train)==TOTAL_TRAIN and len(new_dev)==NEW_DEV
          and len(old_dev_packed)==OLD_DEV
          and all(x["split"]=="TRAIN" for x in train)
          and all(x["split"]=="DEV" for x in new_dev+old_dev_packed)
          and {x["prompt_sha256"] for x in train}.isdisjoint(
              {x["prompt_sha256"] for x in new_dev+old_dev_packed})
          and Counter(x["kind"] for x in train_new)=={k:10 for k in KINDS},
          "exact 70 new TRAIN+6 old TRAIN and separately disjoint new DEV")
    for x in new_dev+old_dev_packed:
        x["tokenizer"]=tok
        x["eos_native_id"]=int(tok.eos_token_id)
    # Ensure correctness is not a benchmark impossibility due to generation cap.
    check(all(len(x["answer_native_with_eos"])<=MAX_GENERATED_NATIVE
              for x in new_dev+old_dev_packed),
          "every DEV complete Russian native answer+EOS must FIT generation cap")
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",recurrent)
    model.eval()
    original_digest=old_original_param_digest(model)
    original_full_gt={k:v.detach().cpu().clone()
                      for k,v in full.state_dict().items()}
    native_full_gt_gate_contract(full,bridge)
    baseline_snapshot={k:v.detach().cpu().clone()
                       for k,v in baseline.state_dict().items()}
    check(all(not p.requires_grad for n,p in model.named_parameters()
              if "nra3_s119m_onpath" not in n),
          "original entire 43M Core12 GreenTunnel and original head frozen")
    out.mkdir(parents=True)
    trained={};gates={};dev_results={};saved={}
    try:
        for mode,child in (("GT_RECURRENT",recurrent),
                           ("GT_NOCARRY_LOCAL",local)):
            set_gt_route(full,mode,starting_gates)
            trained[mode]=train_true_joint(
                model,sources,bridge,full,expansion,child,train,tok,
                condition=mode,continuation_weight=CONTINUATION_WEIGHT,
                expected_total=TOTAL_TRAIN,expected_new=NEW_TRAIN,
                new_id_prefix="S119Q2I8A_",rounds=EPOCHS)
            gates[mode]=full.gates.detach().clone()
            check(trained[mode]["real_all_neural_and_gt_Adam_updates"]==
                      TOTAL_TRAIN*EPOCHS
                  and trained[mode]["trainable_parameter_count"]==
                      NEURAL_PARAMS and
                  trained[mode]["actual_registered_new_TRAIN_rows"]==
                      NEW_TRAIN,
                  "actual original 70NEW+6old 2 epoch full neural physical optimizer")
            check(old_original_param_digest(model)==original_digest,
                  "pretrained S142 43M frozen after each original native training")
        check(trained["GT_RECURRENT"]["trained_original_decoder_calls"]==
              trained["GT_NOCARRY_LOCAL"]["trained_original_decoder_calls"]
              and trained["GT_RECURRENT"]["real_all_neural_and_gt_Adam_updates"]==
                  trained["GT_NOCARRY_LOCAL"]["real_all_neural_and_gt_Adam_updates"],
              "A-B same actual native 49k full-answer/EOS forward budget")
        saved={
            "GT_RECURRENT_neural":persist_half(recurrent,out/RECURRENT_FP16),
            "GT_NOCARRY_LOCAL_neural":persist_half(local,out/LOCAL_FP16),
            "GT_RECURRENT_12_gates":safe_gt_gate_sidecar(
                out/RECURRENT_GT,gates["GT_RECURRENT"],"GT_RECURRENT"),
            "GT_NOCARRY_LOCAL_12_gates":safe_gt_gate_sidecar(
                out/LOCAL_GT,gates["GT_NOCARRY_LOCAL"],"GT_NOCARRY_LOCAL"),
        }
        for key,child,gt,gtmode in (
            ("PRIOR_FROZEN_Q2I7_1X",baseline,starting_gates,"GT_RECURRENT"),
            ("GT_RECURRENT",recurrent,gates["GT_RECURRENT"],"GT_RECURRENT"),
            ("GT_NOCARRY_LOCAL",local,gates["GT_NOCARRY_LOCAL"],"GT_NOCARRY_LOCAL"),
        ):
            set_gt_route(full,gtmode,gt)
            dev_results[key]={
                "NEWDEV":one_own_dev(
                    tok,model,sources,bridge,full,expansion,child,new_dev,
                    condition=gtmode),
                "OLDDEV":one_own_dev(
                    tok,model,sources,bridge,full,expansion,child,
                    old_dev_packed,condition=gtmode),
            }
    finally:
        with torch.no_grad():
            full.gates.copy_(original_full_gt["gates"])
            full.mix_logits.copy_(original_full_gt["mix_logits"])
        for p in full.parameters():
            p.requires_grad_(False)
    check(old_original_param_digest(model)==original_digest
          and all(torch.equal(v,full.state_dict()[k].detach().cpu())
                  for k,v in original_full_gt.items())
          and all(torch.equal(v,baseline.state_dict()[k].detach().cpu())
                  for k,v in baseline_snapshot.items()),
          "all pretrained original S142 Core/GT/native LM and prior fitted FP16 parent unchanged")
    scores={key:{split:score(rows)
                 for split,rows in groups.items()}
            for key,groups in dev_results.items()}
    train_calls=sum(v["trained_original_decoder_calls"]
                    for v in trained.values())
    eval_calls=sum(v["original_43M_Core12_native_decoder_forwards"]
                   for arms in scores.values() for v in arms.values())
    results={
        "stage":STAGE,
        "original_S93_sha256":sha(s93),
        "original_S142_real43M_GT_sha256":sha(s142),
        "prior_true_Q2I7_1X_control_physical_run_SHA256":
            sha(q2i7path),
        "pre_registered_NEW_original_synthetic_7domain_source":new_source,
        "rehearsal_old_Q2I1_TRAIN_only":OLD_TRAIN_REHEARSAL,
        "verified_full_original_native_49152_vocab_and_real_EOS":True,
        "same_prior_physically_trained_q2i7_1x_initial_FP16_AND_GT12_for_A_B":True,
        "joint_actual_trainable_GT_Core_NRA3_NPA1_projector_head":NEURAL_PARAMS,
        "true_epochs":EPOCHS,
        "per_arm_new_TRAIN":NEW_TRAIN,
        "per_arm_total_with_old_TRAIN_rehearsal":TOTAL_TRAIN,
        "per_arm_optimizer_updates":TOTAL_TRAIN*EPOCHS,
        "native_full_BPE_EOS_loss_weight":CONTINUATION_WEIGHT,
        "physical_original43M_GT12_NATIVE_train_forwards":train_calls,
        "physical_original43M_GT12_NATIVE_OWN_DEV_forwards":eval_calls,
        "physical_original43M_GT12_NATIVE_TOTAL_forwards":train_calls+eval_calls,
        "physical_training_per_same_params_GT_A_B":trained,
        "reversible_research_only_FP16_neural_A_B_and_GT12_gates":saved,
        "full_OWN_native_generated_new_and_old_DEV":dev_results,
        "strict_native_EOS_and_full_UTF8_dev_scores":scores,
        "full_70_new_original_author_TRAIN_plus_14_nontrain_DEV_not_BLIND":True,
        "DEV_not_independent_external_and_oldDEV_spent":True,
        "no_gold_correct_tokens_in_autoregressive_OWN_prefix":True,
        "original_S142_Core12_GreenTunnel_full_native_LM_and_original_NRA3_53751_UNCHANGED":True,
        "prior_original_fit_q2i7_neural_FP16_UNCHANGED":True,
        "old_S118B_GOLD_EVAL128_J7_answers_not_loaded":True,
        "protected_full_616_J7_EOS_not_retested":True,
        "packed_allin_bpp_<=0p7_NOT_measured":True,
        "matched_actual_KV_cpu_>=90pct_NOT_measured":True,
        "no_LIVE_main_Render_Supabase_Jessica_chat_APK_or_secrets":True,
        "output_gate":0,
        "candidate_accepted":False,
        "decision":"PRE_REGISTERED_MORE_DIVERSE_SYNTHETIC_TRAIN_RESEARCH_ONLY_NO_PROMOTION",
    }
    (out/REPORT).write_text(json.dumps(
        results,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf8")
    print("S119Q2I8A_REAL43M_BROADER_RU_70TRAIN_JOINT_GATING "+
          json.dumps({
              "train_NEW":NEW_TRAIN,
              "internal_DEV_NEW":NEW_DEV,
              "older_TRAIN_rehearsal":OLD_TRAIN_REHEARSAL,
              "physical_original_native_forwards":train_calls+eval_calls,
              "per_arm_full_native_Adam_updates":TOTAL_TRAIN*EPOCHS,
              "native_70TRAIN_CE_round1_and_final":{
                  k:[v["rounds"][0]["train_mean_native_CE_ALL_target_BPE_EOS"],
                     v["rounds"][-1]["train_mean_native_CE_ALL_target_BPE_EOS"]]
                  for k,v in trained.items()},
              "own_NATIVE_strict_complete_answers":{
                  k:{split:r["correct_full_native_ANSWER_and_EOS"]
                     for split,r in sp.items()}
                  for k,sp in scores.items()},
              "own_NATIVE_visible_text_correct":{
                  k:{split:r["visible_short_text_exact_even_if_EOS_missing"]
                     for split,r in sp.items()}
                  for k,sp in scores.items()},
              "own_NATIVE_EOS":{
                  k:{split:r["true_native_EOS_emitted"]
                     for split,r in sp.items()}
                  for k,sp in scores.items()},
              "true_original_S142_43M_unmodified":True,
              "candidate_accepted":False,
          },sort_keys=True,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2d_dir","q2f_dir",
                "q2i1_dir","q2i3_dir","q2i7_dir","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,
        a.q2i1_dir,a.q2i3_dir,a.q2i7_dir,a.out)


if __name__=="__main__":
    main()
