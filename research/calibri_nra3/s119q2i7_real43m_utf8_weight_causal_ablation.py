"""S119q2i7: causal ORIGINAL43M native UTF8 continuation-loss 1x vs 3x.

S119q2i6 improved TRAIN native CE and EOS stopping, but made ZERO
correct 6-new-DEV own-token answers. That trial changed both GT12
recurrent/no-carry and used 3x weighting. THIS stage isolates ONLY
the 3x continuation-specific weight.

Arm A: reuse EXACT SHA-verified previously PHYSICALLY trained Q2I6
recurrent-GT12 3x FP16 child and 12-gate FP16 sidecar, no updates.
Arm B: retrain from the EXACT same prior Q2I3 recurrent checkpoint
and GT12 12-gate saved starting state with 1x uniform native full-BPE
loss, other setup held constant: same 24 TRAIN rows, 3 epochs,
72 updates, 368051 learned parameters, 0.25 native-head correction,
original 43M model, tokenizer, EOS, seed, optimizer and learning rates.
Causal trained counterpart uses no new or old q2i1 DEV labels.

Read-only compare freely generated OWN-token NEW 6 *already scored*
q2i6 author DEV and OLD 6 *already scored* q2i1 DEV, plus full native
49,152-way GOLD-PREFIX next-BPE diagnostics on the six previously
scored q2i6 DEV questions. Gold never enters SELF generation.
Record exact native EOS, UTF8 validity at EOS, first divergence,
and conditional accuracy specifically for incomplete-byte NEXT-BPE.
Check q2i6 prior physical own-token replay after FP16 reload and
report differences honestly (quantization can change tie-breaks).

No new independent blind evaluation. All frozen original S142 Core/GT
and protected old 53,751 FP16 immutable; only NEW research 1x weights
are saved, production main/Render/Supabase/Jessica unaffected.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter,defaultdict
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    persist_half,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    CONTEXT_CKPT as Q2D_CTX,LOCAL_CKPT as Q2D_LOCAL,
    REPORT as Q2D_REPORT,OLD_PREFIX,VOCAB_SIZE,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    CONTEXT_CKPT as Q2F_CTX,REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119q2f1_physical_projector_onoff import (
    load_physical_q2f_child,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import (
    rows as old_ru_rows,provenance as old_ru_provenance,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    STAGE as Q2I1_STAGE,CONTEXT_CKPT as Q2I1_CTX,
    REPORT as Q2I1_REPORT,verified_native_answer,real_joint_native,
)
from research.calibri_nra3.s119q2i2_gt12_recurrence_connectivity import (
    load_q2i1_physical_context,native_full_gt_gate_contract,
    set_gt_route,safe_gt_gate_sidecar,own_token_dev,
)
from research.calibri_nra3.s119q2i3_real_joint_gt_nra3_npa1_head import (
    STAGE as Q2I3_STAGE,REPORT as Q2I3_REPORT,
    RECURRENT_CKPT,RECURRENT_GATES,
    scoped_reduced_native_head_gain,
    TOTAL_EXPECTED_TRAINABLE,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import (
    check_trained_joint,verified_gt12_gates,
)
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    native_bytelevel_full_utf8_status,first_divergence,
)
from research.calibri_nra3.s119q2i6_fresh_ru_utf8_curriculum import (
    rows as new_ru_rows,verify as verify_curriculum,
)
from research.calibri_nra3.s119q2i6_true43m_ru_utf8_joint_train import (
    STAGE as Q2I6_STAGE,REPORT as Q2I6_REPORT,
    RECURRENT_OUT as Q2I6_RECURRENT_CKPT,
    RECURRENT_GATE_OUT as Q2I6_RECURRENT_GATES,
    prepare_new,native_train_weights,train_true_joint,
    SEED as Q2I6_SEED,ROUNDS,NEW_TRAIN,REHEARSAL_TRAIN,
    TOTAL_TRAIN,NEW_DEV,OLD_DEV,DEV_OWN_CAP,
)

STAGE="CALIBRI_S119Q2I7_REAL43M_GT_RUSSIAN_UTF8_1X_vs_3X_CAUSAL_LOSS"
REPORT="S119Q2I7_TRUE_ORIGINAL43M_BYTE_CONTINUATION_UNIFORM_vs_UPWEIGHTED.json"
CONTROL_FP16="S119Q2I7_UNIFORM1X_GT12_NRA3_NPA1_NATIVEHEAD_FP16_G0.safetensors"
CONTROL_GATES="S119Q2I7_UNIFORM1X_GT12_RECURRENT_GATES_FP16_G0.safetensors"
SOURCE_SIZES=(18,6,6,6)
CONDITIONS=("OLD_PHYSICAL_3X","SAME_START_UNIFORM_1X")
CONTROL_WEIGHT=1.0
PRIOR_WEIGHT=3.0
TRAIN_PARAMS=368051
SEED=Q2I6_SEED


def physical_weighted_q2i6_child(path,metadata,original_child):
    check(path.name==Q2I6_RECURRENT_CKPT
          and sha(path)==metadata["sha256"]
          and path.stat().st_size==metadata["file_bytes"],
          "exact prior Q2I6 3x physically trained FP16 checkpoint SHA")
    state=load_file(str(path),device="cpu")
    check(state and
          sum(x.numel() for x in state.values())==
              metadata["saved_total_child_FP16_params"]==421793
          and all(k.startswith(OLD_PREFIX) and
                  x.dtype==torch.float16 for k,x in state.items()),
          "all prior Q2I6 421793 original neural FP16 tensors, no proxy")
    rekey={k[len(OLD_PREFIX):]:v.float() for k,v in state.items()}
    child=copy.deepcopy(original_child)
    check(set(rekey)==set(child.state_dict()),
          "full original trained q2i3/Q2I6 Core NRA3 NPA1 native head keys")
    child.load_state_dict(rekey,strict=True)
    child.eval()
    check(child.output_gate==0
          and not child.candidate_accepted
          and not child.activation_allowed
          and child.child.output_gate==0,
          "prior Q2I6 3x physical model remains unaccepted")
    return child


def physical_q2i6_gate(path,metadata):
    check(path.name==Q2I6_RECURRENT_GATES
          and sha(path)==metadata["sha256"]
          and path.stat().st_size==metadata["file_bytes"]==128
          and metadata["12_GT12_trainable_only"]
          and metadata["condition"]=="GT_RECURRENT",
          "exact Q2I6 recurrent 3x 12-gate FP16 sidecar")
    state=load_file(str(path),device="cpu")
    check(set(state)=={"research.s119q2i2.fullGT12_gates_fp16"}
          and next(iter(state.values())).numel()==12
          and next(iter(state.values())).dtype==torch.float16,
          "real S126 original GT12 12 scalar gate parameters only")
    return next(iter(state.values())).float()


def scored_utf8_own(tok,report,example):
    check(report["source_id"]==example["id"]
          and report["variant"]=="GT_RECURRENT"
          and "generated_own_native_BPE_ids" in report,
          "same physically generated recurrent GT own-token source")
    own=list(report["generated_own_native_BPE_ids"])
    gold=list(example["answer_native_with_eos"])
    native_status=native_bytelevel_full_utf8_status(tok,own)
    div=first_divergence(gold,own)
    complete=bool(report["native_EOS"])
    return {
        "source_id":example["id"],
        "kind":example["kind"],
        "original_own_native_BPE_ids":own,
        "original_own_native_text":report["decoded_text"],
        "true_native_EOS_emitted":complete,
        "EOS_emitted_with_INCOMPLETE_or_INVALID_UTF8":
            complete and native_status[
                "entire_original_native_BPE_prefix_UTF8_status"]!=
                    "VALID_COMPLETE_UTF8",
        "entire_original_bytelevel_utf8_status":
            native_status["entire_original_native_BPE_prefix_UTF8_status"],
        "strict_full_answer_exact_WITH_EOS":
            bool(report["strict_finished_full_answer_exact"]),
        "visible_text_exact_without_EOS":
            bool(report["text_exact_even_if_not_finished"]),
        "first_native_BPE_correct":div["first_native_BPE_correct"],
        "first_wrong_or_missing_native_BPE_position_1based":
            div["first_wrong_or_missing_native_BPE_position_1based"],
        "wrong_second_after_correct_first":
            div["first_native_correct_but_wrong_OR_missing_second"],
        "physical_original_native_decoder_forwards":
            report["original_full_decoder_calls"],
        "no_gold_target_used_in_original_self_prefix":True,
    }


def physical_dev_gold_continuation(tok,model,sources,bridge,full,expansion,
                                   child,dev,*,condition):
    """DEV diagnostic AFTER training only; no optimizer/weight selection."""
    check(len(dev)==NEW_DEV and
          all(x["split"]=="DEV" for x in dev)
          and condition in CONDITIONS,
          "only already scored new six Q2I6 DEV and identical full q2i3 core")
    by_kind=defaultdict(list)
    n=0
    for example in dev:
        gold=example["answer_native_with_eos"]
        weights,statuses=native_train_weights(
            tok,gold,continuation_weight=PRIOR_WEIGHT)
        check(len(weights)==len(gold)==len(statuses)
              and gold[-1]==tok.eos_token_id,
              "actual original full gold native EOS correct BPE")
        for idx,target in enumerate(gold):
            prefix=example["prefix"]+gold[:idx]
            with scoped_reduced_native_head_gain(
                child.language_head) as count:
                logits=real_joint_native(
                    model,sources,bridge,full,expansion,child,prefix,
                    example["prompt_features"],contextual=True)
            check(count[0]==1
                  and full.last_forward_coverage==tuple(range(12))
                  and logits.shape==(1,VOCAB_SIZE)
                  and not full._active and full._carry is None,
                  "ONE original S142 full 49k GT12 forward per DEV GOLD BPE")
            v=logits.detach().float().cpu()[0]
            ce=float(F.cross_entropy(v.view(1,-1),
                    torch.tensor([int(target)])).item())
            predicted=int(torch.argmax(v))
            kind=("CONTINUATION" if
                statuses[idx]=="INCOMPLETE_UTF8_NEEDS_CONTINUATION"
                else "EOS" if idx==len(gold)-1 else "OTHER")
            by_kind[kind].append({
                "source_id":example["id"],
                "gold_native_BPE_index_1based":idx+1,
                "native_target_correct_top1":predicted==int(target),
                "native_correct_gold_rank_49152":
                    int(torch.sum(v>v[int(target)]))+1,
                "gold_next_native_BPE_CE":ce,
                "gold_BPE_never_in_self_prefix":True,
            })
            n+=1
            del logits
    check("CONTINUATION" in by_kind and
          "EOS" in by_kind and
          n>=NEW_DEV*2,
          "real Russian continuation and EOS gold positions actually checked")
    return {
        "number_REAL_original43M_decoder_forwards":n,
        "native_source_next_token_diagnostic_NOT_self_generated":True,
        "metrics":{
            kind:{
                "count":len(items),
                "correct_gold_next_BPE_TOP1":sum(
                    x["native_target_correct_top1"] for x in items),
                "mean_gold_next_native_BPE_CE":
                    sum(x["gold_next_native_BPE_CE"] for x in items)/len(items),
                "mean_gold_next_native_BPE_rank_49152":
                    sum(x["native_correct_gold_rank_49152"]
                        for x in items)/len(items),
            } for kind,items in sorted(by_kind.items())
        },
        "per_target_only_previously_scored_NEWDEV":dict(by_kind),
    }


def aggregate_own(records):
    check(len(records)==6,"all six DEV examples complete")
    return {
        "count":6,
        "strict_fully_correct_WITH_native_EOS":sum(
            x["strict_full_answer_exact_WITH_EOS"] for x in records),
        "visible_short_answer_exact_even_if_EOS_missing":sum(
            x["visible_text_exact_without_EOS"] for x in records),
        "real_native_EOS_finished":sum(
            x["true_native_EOS_emitted"] for x in records),
        "EOS_emitted_after_invalid_or_incomplete_utf8":sum(
            x["EOS_emitted_with_INCOMPLETE_or_INVALID_UTF8"]
            for x in records),
        "VALID_COMPLETE_UTF8_own_output":sum(
            x["entire_original_bytelevel_utf8_status"]==
            "VALID_COMPLETE_UTF8" for x in records),
        "correct_first_native_BPE":sum(
            x["first_native_BPE_correct"] for x in records),
        "first_native_correct_but_second_wrong":sum(
            x["wrong_second_after_correct_first"] for x in records),
        "real_original_S142_43M_native_decoder_forwards":sum(
            x["physical_original_native_decoder_forwards"] for x in records),
    }


def run(s93,side,s142,q2d_dir,q2f_dir,q2i1_dir,q2i3_dir,q2i6_dir,out):
    check(not out.exists(),"research-only new files, no overwrite")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    src=verify_curriculum()
    old_meta=old_ru_provenance()
    train_new=new_ru_rows("TRAIN")
    new_dev=new_ru_rows("DEV")
    old_rehearsal=[v for v in old_ru_rows("TRAIN")
                    if int(v["id"].rsplit("_",1)[-1])==0]
    old_dev=[v for v in old_ru_rows("DEV")
             if int(v["id"].rsplit("_",1)[-1])==6]
    check((len(train_new),len(old_rehearsal),len(new_dev),len(old_dev))==
          SOURCE_SIZES
          and src["fresh_original_author_TRAIN"]==NEW_TRAIN
          and src["fresh_original_author_DEV"]==NEW_DEV,
          "exact previous original-author TRAIN and already-spent DEV")
    q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/Q2D_CTX,q2d_dir/Q2D_LOCAL,s93,s142)
    q2f=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    q2i1=json.loads((q2i1_dir/Q2I1_REPORT).read_text(encoding="utf8"))
    q2i3=json.loads((q2i3_dir/Q2I3_REPORT).read_text(encoding="utf8"))
    q2i6path=q2i6_dir/Q2I6_REPORT
    q2i6=json.loads(q2i6path.read_text(encoding="utf8"))
    check(q2i1["stage"]==Q2I1_STAGE and
          q2i3["stage"]==Q2I3_STAGE and
          q2i6["stage"]==Q2I6_STAGE and
          q2i6["original_S93_SHA256"]==sha(s93)
          and q2i6["original_S142_Core_GT12_43M_SHA256"]==sha(s142)
          and q2i6["parent_q2i3_full_988_physical_run_SHA256"]==
              sha(q2i3_dir/Q2I3_REPORT)
          and q2i6["fresh_original_author_bytelevel_SOURCE"]==src
          and q2i6["matched_368051_actual_optimizer_params"]==
              TOTAL_EXPECTED_TRAINABLE==TRAIN_PARAMS
          and q2i6["actual_original_43M_native_decoder_calls_TRAIN"]==684
          and q2i6["true_native_russian_bytes_incomplete_prefix_CE_weight"]==
              PRIOR_WEIGHT
          and q2i6["three_joint_epochs_with_rehearsal"]==ROUNDS
          and q2i6["candidate_accepted"] is False
          and q2i6["output_gate"]==0,
          "complete previous real 977-original-decoder 3x q2i6 provenance")
    q2d_parent,_=load_q2d_trained_child(
        q2d_dir/Q2D_CTX,q2d["new_research_checkpoint_contextual"])
    q2f_parent=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,
        q2f["projector_contextual_child_checkpoint"],q2d_parent)
    q2i1_base=load_q2i1_physical_context(
        q2i1_dir/Q2I1_CTX,
        q2i1["learned_checkpoint_fp16"]["joint_contextual"],q2f_parent)
    q2i3_fitted=check_trained_joint(
        q2i3_dir/RECURRENT_CKPT,
        q2i3["research_only_updated_FULL_neural_child_FP16"][
            "GT_RECURRENT"],q2i1_base)
    q2i3_initial_gates=verified_gt12_gates(
        q2i3_dir/RECURRENT_GATES,
        q2i3["research_only_updated_GT12_gate_FP16_sidecar"][
            "GT_RECURRENT"])
    pretrained_3x=physical_weighted_q2i6_child(
        q2i6_dir/Q2I6_RECURRENT_CKPT,
        q2i6["new_training_FP16_sidecar_provenance"][
            "recurrent_child"],q2i3_fitted)
    pretrained_3x_gates=physical_q2i6_gate(
        q2i6_dir/Q2I6_RECURRENT_GATES,
        q2i6["new_training_FP16_sidecar_provenance"][
            "recurrent_GT12"])
    trained_1x=copy.deepcopy(q2i3_fitted)
    start_q2i3_sha={k:v.detach().cpu().clone()
                    for k,v in q2i3_fitted.state_dict().items()}
    prior_3x_state={k:v.detach().cpu().clone()
                    for k,v in pretrained_3x.state_dict().items()}
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE and tok.eos_token_id is not None,
          "original S142 49152 native byte-level GPT tokenizer and EOS")
    new_train=[prepare_new(tok,v) for v in train_new]
    rehearsal=[verified_native_answer(tok,v) for v in old_rehearsal]
    dev_new=[prepare_new(tok,v) for v in new_dev]
    dev_old=[verified_native_answer(tok,v) for v in old_dev]
    train=new_train+rehearsal
    check(len(train)==TOTAL_TRAIN and
          all(v["split"]=="TRAIN" for v in train)
          and len(dev_new)==len(dev_old)==6
          and {x["prompt_sha256"] for x in train}.isdisjoint(
              {x["prompt_sha256"] for x in dev_new+dev_old}),
          "0 DEV target used in real Adam native training")
    for x in dev_new+dev_old:
        x["tokenizer"]=tok
        x["eos_native_id"]=int(tok.eos_token_id)
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    model.model.add_module("nra3_s119m_onpath",trained_1x)
    model.eval()
    original_digest=old_original_param_digest(model)
    original_gt_state={k:v.detach().cpu().clone()
                       for k,v in full.state_dict().items()}
    native_full_gt_gate_contract(full,bridge)
    check(all(not p.requires_grad for name,p in model.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "all original pretrained 43M Core/GT and 49k LM-head frozen")
    out.mkdir(parents=True)
    output={};train_result={};first_1x_gates=None
    physical=0
    try:
        set_gt_route(full,"GT_RECURRENT",q2i3_initial_gates)
        train_result=train_true_joint(
            model,sources,bridge,full,expansion,
            trained_1x,train,tok,condition="GT_RECURRENT",
            continuation_weight=CONTROL_WEIGHT)
        check(train_result["native_byte_continuation_weight"]==
              CONTROL_WEIGHT
              and train_result["real_all_neural_and_gt_Adam_updates"]==
              TOTAL_TRAIN*ROUNDS==72
              and train_result["trained_original_decoder_calls"]==
                  q2i6["training_real_fullBPE_EOS"]["GT_RECURRENT"][
                      "trained_original_decoder_calls"]==342,
              "EXACT same 342 physical original forward 72optimizer 1x/3x budget")
        first_1x_gates=full.gates.detach().clone()
        check(all(torch.equal(v,q2i3_fitted.state_dict()[k].detach().cpu())
                  for k,v in start_q2i3_sha.items()) and
              all(torch.equal(v,pretrained_3x.state_dict()[k].detach().cpu())
                  for k,v in prior_3x_state.items()),
              "new 1x Adam changed only new independent child, not parent 3x")
        physical+=train_result["trained_original_decoder_calls"]
        output["trained_1x_neural_fp16"]=persist_half(
            trained_1x,out/CONTROL_FP16)
        output["trained_1x_original_GT12_gate_fp16"]=safe_gt_gate_sidecar(
            out/CONTROL_GATES,first_1x_gates,"GT_RECURRENT")
        records={};diag={}
        for condition,child,gate in (
            ("OLD_PHYSICAL_3X",pretrained_3x,pretrained_3x_gates),
            ("SAME_START_UNIFORM_1X",trained_1x,first_1x_gates),
        ):
            set_gt_route(full,"GT_RECURRENT",gate)
            splits={}
            for split,examples in (("NEWDEV",dev_new),("OLDDEV",dev_old)):
                current=[]
                for e in examples:
                    with scoped_reduced_native_head_gain(
                        child.language_head) as witness:
                        measured=own_token_dev(
                            model,sources,bridge,full,expansion,child,e,
                            condition="GT_RECURRENT")
                    check(witness[0]==measured[
                        "original_full_decoder_calls"],
                        "all own tokens come from exact same native 0.25 head")
                    current.append(scored_utf8_own(tok,measured,e))
                    physical+=measured["original_full_decoder_calls"]
                splits[split]={
                    "own_token_trace":current,
                    "own_token_summary":aggregate_own(current),
                }
            records[condition]=splits
            diag[condition]=physical_dev_gold_continuation(
                tok,model,sources,bridge,full,expansion,child,dev_new,
                condition=condition)
            physical+=diag[condition]["number_REAL_original43M_decoder_forwards"]
    finally:
        with torch.no_grad():
            full.gates.copy_(original_gt_state["gates"])
            full.mix_logits.copy_(original_gt_state["mix_logits"])
        for p in full.parameters():
            p.requires_grad_(False)
    check(old_original_param_digest(model)==original_digest
          and all(torch.equal(v,full.state_dict()[k].detach().cpu())
                  for k,v in original_gt_state.items())
          and all(torch.equal(v,pretrained_3x.state_dict()[k].detach().cpu())
                  for k,v in prior_3x_state.items()),
          "frozen original S142 43M full GreenTunnel and q2i6 3x FP16 intact")
    # Check previously physically generated A own tokens after loading
    # previous 3x *FP16* to distinguish harmless re-quantization effects.
    parity={}
    for split in ("NEWDEV","OLDDEV"):
        previous=q2i6["every_own_native_DEV_full_answer_trace"][
            "GT_RECURRENT"][split]
        new=records["OLD_PHYSICAL_3X"][split]["own_token_trace"]
        check(len(previous)==len(new)==6
              and all(a["source_id"]==b["source_id"]
                      for a,b in zip(previous,new)),
              "exact 6 previous physical own-token source IDs same order")
        parity[split]={
            "fully_reproduced_OWN_native_BPE":sum(
                a["generated_own_native_BPE_ids"]==
                b["original_own_native_BPE_ids"]
                for a,b in zip(previous,new)),
            "first_native_BPE_reproduced":sum(
                a["generated_own_native_BPE_ids"][0]==
                b["original_own_native_BPE_ids"][0]
                for a,b in zip(previous,new)),
            "FP16_round_trip_may_shift_native_ties":True,
        }
    report={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_43M_GT_SHA256":sha(s142),
        "previous_original_q2i3_988_run_SHA256":sha(q2i3_dir/Q2I3_REPORT),
        "previous_q2i6_actual_977_forwards_3x_report_SHA256":sha(q2i6path),
        "new_original_author_SOURCE_hashes":src,
        "old_q2i1_original_TRAIN_rehearsal_only":REHEARSAL_TRAIN,
        "matched_RU_TRAIN_source_examples_24_per_epoch":TOTAL_TRAIN,
        "matched_three_original43M_full_native_train_rounds":ROUNDS,
        "both_START_same_saved_q2i3_recurrent_FP16_and_original_GT12":True,
        "actual_train_param_count_each_368051":TRAIN_PARAMS,
        "prior_exactly_trained_3x_native_continuation_weight":PRIOR_WEIGHT,
        "new_exact_matched_train_1x_native_continuation_weight":CONTROL_WEIGHT,
        "previous_3x_training_original_full_native_43M_forwards":
            q2i6["training_real_fullBPE_EOS"]["GT_RECURRENT"][
                "trained_original_decoder_calls"],
        "new_1x_training_original_full_native_43M_forwards":
            train_result["trained_original_decoder_calls"],
        "new_1x_training_metrics":train_result,
        "prior_preexisting_3x_training_metrics":
            q2i6["training_real_fullBPE_EOS"]["GT_RECURRENT"],
        "new_1x_physical_FP16_checkpoint_and_GT12_sidecar":output,
        "already_scored_small_DEV_own_generated_outputs":records,
        "already_scored_NEWDEV_native_gold_prefix_continuation_only":diag,
        "prior_3x_fp16_readback_own_BPE_parity":parity,
        "new_original_43M_1x_train_AND_two_oldDEV_READONLY_forward_calls":physical,
        "all_orig_pretrained_S142_43M_GT12_native_LM_unchanged":True,
        "old_53751_pretrained_neural_FP16_unchanged":True,
        "all_previous_saved_physical_3x_and_q2i3_full_weights_unchanged":True,
        "no_DEV_gradient_or_older_spent_Q2G_answers":True,
        "new_and_OLD_DEV_already_exposed_NOT_BLIND_external":True,
        "no_full_616_J7_EOS_replay_packed_bpp_or_KV_CPU_throughput":True,
        "no_S118_GOLD_spent_EVAL128_J7_labels_or_Jessica_chat":True,
        "no_main_LIVE_Render_Supabase_APK_secrets":True,
        "candidate_accepted":False,
        "output_gate":0,
        "decision":"READONLY_CAUSAL_WEIGHT_1X_VS_3X_WITH_NEW_1X_FP16_RESEARCH_ONLY",
    }
    (out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2I7_TRUE43M_NATIVE_BYTE_CONTINUATION_LOSS_CAUSAL_AB "+
          json.dumps({
            "new_original43M_decoder_forwards":physical,
            "matched_optimizer_steps":TOTAL_TRAIN*ROUNDS,
            "uniform_1x_TRAIN_native_CE_first_last":[
                train_result["rounds"][0]["train_mean_native_CE_ALL_target_BPE_EOS"],
                train_result["rounds"][-1]["train_mean_native_CE_ALL_target_BPE_EOS"],
            ],
            "prior_3x_TRAIN_native_CE_first_last":[
                q2i6["training_real_fullBPE_EOS"]["GT_RECURRENT"][
                    "rounds"][0]["train_mean_native_CE_ALL_target_BPE_EOS"],
                q2i6["training_real_fullBPE_EOS"]["GT_RECURRENT"][
                    "rounds"][-1]["train_mean_native_CE_ALL_target_BPE_EOS"],
            ],
            "DEV_strict_full_answer_and_EOS":{
                k:{sp:v[sp]["own_token_summary"][
                    "strict_fully_correct_WITH_native_EOS"]
                    for sp in ("NEWDEV","OLDDEV")}
                for k,v in records.items()},
            "DEV_EOS_and_invalid_UTF8":{
                k:{sp:[v[sp]["own_token_summary"]["real_native_EOS_finished"],
                      v[sp]["own_token_summary"][
                        "EOS_emitted_after_invalid_or_incomplete_utf8"]]
                    for sp in ("NEWDEV","OLDDEV")}
                for k,v in records.items()},
            "NEWDEV_gold_continuation_metrics":{
                k:v["metrics"].get("CONTINUATION",{})
                for k,v in diag.items()},
            "all_original_weights_unchanged":True,
            "candidate_accepted":False,
          },ensure_ascii=False,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2d_dir","q2f_dir",
                 "q2i1_dir","q2i3_dir","q2i6_dir","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2d_dir,a.q2f_dir,
        a.q2i1_dir,a.q2i3_dir,a.q2i6_dir,a.out)


if __name__=="__main__":
    main()
