"""S119q2f1: real PHYSICAL projector ON/OFF causal ablation on trained child.

Strict source-only analysis of actually trained S119q2f contextual/local
rank8 projector + Core-NRA3-NPA1 joint children on untouched original S142
43M Core12+GreenTunnel full 49152-native LM head, using original layer5
onepass hook. ONE intervention dimension per candidate: same weights with
projector ON (exact learned 6-slot h0..h5 path) or temporarily OFF (strict
repeated physical h5 virtual slots). All trained A/B state dict tensors
verified identical before/after. No optimization, parameter changes,
GOLD/EVAL128/J7 or independent blind benchmark claims. Same 30 exposed
q2a CALIBRATION + q2c1 SOURCE_PROBE original authored template prompts:
conditional native first/divergent BPE only; not own-token full answer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
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
    verify_q2a_artifact,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    OLD_PREFIX,source_inventory,pack_sources,VOCAB_SIZE,PROBE_COUNTS,
    REPORT as Q2D_REPORT,CONTEXT_CKPT as Q2D_CONTEXT_CKPT,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    load_q2d_proof,STAGE as Q2D1_STAGE,
)
from research.calibri_nra3.s119q2e_causal_early_view_diagnostic import (
    load_q2d_trained_child,
)
from research.calibri_nra3.s119q2f_learned_causal_early_projection import (
    LearnedCausalEarlyProjector,STAGE as Q2F_STAGE,
    CONTEXT_CKPT as Q2F_CTX,LOCAL_CKPT as Q2F_LOCAL,
    REPORT as Q2F_REPORT,
)
from research.calibri_nra3.s119p_real43m_onpath_component_ablation import (
    temporary_gate_mask,MASKS,
)
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    onepass_full_native_logits,
)

STAGE="CALIBRI_S119Q2F1_REAL43M_GT_CAUSAL_LEARNED_PROJECTOR_ON_OFF_READONLY"
REPORT="S119Q2F1_REAL43M_GT_PROJECTOR_ON_OFF_30SOURCE_CAUSAL_REPORT.json"
N_PROBES=30
N_AB_VARIANTS=2
N_PROJECTION_CONDITIONS=2
N_ZERO_GATE=2
N_ORIGINAL_DECODER_FORWARDS=N_PROBES*N_AB_VARIANTS*N_PROJECTION_CONDITIONS+N_ZERO_GATE
SOURCE_SPLITS=PROBE_COUNTS
PER_KIND=5


@contextmanager
def temporarily_switch_projection(adapter,enabled:bool):
    """Change ONLY runtime ON/OFF switch; never mutate fitted weights."""
    check(type(enabled) is bool and
          adapter.projection_enabled is True and
          not adapter.candidate_accepted and
          adapter.output_gate==0,
          "predeclared physical fitted projector ON and unaccepted")
    old=adapter.projection_enabled
    try:
        adapter.projection_enabled=enabled
        yield
    finally:
        adapter.projection_enabled=old
        check(adapter.projection_enabled is True,
              "restore FULL trained projector ON after every ablation")


def load_physical_q2f_child(file:Path,metadata,old_q2d_context):
    check(file.name in (Q2F_CTX,Q2F_LOCAL) and
          sha(file)==metadata["sha256"] and
          file.stat().st_size==metadata["file_bytes"],
          "physically saved q2f complete FP16 child provenance exact")
    raw=load_file(str(file),device="cpu")
    check(len(raw)>20 and
          all(k.startswith(OLD_PREFIX) and v.dtype==torch.float16
              for k,v in raw.items()) and
          sum(v.numel() for v in raw.values())==
              metadata["saved_total_child_FP16_params"],
          "entire q2f 120999-trainable + protected trained child FP16")
    adapter=LearnedCausalEarlyProjector(old_q2d_context)
    state={k[len(OLD_PREFIX):]:v.float() for k,v in raw.items()}
    check(set(adapter.state_dict())==set(state),
          "all learned rank8/projector/Core/NRA3/NPA1 tensors present")
    adapter.load_state_dict(state,strict=True)
    check(adapter.projection_enabled and
          adapter.output_gate==0 and
          not adapter.activation_allowed and not adapter.candidate_accepted
          and adapter.child.output_gate==0 and
          not adapter.child.candidate_accepted,
          "fail-closed unaccepted model; default projector ON")
    return adapter


def fixed_native_source_case(original,sources,bridge,full,expansion,
                             adapter,item,*,contextual,projector_on):
    check(item["source_split"] in SOURCE_SPLITS and
          type(projector_on) is bool,
          "no TRAIN, new blind QA or unregistered projection conditions")
    target_position=(
        item["first_distinct_position"]
        if item["source_family"]=="q2c1" else 0)
    target=int(item["completion_ids"][target_position])
    prefix=item["prefix"]+item["completion_ids"][:target_position]
    with temporarily_switch_projection(adapter,projector_on):
        logits,witness=onepass_full_native_logits(
            original,sources,bridge,full,expansion,adapter,
            prefix,item["prompt_features"],contextual=contextual)
        logits=logits.detach().cpu()
        scaled_norm=adapter.last_projected_H320_norm
    check(logits.shape==(1,VOCAB_SIZE) and
          witness["candidate_forward_calls"]==1 and
          0<=target<VOCAB_SIZE and
          (scaled_norm==0. if not projector_on else scaled_norm>0.),
          "ONE original forward, ON/OFF really affected learned projected H320")
    ce=float(F.cross_entropy(
        logits,torch.tensor([target],dtype=torch.long)).item())
    result={
        "kind":item["kind"],
        "source_id":item["id"],
        "source_split":item["source_split"],
        "source_prompt_sha256":item["prompt_sha256"],
        "target_native_id":target,
        "target_position":target_position,
        "projector_on":projector_on,
        "projector_virtual_H320_norm":scaled_norm,
        "actual_core_H320_residual_norm":witness["adapter_delta_H320"],
        "conditional_native_BPE_CE":ce,
        "correct_native_next_BPE_greedy":int(logits.argmax(-1))==target,
        "predicted_native_BPE_id":int(logits.argmax(-1)),
        "not_independent_own_token_full_generation":True,
    }
    return result,logits


def compare_same_weights_onoff(on,off):
    check(len(on)==len(off)==N_PROBES and
          Counter(r["source_split"] for r in on)==SOURCE_SPLITS and
          all(a["source_id"]==b["source_id"] and
              a["target_native_id"]==b["target_native_id"] and
              a["target_position"]==b["target_position"] and
              a["kind"]==b["kind"] and a["projector_on"] is True and
              b["projector_on"] is False for a,b in zip(on,off)),
          "exactly same 30 physically measured native positions, no model tuning")
    per_domain=defaultdict(list)
    paired=[]
    for a,b in zip(on,off):
        difference=a["conditional_native_BPE_CE"]-b["conditional_native_BPE_CE"]
        per_domain[a["kind"]].append(difference)
        paired.append({
            "source_id":a["source_id"],"kind":a["kind"],
            "ON_minus_OFF_native_CE":difference,
            "ON_greedy_native_BPE_id":a["predicted_native_BPE_id"],
            "OFF_greedy_native_BPE_id":b["predicted_native_BPE_id"],
            "greedy_changed":a["predicted_native_BPE_id"]!=b[
                "predicted_native_BPE_id"],
            "ON_correct":a["correct_native_next_BPE_greedy"],
            "OFF_correct":b["correct_native_next_BPE_greedy"],
        })
    check(len(per_domain)==6 and
          all(len(v)==PER_KIND for v in per_domain.values()),
          "six exact original source domains each five nonblind prompts")
    return {
        "ON_mean_conditional_native_BPE_CE":
            sum(x["conditional_native_BPE_CE"] for x in on)/N_PROBES,
        "OFF_mean_conditional_native_BPE_CE":
            sum(x["conditional_native_BPE_CE"] for x in off)/N_PROBES,
        "ON_minus_OFF_mean_native_CE":
            sum(x["ON_minus_OFF_native_CE"] for x in paired)/N_PROBES,
        "ON_lower_native_CE_prompts":
            sum(x["ON_minus_OFF_native_CE"]<0 for x in paired),
        "ON_higher_native_CE_prompts":
            sum(x["ON_minus_OFF_native_CE"]>0 for x in paired),
        "native_greedy_predictions_CHANGED":
            sum(x["greedy_changed"] for x in paired),
        "ON_correct_native_greedy":
            sum(x["ON_correct"] for x in paired),
        "OFF_correct_native_greedy":
            sum(x["OFF_correct"] for x in paired),
        "ON_minus_OFF_mean_native_CE_by_domain":{
            k:sum(vals)/len(vals) for k,vals in sorted(per_domain.items())},
        "known_same_author_SOURCE_PROBE_not_blind":True,
        "conditional_teacher_forcing_NOT_own_native_full_answer":True,
        "per_source_readonly_comparisons":paired,
    }


def run(s93,side,s142,q2a_dir,q2d_dir,q2f_dir,reference,out):
    check(not out.exists(),"only fresh unaccepted read-only research diagnostic")
    torch.manual_seed(119137)
    torch.set_num_threads(2)
    source_rows,source_meta=verify_q2a_artifact(q2a_dir)
    parent_q2d=load_q2d_proof(
        q2d_dir/Q2D_REPORT,
        q2d_dir/"S119Q2D_DEEPER_CONTEXTUAL_NATIVE_96TRAIN4EPOCH_GATE0_FP16.safetensors",
        q2d_dir/"S119Q2D_DEEPER_LOCAL_NATIVE_96TRAIN4EPOCH_GATE0_FP16.safetensors",
        s93,s142)
    proof=json.loads((q2f_dir/Q2F_REPORT).read_text(encoding="utf8"))
    check(proof["stage"]==Q2F_STAGE and
          proof["original_S93_SHA256"]==sha(s93) and
          proof["original_S142_GT_SHA256"]==sha(s142) and
          proof["parent_q2d_training_report_SHA256"]==sha(q2d_dir/Q2D_REPORT)
          and proof["original_q2a_synthetic_source_SHA256"]==
              source_meta["synthetic_rows_SHA256"] and
          proof["actual_original_decoder_forward_calls"]==504 and
          proof["same_trainable_parameters_and_optimizer_A_B"]
          and proof["original_12layer_GreenTunnel_original_core_weights_unchanged"]
          and proof["old_S115_53751_FP16_neural_weights_unchanged"]
          and proof["candidate_accepted"] is False and proof["output_gate"]==0,
          "original q2f trained candidate and 504 physical forward proof only")
    old_ref=json.loads(reference.read_text(encoding="utf8"))
    check(old_ref["stage"]==Q2D1_STAGE and
          old_ref["q2d_real_research_source_report_SHA256"]==
              sha(q2d_dir/Q2D_REPORT)
          and old_ref["no_optimization_or_retraining"] and
          old_ref["candidate_accepted"] is False,
          "same original frozen real native source baseline provenance")
    parent,raw_parent=load_q2d_trained_child(
        q2d_dir/Q2D_CONTEXT_CKPT,
        parent_q2d["new_research_checkpoint_contextual"])
    ctx=load_physical_q2f_child(
        q2f_dir/Q2F_CTX,proof["projector_contextual_child_checkpoint"],
        parent)
    # Local receives its own identical-initial architecture; trained local
    # FP16 state entirely overwrites it, so there is no mutable shared child.
    parent_local,raw_parent_local=load_q2d_trained_child(
        q2d_dir/Q2D_CONTEXT_CKPT,
        parent_q2d["new_research_checkpoint_contextual"])
    local=load_physical_q2f_child(
        q2f_dir/Q2F_LOCAL,proof["projector_local_matched_checkpoint"],
        parent_local)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE,"original 49152 native vocab not proxy")
    train_raw,probe_raw=source_inventory(source_rows)
    _,probes=pack_sources(tok,train_raw,probe_raw)
    check(len(probes)==N_PROBES and
          Counter(x["source_split"] for x in probes)==SOURCE_SPLITS,
          "only 18 previous CAL + 12 previous source probes, not TRAIN")
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    original.model.add_module("nra3_s119m_onpath",ctx)
    original.eval()
    original_digest=old_original_param_digest(original)
    check(all(not p.requires_grad for name,p in original.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "original 12-layer S142 and GreenTunnel completely frozen")
    saved_context={k:v.detach().clone().cpu()
                   for k,v in ctx.state_dict().items()}
    saved_local={k:v.detach().clone().cpu()
                  for k,v in local.state_dict().items()}
    results={}
    counted_forwards=0
    for label,adapter,contextual in (
        ("contextual",ctx,True),("token_local",local,False)
    ):
        outcomes={}
        all_raw={}
        for cond in (True,False):
            trace=[]
            for row in probes:
                note,logits=fixed_native_source_case(
                    original,sources,bridge,full,expansion,adapter,row,
                    contextual=contextual,projector_on=cond)
                trace.append(note)
                all_raw[(cond,row["id"])]=logits
                counted_forwards+=1
            outcomes["ON" if cond else "OFF"]=trace
        result=compare_same_weights_onoff(outcomes["ON"],outcomes["OFF"])
        result["native_max_logit_shift_Linf_ON_OFF"]=max(
            float(torch.max(torch.abs(
                all_raw[(True,row["id"])]-all_raw[(False,row["id"])])))
            for row in probes
        )
        result["num_prompt_native_logits_changed_ON_OFF"]=sum(
            not torch.equal(
                all_raw[(True,row["id"])],all_raw[(False,row["id"])])
            for row in probes)
        previous=proof[
            "after_training_contextual_source_probe"
            if contextual else "after_training_local_source_probe"]
        check(abs(result["ON_mean_conditional_native_BPE_CE"]-
                  previous["mean_conditional_native_CE"])<0.0005,
              "FULL fitted projector ON must reproduce q2f original native CE")
        results[label]=result
    # Physically check that the full native original S142 is recovered
    # if all existing Core/NRA3/NPA1 output gates are zero, independent of
    # ON/OFF projection. 2 registered previously seen old probe IDs only.
    gate0=[]
    from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
        PLACEHOLDER_MAPPING,
    )
    for row in (probes[0],probes[-1]):
        pos=(row["first_distinct_position"] if row["source_family"]=="q2c1"
             else 0)
        prefix=row["prefix"]+row["completion_ids"][:pos]
        with temporary_gate_mask(ctx.child,MASKS["zero_exact_original"]):
            with temporarily_switch_projection(ctx,False):
                logits,witness=onepass_full_native_logits(
                    original,sources,bridge,full,expansion,ctx,
                    prefix,row["prompt_features"],contextual=True,
                    view_mapping=PLACEHOLDER_MAPPING)
                native_logits=logits.detach().cpu()
        measured=float(F.cross_entropy(
            native_logits,torch.tensor(
                [row["completion_ids"][pos]],dtype=torch.long)))
        old_item=next(z for z in old_ref[
            "full_original_source_native_baseline"]["source_details"]
            if z["source_id"]==row["id"])
        check(witness["candidate_forward_calls"]==1
              and witness["adapter_delta_H320"]==0
              and abs(measured-old_item["next_native_CE"])<0.0002,
              "ZERO gate + projector disabled EXACT original physical 43M native CE")
        counted_forwards+=1
        gate0.append({
            "source_id":row["id"],
            "native_original_frozen_CE":old_item["next_native_CE"],
            "zero_gate_ONOFF_natively_recovered_CE":measured,
        })
    check(counted_forwards==N_ORIGINAL_DECODER_FORWARDS and
          old_original_param_digest(original)==original_digest,
          "original physically unchanged after exact 122 decoder calls")
    for label,adapter,snapshot in (
        ("contextual",ctx,saved_context),("token_local",local,saved_local)
    ):
        check(adapter.projection_enabled and adapter.output_gate==0
              and not adapter.candidate_accepted
              and not adapter.activation_allowed and
              all(torch.equal(v,adapter.state_dict()[key].detach().cpu())
                  for key,v in snapshot.items()),
              "all old original trained q2f FP16 A/B weights exactly unchanged")
    report={
        "stage":STAGE,
        "original_S142_43M_GT_SHA256":sha(s142),
        "original_S93_SHA256":sha(s93),
        "q2f_real_trained_report_sha256":sha(q2f_dir/Q2F_REPORT),
        "original_q2d_contextual_checkpoint_sha256":sha(
            q2d_dir/Q2D_CONTEXT_CKPT),
        "original_q2a_source_sha256":source_meta["synthetic_rows_SHA256"],
        "original_12layer_43M_GT_and_old_53751_FP16_UNCHANGED":True,
        "physically_trained_q2f_A_B_FP16_children_UNCHANGED":True,
        "real_original_decoder_forward_calls":counted_forwards,
        "original_full_native_lm_head_vocab":VOCAB_SIZE,
        "source_only_result":results,
        "zero_gate_full_original_parity":gate0,
        "only_projector_ON_OFF_differs_within_identical_child":True,
        "projector_OFF_means_physical_actual_early_h5_repeats_only":True,
        "source_labels_not_in_backprop_or_hyperparam_selection":True,
        "source_probes_are_previously_exposed_not_BLIND":True,
        "no_natural_own_token_full_answer_generation":True,
        "no_616_J7_EOS_bpp_or_paired_KV_CPU_cert":True,
        "no_GOLD_S118_B_or_spent_EVAL128":True,
        "no_Jessica_chat_LIVE_Render_Supabase_main_APK":True,
        "no_optimization_or_new_trainable_weights":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"READONLY_CAUSAL_PROJECTOR_ON_OFF_SAME_WEIGHTS_NOT_QUALITY_CERT",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2F1_REAL43M_GT_SAME_TRAINED_CHILD_PROJECTOR_ABLATION "+
        json.dumps({
            "true_original_full_decoder_calls":counted_forwards,
            "contextual_ON_CE":
                results["contextual"]["ON_mean_conditional_native_BPE_CE"],
            "contextual_OFF_CE":
                results["contextual"]["OFF_mean_conditional_native_BPE_CE"],
            "contextual_projector_ON_MINUS_OFF_CE":
                results["contextual"]["ON_minus_OFF_mean_native_CE"],
            "local_projector_ON_MINUS_OFF_CE":
                results["token_local"]["ON_minus_OFF_mean_native_CE"],
            "contextual_native_greedy_changed":
                results["contextual"]["native_greedy_predictions_CHANGED"],
            "local_native_greedy_changed":
                results["token_local"]["native_greedy_predictions_CHANGED"],
            "original_weights_unchanged":True,
            "accepted":False,
        },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2a_dir","q2d_dir","q2f_dir",
                 "reference","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    run(a.s93,a.side,a.s142,a.q2a_dir,a.q2d_dir,a.q2f_dir,a.reference,a.out)


if __name__=="__main__":
    main()
