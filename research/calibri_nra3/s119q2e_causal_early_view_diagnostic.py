"""S119q2e: PHYSICAL causal early-layer semantic-view ablation, NO training.

Source diagnosis after failed semantic contextual advantage in q2d/d1.
Evaluate actual frozen ORIGINAL S142 12-layer+GT native 49152 LM on EXACT
30 previously exposed q2a CAL + q2c1 SOURCE_PROBE teacher-forced native
positions. Two real physically learned FP16 q2d children; two predeclared
CAUSALLY AVAILABLE early-state representations after original layer5:

REPEAT_H5: [h0,h1,h2,h3,h4,h5,h5,h5,h5,h5,h5,h5] (trained baseline)
MIRROR_H0_H5: [h0,h1,h2,h3,h4,h5,h0,h1,h2,h3,h4,h5] (NOT TRAINED)
No h6..11 values accessed before executing true frozen original late
layers6..11. Neither uses future output/label in the adapter.
Compare 30 x2 representation maps x2 A/B variants (=120 forward passes).
Supplement with 12 x3 physical leave-one-out Core/NRA3/NPA1 gate
component ablations and two no-modification zero-gate controls.
NO optimizer, no candidate acceptance, no independent quality claim.
A change under MIRROR is a source-only untrained input distribution shift
NOT a new competing semantic architecture. Never tune on these probes.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119q1_real43m_onepass_causal_early import (
    onepass_full_native_logits, PLACEHOLDER_MAPPING,
)
from research.calibri_nra3.s119p_real43m_onpath_component_ablation import (
    temporary_gate_mask,MASKS,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    verify_q2a_artifact,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    read_parent,source_inventory,pack_sources,VOCAB_SIZE,PROBE_COUNTS,
    REPORT as Q2D_REPORT,CONTEXT_CKPT,LOCAL_CKPT,
)
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    STAGE as REF_STAGE,REPORT as REF_REPORT,load_q2d_proof,
)

STAGE="CALIBRI_S119Q2E_REAL43M_GT_CAUSAL_EARLY_VIEW_AND_COMPONENT_DIAGNOSTIC"
REPORT="S119Q2E_REAL43M_GT_30SOURCE_CAUSAL_VIEW_GATES_DIAGNOSTIC.json"
MIRROR_H0_H5=(0,1,2,3,4,5,0,1,2,3,4,5)
MAPS={
    "REPEAT_H5":PLACEHOLDER_MAPPING,
    "MIRROR_H0_H5":MIRROR_H0_H5,
}
SOURCE_COUNT=30
COMPONENT_PROMPTS=12
EXPECTED_REAL_DECODER_CALLS=30*2*2 + 12*3 + 2


def source_probes(tok,q2a_rows):
    train_raw,probe_raw=source_inventory(q2a_rows)
    train,probe=pack_sources(tok,train_raw,probe_raw)
    check(len(train)==96 and len(probe)==30 and
          Counter(x["source_split"] for x in probe)==PROBE_COUNTS,
          "fixed exact already exposed 18 CAL and 12 SOURCE_PROBE")
    return probe


def trace_to_summary(trace):
    check(len(trace)==SOURCE_COUNT and
          Counter(x["source_split"] for x in trace)==PROBE_COUNTS,
          "precisely same 30 old probes, not user/chat/new evaluation")
    by_domain=defaultdict(list)
    for x in trace:
        check(0<=x["target_id"]<VOCAB_SIZE and
              x["source_kind"] in ("q2a","q2c1") and
              x["mask"]=="trained_full",
              "true original full-vocab native CE only")
        by_domain[x["kind"]].append(x["native_target_CE"])
    check(len(by_domain)==6 and all(len(v)==5 for v in by_domain.values()),
          "all exactly six original diagnostic source domains")
    return {
        "mean_native_source_only_CE":
            sum(x["native_target_CE"] for x in trace)/len(trace),
        "source_only_greedy_target_count":
            sum(x["native_greedy_target"] for x in trace),
        "mean_residual_H320_norm":
            sum(x["H320_residual_norm"] for x in trace)/len(trace),
        "per_domain_source_only_CE":{
            kind:sum(values)/len(values)
            for kind,values in sorted(by_domain.items())},
        "read_only_same_author_templates_not_independent":True,
    }


def extract_probe_native(original,sources,bridge,full,expansion,adapter,
                         item,*,contextual,mapping,mask="trained_full"):
    check(mask in MASKS and tuple(mapping) in MAPS.values(),
          "only preregistered source-only gate and causal early view")
    pos=(item["first_distinct_position"]
         if item["source_family"]=="q2c1" else 0)
    check(0<=pos<len(item["completion_ids"]) and
          item["source_split"] in PROBE_COUNTS,
          "read-only old source probe, no gradient labels exposed to training")
    causal_prefix=item["prefix"]+item["completion_ids"][:pos]
    target=int(item["completion_ids"][pos])
    with temporary_gate_mask(adapter,MASKS[mask]):
        logits,witness=onepass_full_native_logits(
            original,sources,bridge,full,expansion,adapter,
            causal_prefix,item["prompt_features"],
            contextual=contextual,view_mapping=mapping)
        native_logits=logits.detach().cpu()
    check(witness["candidate_forward_calls"]==1
          and witness["placeholder_mapping"]==list(mapping)
          and native_logits.shape==(1,VOCAB_SIZE),
          "ONE actual original decoder forward, ONLY causal h0-h5")
    loss=F.cross_entropy(
        native_logits,torch.tensor([target],dtype=torch.long))
    return {
        "kind":item["kind"],
        "source_id":item["id"],
        "source_split":item["source_split"],
        "source_kind":item["source_family"],
        "target_position":pos,
        "target_id":target,
        "mask":mask,
        "native_target_CE":float(loss.item()),
        "native_greedy_target":int(native_logits.argmax(-1))==target,
        "H320_residual_norm":float(witness["adapter_delta_H320"]),
        "native_predicted_id":int(native_logits.argmax(-1)),
        "source_prompt_sha256":item["prompt_sha256"],
        "full_NATIVE_head_classes":VOCAB_SIZE,
        "never_optimizer_steps":True,
    }


def check_previous_reference(reference_path,q2d_report,probe):
    check(reference_path.name==REF_REPORT,
          "exact old frozen original native same-source reference report")
    parent=json.loads(reference_path.read_text(encoding="utf8"))
    check(parent["stage"]==REF_STAGE and
          parent["q2d_real_research_source_report_SHA256"]==
             sha(q2d_report) and
          parent["original_true_12layer_all_30_full_native_physical_CPU_forwards"]
          and parent["no_optimization_or_retraining"]
          and parent["candidate_accepted"] is False,
          "previous original all30 physical comparator MUST have passed")
    baseline=parent["full_original_source_native_baseline"]
    check(len(baseline["source_details"])==len(probe)==SOURCE_COUNT,
          "all 30 previously frozen original native target prefix evidence")
    for earlier,item in zip(baseline["source_details"],probe):
        pos=(item["first_distinct_position"]
             if item["source_family"]=="q2c1" else 0)
        check(earlier["source_id"]==item["id"] and
              earlier["target_position"]==pos,
              "original frozen reference same exact source ID and position")
    return parent


def run(s93,side,s142,q2a_dir,q2d_dir,reference,out):
    check(not out.exists(),"new isolated diagnostic only; no overwriting")
    torch.manual_seed(119135)
    torch.set_num_threads(2)
    physical_report=load_q2d_proof(
        q2d_dir/Q2D_REPORT,q2d_dir/CONTEXT_CKPT,q2d_dir/LOCAL_CKPT,
        s93,s142)
    q2a_rows,manifest=verify_q2a_artifact(q2a_dir)
    check(physical_report["q2a_source_SHA256"]==
          manifest["synthetic_rows_SHA256"],
          "exact SHA-bound previous original source and trained evidence")
    (parent,(ctx,raw_ctx),(local,raw_local))=read_parent(
        q2d_dir/CONTEXT_CKPT,q2d_dir/LOCAL_CKPT,q2d_dir/Q2D_REPORT,
        s93,s142)
    check(parent==physical_report and
          set(ctx.state_dict())==set(local.state_dict()),
          "exact two physical trained A/B joint child states")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE,"true original 49152 native tokens")
    probe=source_probes(tok,q2a_rows)
    old_base=check_previous_reference(reference,q2d_dir/Q2D_REPORT,probe)
    original,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    original.model.add_module("nra3_s119m_onpath",ctx)
    original.eval()
    frozen_digest=old_original_param_digest(original)
    check(all(not p.requires_grad for name,p in original.named_parameters()
              if "nra3_s119m_onpath" not in name),
          "original S142 core-GT pretrained is COMPLETELY frozen")
    snapshots={
        "contextual":{k:v.detach().cpu().clone()
                     for k,v in ctx.state_dict().items()},
        "local":{k:v.detach().cpu().clone()
                 for k,v in local.state_dict().items()},
    }
    results={}
    n_actual=0
    for label,adapter,contextual in (
        ("contextual",ctx,True),("local",local,False)):
        results[label]={}
        for name,mapping in MAPS.items():
            trace=[]
            for ex in probe:
                trace.append(extract_probe_native(
                    original,sources,bridge,full,expansion,adapter,ex,
                    contextual=contextual,mapping=mapping))
                n_actual+=1
            summary=trace_to_summary(trace)
            results[label][name]={"summary":summary,"source_trace":trace}
        # Precisely the original trained REPEAT_H5 representation, no new
        # training; numerical parity with old q2d report near tolerance.
        q2d_source=physical_report[
            "source_diagnostic_contextual" if contextual
            else "source_diagnostic_local"]
        check(abs(results[label]["REPEAT_H5"]["summary"][
                  "mean_native_source_only_CE"]-
                  q2d_source["mean_source_only_native_CE"])<0.0005,
              "same already-trained original source probe CE must reproduce")
    ablations={}
    for name in ("without_core","without_structure","without_planning"):
        slice12=[]
        for ex in probe[-12:]:
            slice12.append(extract_probe_native(
                original,sources,bridge,full,expansion,ctx,ex,
                contextual=True,mapping=PLACEHOLDER_MAPPING,mask=name))
            n_actual+=1
        ablations[name]={
            "prompt_count":len(slice12),
            "source_only_mean_native_CE":
                sum(x["native_target_CE"] for x in slice12)/len(slice12),
            "source_only_residual_H320_norm":
                sum(x["H320_residual_norm"] for x in slice12)/len(slice12),
            "first12_source_traces":slice12,
        }
    # Zero-gate FULL physical native head parity to known frozen S142
    # matching original reference for the identical two original inputs.
    zero=[]
    for ex in (probe[0],probe[-1]):
        result=extract_probe_native(
            original,sources,bridge,full,expansion,ctx,ex,
            contextual=True,mapping=MIRROR_H0_H5,mask="zero_exact_original")
        earlier=next(x for x in old_base[
            "full_original_source_native_baseline"]["source_details"]
            if x["source_id"]==ex["id"])
        check(abs(result["native_target_CE"]-earlier["next_native_CE"])
              <0.0002 and
              result["H320_residual_norm"]==0.0,
              "real original full vocab CE ZERO-gate causal representation parity")
        zero.append(result)
        n_actual+=1
    check(n_actual==EXPECTED_REAL_DECODER_CALLS and
          old_original_param_digest(original)==frozen_digest and
          all(torch.equal(value,adapter.state_dict()[key].detach().cpu())
              for label,adapter in (("contextual",ctx),("local",local))
              for key,value in snapshots[label].items()),
          "full real physical decoder call count and BOTH adapters untouched")
    report={
        "stage":STAGE,
        "original_S93_SHA256":sha(s93),
        "original_S142_GT_SHA256":sha(s142),
        "parent_q2d_SHA256":sha(q2d_dir/Q2D_REPORT),
        "old_frozen_original_baseline_report_SHA256":sha(reference),
        "original_full_native_vocab":VOCAB_SIZE,
        "actual_original_decoder_forward_calls":n_actual,
        "original_neural_core_GreenTunnel_and_old_53751_FP16_UNCHANGED":True,
        "trained_parent_q2d_contextual_local_weights_UNCHANGED":True,
        "only_h0_h5_CAUSAL_ORIGINAL_states_in_onepass":True,
        "mirror_has_ZERO_original_future_h6_h11_before_layer6":True,
        "mirror_never_trained_no_new_weights":True,
        "gates_not_amplified_and_reverted_on_every_perturbation":True,
        "same_30_PREVIOUSLY_EXPOSED_old_source_probes_not_independent":True,
        "teacher_forced_target_prefix_not_natural_free_generation":True,
        "no_optimizer_training_or_model_promotion":True,
        "comparison":results,
        "three_component_leave_one_out":ablations,
        "two_ZERO_gate_physical_original_parity":zero,
        "no_new_blind_QA_or_616J7_EOS_replay":True,
        "no_packed_all_in_BPP_or_KV_CPU_benchmark":True,
        "no_Jessica_chat_LIVE_Render_Supabase_main_APK":True,
        "candidate_accepted":False,"output_gate":0,
        "decision":"RESEARCH_ONLY_DIAGNOSE_CAUSAL_INPUT_DISTRIBUTION_SHIFT_NOT_ACCEPTED",
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(json.dumps(
        report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2E_TRUE_ORIGINAL43M_CAUSAL_ABLATION "+json.dumps({
        "physical_native_decoder_forwards":n_actual,
        "source_only_CE":{key:{m:results[key][m]["summary"][
            "mean_native_source_only_CE"] for m in MAPS}
            for key in ("contextual","local")},
        "frozen_original_native_source_CE":old_base["summary"][
            "original_frozen_native_CE"],
        "unchanged_original_and_trained_children":True,
        "accepted":False,
    },sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser()
    for key in ("s93","side","s142","q2a_dir","q2d_dir","reference","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    x=p.parse_args()
    run(x.s93,x.side,x.s142,x.q2a_dir,x.q2d_dir,x.reference,x.out)


if __name__=="__main__":
    main()
