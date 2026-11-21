"""NRA3 S119d: true-original12 TRAIN-A, in-weight shared graph without new Q/K.

Physical SMALL research step, NOT a candidate model. Reuse existing S91+S103
NRA2 mini8, 12-layer router, Q/K, role bilinear, memory and morphology.
Drop S107 16,656 experimental duplicate-structural weights plus S112 8
trust scalars in an *isolated shadow derived checkpoint*, not in the original
trained S115 artifact. Reusing already-trained old neural Q/K to propagate
event states twice requires ZERO NEW trainable tensors or outside model.

This deliberately changes only the shadow role probability graph, not original
616 Calibri outputs. S115 previous structural TRAIN top1 did not improve;
nevertheless removing those weights may change or lose learned behavior.
No measured quality gain or unchanged NRA3 learning should be claimed.

Sources: genuine frozen S114 original CB59+Green Tunnel ALL12 native-BPE
SOURCE-ONLY TRAIN-A (NOT B), and unchanged trained S115 SHA checkpoint.
Use source TRAIN-A only. Independent S118 SynTagRus-B GOLD remains sealed.
No model loader, optimizer, original Calibri forward, LIVE or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119b_S194_parent_S115_child_physical_budget import (
    S115_EXPECTED_SHA256, audit, historical_projection,
)
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import (
    NRA3BoundedStructuralTrustInCalibriWeights,
)

STAGE = "NRA3_S119D_REAL12_SOURCE_A_37087_INWEIGHT_NEURAL_GRAPH_REUSE_GATE0"
PARENT_H_SHA = "e5b91042eae7311226f693222c557898caca27d6a3ec6f029574714a27f6d7ec"
PARENT_H = "NRA3_S114_SYNTAGRUS_TRAIN256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors"
PARENT_REPORT = "NRA3_S114_REAL43MGT_SYNTAGRUS_TRAIN256_ALLVERBS_SOURCE_PROVENANCE.json"
DERIVED_WEIGHTS = "NRA3_S119D_SHARED_EXISTING_37087_FP16_NO_NEW_MATRIX_UNACCEPTED.safetensors"
REPORT = "NRA3_S119D_REAL12_TRAIN_A_NRA2_MINI8_SHARED_GRAPH_AND_BUDGET.json"
PREFIX = "model.nra3_neural_weights."
DROPPED = frozenset({
    "structural_predicate_query.weight", "structural_argument_key.weight",
    "structural_role_bilinear", "structural_native_BPE_distance.weight",
    "structural_competition_gate", "structural_trust_gate",
})


def sha(path: Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition_full_state(checkpoint: dict) -> tuple[dict, dict]:
    check(len(checkpoint) >= 50, "full trained S115 neural child was not loaded")
    old, discarded = {}, {}
    for key,value in checkpoint.items():
        check(key.startswith(PREFIX) and value.dtype == torch.float16,
              "only S115 in-Calibri native FP16 neural tensors allowed")
        name=key[len(PREFIX):]
        check(name not in old and name not in discarded, "duplicated state key")
        if name in DROPPED:
            discarded[name] = value
        else:
            old[name] = value
    check(set(discarded) == DROPPED and
          sum(v.numel() for v in discarded.values())==16_664 and
          sum(v.numel() for v in old.values())==37_087 and
          sum(v.numel() for v in checkpoint.values())==53_751,
          "all 6 old structural/trust tensors must be removed, 37,087 retained")
    check(len([x for x in old if x.startswith("mini_areas.")])==32,
          "eight independent original NRA2 mini-areas must remain")
    for key in ("layer_router_logits","shared_hidden_projector.weight",
                "role_query.weight","role_key.weight","role_bilinear",
                "event_memory_slots","memory_query.weight","role_morph_affinity"):
        check(key in old, "neural routing / QK / memory not preserved: "+key)
    return old,discarded


@torch.no_grad()
def two_hop_parameter_free_event_relation_graph(
        original_graph, original_valid):
    """Reuses learned OLD role-arcs and neural state probabilities, no params.

    Two message-passing hops performed on the same existing OLD
    role-attention [B,8,T,T]. Parameter-free mixing is a RESEARCH
    hypothesis, not an already-trained brain. Source-only, no ID/gold.
    """
    valid=original_valid
    logits=original_graph.role_arc_logits.float()
    states=original_graph.event_state_logits.float()
    check(logits.ndim==4 and logits.shape[1]==8 and
          states.shape[:2]==valid.shape and states.shape[-1]==6 and
          logits.shape[0]==valid.shape[0] and logits.shape[-1]==valid.shape[1],
          "existing source-only event graph ABI")
    # Invalid argument positions are suppressed; no future gold or teacher.
    attn=torch.softmax(logits,dim=-1)
    attn=attn * valid[:,None,:,None] * valid[:,None,None,:]
    attn=attn / attn.sum(-1,keepdim=True).clamp_min(1.e-9)
    current=torch.softmax(states,dim=-1) * valid[...,None]
    one_hop=torch.einsum("brts,bsc->brtc",attn,current).mean(dim=1)
    updated=(current+one_hop)*0.5*valid[...,None]
    two_hop=torch.einsum("brts,bsc->brtc",attn,updated).mean(dim=1)
    combined=(updated+two_hop)*0.5*valid[...,None]
    check(torch.isfinite(combined).all() and
          combined.shape==current.shape and
          bool((combined[~valid]==0).all()) and
          (combined[valid].sum(-1)>0).all(),
          "two-hop graph must be finite and never leak into invalid tokens")
    return combined


def run(s194:Path,s115:Path,s115_report:Path,
        source:Path,source_report:Path,
        out:Path, *, max_rows:int=8) -> dict:
    check(1<=max_rows<=16 and not out.exists(),
          "single small TRAIN-A experiment only / cannot overwrite")
    check(source.name==PARENT_H and source_report.name==PARENT_REPORT,
          "only frozen genuine S114 TRAIN-A source ABI")
    check(sha(source)==PARENT_H_SHA and sha(s115)==S115_EXPECTED_SHA256,
          "literal S114 true12 source and S115 full child SHA must be immutable")
    old_proof=audit(s194,s115,s115_report)
    s114=json.loads(source_report.read_text(encoding="utf8"))
    check(s114.get("ALL12_CPU_SAFETENSORS_SHA256")==PARENT_H_SHA and
          s114.get("old_616_J7_physically_replayed_BEFORE_NRA3_hooks") is True and
          s114.get("output_gate")==0 and
          s114.get("candidate_accepted") is False and
          s114.get("new_NRA3_train_steps")==0,
          "S114 source old original12 and 616 baseline before TRAIN-only capture")
    fp16=load_file(str(s115),device="cpu")
    frozen,removed=partition_full_state(fp16)
    old_hash={k:hashlib.sha256(v.numpy().tobytes()).hexdigest()
              for k,v in frozen.items()}
    raw=load_file(str(source),device="cpu")
    h=raw["original_Calibri43M_GT_ALL12_native_H320_FP16_READONLY"]
    v=raw["native_BPE_source_valid_mask"]
    rows=raw["source_row_indices_in_PINNED_SYNTAGRUS_TRAIN256"]
    check(h.ndim==4 and h.shape[0]==256 and h.shape[1]==12 and
          h.shape[-1]==320 and h.dtype==torch.float16 and
          v.shape==h.shape[:1]+h.shape[2:3] and v.dtype==torch.bool and
          rows.dtype==torch.int32 and
          torch.equal(rows,torch.arange(256,dtype=torch.int32)),
          "genuine all12 TRAIN-A H320 and native BPE row ID alignment")
    shared=NRA3RelationalMorphologyInCalibriWeights().eval()
    shared.load_state_dict({k:v.float() for k,v in frozen.items()},strict=True)
    full=NRA3BoundedStructuralTrustInCalibriWeights().eval()
    full.load_state_dict({k[len(PREFIX):]:value.float()
                          for k,value in fp16.items()},strict=True)
    check(shared.budget()["new_neural_parameters"]==37_087 and
          full.budget()["new_neural_parameters"]==53_751,
          "reuse exactly 37,087 OLD trained weights; no new parameters")
    for m in (shared,full):
        check(m.output_gate==0 and m.candidate_accepted is False and
              m.activation_allowed is False,
              "must remain unaccepted and disconnected from decoder")
    source_digests=[]
    graph_row_results=[]
    original_all8_mini_exact=True
    original_morph_exact=True
    role_delta_counts=[]
    graph_seconds=0.
    torch.set_num_threads(2)
    for i in range(max_rows):
        length=int(v[i].sum())
        check(length>=2 and bool(v[i,:length].all()) and
              not bool(v[i,length:].any()),
              "native length BPE contiguous and source-only")
        src=h[i:i+1,:,:length].float().detach()
        valid=v[i:i+1,:length]
        source_digests.append(hashlib.sha256(src.numpy().tobytes()).hexdigest())
        with torch.inference_mode():
            previous=NRA3RelationalMorphologyInCalibriWeights.forward(
                shared,src,valid)
            old_full=full(src,valid)
            minis=torch.equal(previous.mini_area_logits,old_full.mini_area_logits)
            morph=all(torch.equal(getattr(previous, attr),getattr(old_full,attr))
                      for attr in ("upos_logits","case_logits",
                                   "number_logits","voice_logits"))
            original_all8_mini_exact &= minis
            original_morph_exact &= morph
            before=time.perf_counter()
            proposed=two_hop_parameter_free_event_relation_graph(
                previous,valid)
            graph_seconds+=time.perf_counter()-before
            delta=int((previous.role_arc_logits.argmax(-1)!=
                       old_full.role_arc_logits.argmax(-1)).sum().item())
            role_delta_counts.append(delta)
            graph_row_results.append({
                "original_native_BPE_tokens":length,
                "original_eight_mini_areas_exact":bool(minis),
                "all_four_original_neural_morph_heads_exact":bool(morph),
                "S115_structural_vs_shared_original_top1_flip_positions_NO_GOLD":delta,
                "derived_2hop_state_shape":list(proposed.shape),
                "exact_S114_FROZEN_12layer_source_SHA256":source_digests[-1],
            })
    check(original_all8_mini_exact and original_morph_exact,
          "all eight old NRA2 mini-heads and neural morphology must be bitwise identical")
    check({k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in frozen.items()}==
          old_hash,"inherited original FP16 weights changed")
    out.mkdir(parents=True)
    subset=out/DERIVED_WEIGHTS
    save_file({k:frozen[k].contiguous() for k in sorted(frozen)},str(subset))
    reloaded=load_file(str(subset),device="cpu")
    check(set(reloaded)==set(frozen) and
          all(torch.equal(reloaded[k],frozen[k]) for k in frozen),
          "physically persisted 37,087 shared QK + original 8 mini weights must roundtrip exact")
    parent_bpp=old_proof["budget_projection"][
        "s194_reported_parent_bpp_on_historical_43261760_denominator"]
    proposed_budget=historical_projection(
        parent_bpp=parent_bpp,physical_child_bytes=subset.stat().st_size)
    report={
        "stage":STAGE,
        "S114_original_all12_SynTagRusA_TRAIN_ONLY_source_SHA256":PARENT_H_SHA,
        "S115_trained_full_53751_source_SHA256":S115_EXPECTED_SHA256,
        "S119D_repack_full_37087_OLD_neural_child_SHA256":sha(subset),
        "old_S115_inherited_37087_weights_FP16_exact":True,
        "original_NRA2_eight_mini_areas_exact_on_REAL_original12":bool(original_all8_mini_exact),
        "original_four_morphology_heads_exact_on_REAL_original12":bool(original_morph_exact),
        "original_12layer_router_and_role_QK_and_memory_REUSED":True,
        "discarded_S110_structural_weight_count":16_656,
        "discarded_S115_trust_weight_count":8,
        "new_trainable_parameters_in_two_hop_graph":0,
        "inherited_trainable_neural_parameters":37_087,
        "full_original_child_trainable_neural_parameters":53_751,
        "true_S114_TRAIN_A_rows_seen_without_GOLD":max_rows,
        "genuine_original_BPE_count":sum(x["original_native_BPE_tokens"] for x in graph_row_results),
        "existing_S115_vs_shared_role_top1_flips_UNLABELED":sum(role_delta_counts),
        "rows":graph_row_results,
        "two_hop_graph_kernel_only_seconds_NOT_FullDecode_CPU":graph_seconds,
        "shared_FP16_checkpoint_file_bytes":subset.stat().st_size,
        "original_s115_FP16_checkpoint_file_bytes":s115.stat().st_size,
        "FP16_file_bytes_saved_without_proof_of_quality":
            s115.stat().st_size-subset.stat().st_size,
        "physical_budget_projection_REQUIRES_LOADER_AUDIT":proposed_budget,
        "S118_SynTagRusB_GOLD_never_opened":True,
        "no_new_teacher_or_LABEL_ever_used":True,
        "old_616_original_calibri_replayed_only_in_prior_S114_source_run":True,
        "new_616_candidate_not_replayed":True,
        "new_CPU_full_decode_NOT_benchmarked":True,
        "independent_quality_unproven":True,
        "previous_16_664_structural_weights_discarded_in_shadow_MAY_LOSE_LEARNED_BEHAVIOR":True,
        "original_model_and_GreenTunnel_unchanged":True,
        "model_accepted":False,
        "activation_allowed":False,
        "output_gate":0,
        "decision":"SHARED_GRAPH_PHYSICAL_SHADOW_ONLY_NOT_ACCEPTED",
    }
    (out/REPORT).write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("S119D_SHARED_INWEIGHT_PHYSICAL_SOURCE_AUDIT "+
          json.dumps({
              "source_tokens":report["genuine_original_BPE_count"],
              "s115_bytes":report["original_s115_FP16_checkpoint_file_bytes"],
              "shared_child_bytes":report["shared_FP16_checkpoint_file_bytes"],
              "saved":report["FP16_file_bytes_saved_without_proof_of_quality"],
              "legacy_bpp":proposed_budget["projected_legacy_bpp"],
              "conservative_bpp":proposed_budget["projected_conservative_bpp"],
              "all8_miniareas_exact":report["original_NRA2_eight_mini_areas_exact_on_REAL_original12"],
              "candidate_accepted":False},sort_keys=True),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    for name in ("s194","s115","s115_report","source","source_report","out"):
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    p.add_argument("--rows",type=int,default=8)
    a=p.parse_args()
    run(a.s194,a.s115,a.s115_report,a.source,a.source_report,a.out,
        max_rows=a.rows)


if __name__=="__main__":
    main()
