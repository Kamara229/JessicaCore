"""NRA2 3.0 S12c: ONE frozen holdout evaluation, never a TRAIN target.

This compares original S8b 43 real native CB59 C matrices vs the SAME
43 matrices + preexisting S11 physically paid and frozen 39xFP16 tiny 3way
router. Original Calibri 43M + GT EVAL16 features were read-only captured
in S12b AFTER S12 full EVAL labels/offsets SHA and model weights prereg.

No .backward(), optimizer, tuning, PR promotion, confidence-based
sample rejection, alternate model search, output gate or user chat.
Important: the router is downstream of grammar and semantic role heads,
so those exact upstream predictions CANNOT improve by adding this
router. Report their heldout accuracy and exact parity explicitly.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import (
    NRA3Config,UPOS,SEMANTIC_ROLES,
)
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord,CausalAlignment,
)
from research.nra3_s7.parallel_hypotheses import _ROLE_EDGES
from research.nra3_s11.paid_router import (
    PaidThreeWayRouter,PHYSICAL_BYTES,SOURCE_PARAMETERS,
)
from research.nra3_s12.heldout_ru_eval16 import (
    S12NovelEval,SPLIT,EVAL16_FULL_CANONICAL_SHA,
    build_eval16,sealed_eval16,
)

S12B_EVAL16_EXPECTED_FP16_SHA="UNPINNED_EXPECTED_S12B_PHYSICAL_SHA"
S11_ORIGINAL_39_FP16_SHA="05ce14e20c9f4e7e09f85dbd9bd46f0dc0a700d72f1a96ffa921b2e822f235c3"
S8b_ORIGINAL_FROZEN_43_CB59_SHA="d710aed710a44719b6003434a2dd83150861928ce167364cb92329beabf7be97"
S11_IMMUTABLE_SUCCESS_RUN=43734571081
S12_PRE_REGISTER_SUCCESS_RUN=43735385201
S12_PRE_REGISTER_FULL_SHA="f2d2d5beba48dbe294a90f6918a3fd15870aa103abbd569e907c44729a43977c"


def guard(ok:bool,reason:str):
    if not ok:
        raise ValueError("NRA3 S12c one-time presealed EVAL16 fail closed: "+reason)


@dataclass(frozen=True)
class SealedEvalSample:
    row:S12NovelEval
    hidden320:torch.Tensor
    aligned:CausalAlignment


def load_real_frozen_eval16(hidden:Path,report:Path)->tuple[
                            tuple[SealedEvalSample,...],dict]:
    hidden=Path(hidden);report=Path(report)
    guard(S12B_EVAL16_EXPECTED_FP16_SHA!="UNPINNED_EXPECTED_S12B_PHYSICAL_SHA",
          "must commit genuine read-only S12b FP16 SHA BEFORE first eval")
    guard(hidden.is_file() and report.is_file(),"missing true physical S12b source")
    sha=hashlib.sha256(hidden.read_bytes()).hexdigest()
    guard(sha==S12B_EVAL16_EXPECTED_FP16_SHA,
          "original physical 43M EVAL16 hidden is not SHA pinned")
    r=json.loads(report.read_text(encoding="utf-8"))
    required={
      "stage":"NRA3_S12B_TRUE_FROZEN_CALIBRI_43M_POST_GT_LAYER8_NEW_RUSSIAN_EVAL16_READONLY",
      "S12_RUSSIAN_UD_EVAL16_CANONICAL_SHA256":EVAL16_FULL_CANONICAL_SHA,
      "S12_PREREG_PHYSICAL_RUN_ID":S12_PRE_REGISTER_SUCCESS_RUN,
      "S12_EVAL16_PREREG_BEFORE_MODEL":True,
      "S12_S11_FROZEN_BEFORE_EVAL":True,
      "actual_fp16_feature_sha256":S12B_EVAL16_EXPECTED_FP16_SHA,
      "S194_ORIGINAL_CHECKPOINT_SHA256":"b1e58fde43bc9d4593f6db78f1c9b55d556255f4dcac7c96bce09302b68b34c2",
      "all_616_native_CPU_generated_token_digests_unchanged":True,
      "original_456_full_EOS":456,
      "original_new128_full_EOS":128,
      "real_source_eval16_count":16,
      "actual_genuine_43M_CPU_prefills":16,
      "observed_layer_index_zero_based":7,
      "GT_residuals_before_readport_in_same_active_S93_scope":True,
      "NRA3_output_gate":0.,
      "NRA3_candidate_accepted":False,
      "true_real_Calibri_token_alignment_all_16":True,
      "S12_eval_labels_loaded_into_Calibri_forward":False,
      "S12_eval16_scored":False,
      "S12_original_43M_EVAL16_only_after_SHA_pinning":True,
      "S12_all_16_prompts_are_new_EVAL_only":True,
      "no_training_or_optimizer_for_original_43M_and_GT":True,
      "all_gold_labels_excluded_from_frozen_decoder":True,
      "five_consumed_independent_EVAL128_not_loaded":True,
      "sixth_reserved_S3CI_EVAL128_not_loaded":True,
      "no_Render_Supabase_APK_LIVE_or_Jessica_chat":True,
    }
    guard(all(r.get(k)==v for k,v in required.items()),
          "S12b actual original 616 J7/GT/heldout source SHA mismatch")
    guard(EVAL16_FULL_CANONICAL_SHA==S12_PRE_REGISTER_FULL_SHA and
          sealed_eval16()["complete_annotation_sha256"]==S12_PRE_REGISTER_FULL_SHA,
          "full 16 UD+semantic labels must be preregistered immutable")
    tensors=load_file(str(hidden),device="cpu")
    guard(set(tensors)=={"token_hidden320","input_ids","token_mask","token_lengths"},
          "gold/labels/other artifact tensor forbidden")
    h=tensors["token_hidden320"];ids=tensors["input_ids"]
    mask=tensors["token_mask"];lengths=tensors["token_lengths"]
    guard(h.ndim==3 and h.shape[0]==16 and h.shape[-1]==320
          and h.dtype==torch.float16 and torch.isfinite(h).all()
          and ids.dtype==torch.int32 and mask.dtype==torch.bool
          and ids.shape==mask.shape==h.shape[:2]
          and lengths.shape==(16,) and
          list(h.shape)==r["actual_hidden320_shape"] and
          lengths.tolist()==r["true_eval16_token_counts"],
          "heldout original 43M CPU token FP16 ABI drift")
    rows=build_eval16()
    maps=r["wordpiece_alignment"]
    guard(len(rows)==16 and isinstance(maps,list) and len(maps)==16,
          "source or real-wordpiece row count changed")
    samples=[];ids_all=[]
    for i,row in enumerate(rows):
        n=int(lengths[i])
        guard(1<n<=h.shape[1] and
              torch.equal(mask[i],torch.arange(h.shape[1])<n) and
              bool((ids[i,:n]>=0).all()) and
              bool((ids[i,n:]==-1).all()) and
              bool((h[i,n:]==0).all()),
              "non-authentic token IDs or physical padded frozen 43M hidden")
        token_ids=tuple(int(x) for x in ids[i,:n].tolist())
        ids_all.append(list(token_ids))
        a=maps[i]
        guard(a["text_sha256"]==row.sha256() and a["token_count"]==n,
              "physical 16 frozen original EVAL text SHA/token identity drift")
        positions=a["word_token_positions"]
        lasts=a["word_last_token_positions"]
        guard(len(positions)==len(lasts)==len(row.words),
              "missing true original token word offsets")
        aligned_words=[];used=set()
        for j,(pos,last) in enumerate(zip(positions,lasts)):
            guard(isinstance(pos,list) and pos and
                  type(last) is int and last==max(pos)
                  and all(type(x) is int and 0<=x<n and x not in used
                          for x in pos),
                  "EVAL16 source wordpiece overlaps or future reference")
            used.update(pos)
            aligned_words.append(AlignedWord(j+1,tuple(pos),last))
        guard(all(aligned_words[j].last_token<aligned_words[j+1].last_token
                  for j in range(len(aligned_words)-1)),
              "unordered original Calibri wordpieces")
        aligned=CausalAlignment(row.sha256(),n,token_ids,
                                tuple(aligned_words),n-1,True)
        samples.append(SealedEvalSample(row,h[i,:n].float().contiguous(),aligned))
    packed_ids=json.dumps(ids_all,sort_keys=True,separators=(",",":")).encode()
    guard(hashlib.sha256(packed_ids).hexdigest()==
          r["actual_token_ids_lengths_sha256"],
          "true Calibri 43M original EVAL16 physical input IDs SHA changed")
    return tuple(samples),r


# Frozen S11 TRAIN heuristic is reproduced verbatim at EVAL *after* forward.
# EVAL labels never enter any native model, router forward or optimizer.
_EDGES=frozenset(tuple(sorted(pair)) for pair in _ROLE_EDGES)
def weak_eval_labels_posthoc(row:S12NovelEval,aligned:CausalAlignment)->torch.Tensor:
    allowed={r.sha256():r for r in build_eval16()}
    guard(row==allowed.get(row.sha256()) and row.split==SPLIT
          and aligned.independent_eval_labels_loaded and
          aligned.original_prompt_sha256==row.sha256() and
          len(aligned.word_alignment)==len(row.words),
          "new EVAL labels may only be used AFTER frozen native inference")
    positions=[w.last_token for w in aligned.word_alignment]
    classes=[]
    for i,(word,role) in enumerate(zip(row.words,row.roles)):
        comp=role!="NONE" and any(
            tuple(sorted((role,prior))) in _EDGES
            for prior in row.roles[:i])
        seen=0<word.head<=len(row.words) and
             positions[word.head-1]<=positions[i]
        near=seen and positions[i]-positions[word.head-1]<16
        classes.append(2 if comp else (1 if near else 0))
    return torch.tensor(classes,dtype=torch.long)


def pure_inference_word_grammar_score(native,features,aligned:CausalAlignment):
    """No labels, eval permitted only as inference; uses original paid CB59."""
    guard(aligned.independent_eval_labels_loaded and
          native.output_gate==0 and not native.candidate_accepted,
          "pure original frozen native EVAL readout only")
    ids=torch.tensor([w.last_token for w in aligned.word_alignment],
                     dtype=torch.long)
    last=aligned.complete_prefix_position
    gram=features.grammar[0,ids]+features.grammar[0,last]
    sem=features.semantics[0,ids]+features.semantics[0,last]
    q=native.linear("dep_q",gram)
    k=native.linear("dep_k",gram)
    root=native.linear("dep_k",features.grammar[0,last:last+1])
    dep=torch.cat((q@k.T,q@root.T),dim=1)/math.sqrt(native.config.hidden)
    return (native.linear("upos_head",gram).argmax(-1),
            native.linear("role_head",sem).argmax(-1),
            dep.argmax(-1))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eval16-hidden",required=True,type=Path)
    ap.add_argument("--eval16-report",required=True,type=Path)
    ap.add_argument("--s8-cb59",required=True,type=Path)
    ap.add_argument("--s11-router",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    guard(not args.out.exists() or not any(args.out.iterdir()),
          "refuse overwriting first ever heldout evaluation")
    torch.set_num_threads(2)
    samples,provenance=load_real_frozen_eval16(args.eval16_hidden,args.eval16_report)
    physical8=args.s8_cb59.read_bytes()
    physical11=args.s11_router.read_bytes()
    guard(len(physical8)==653798 and
          hashlib.sha256(physical8).hexdigest()==S8b_ORIGINAL_FROZEN_43_CB59_SHA
          and CB59(physical8).meta["original_unique_parameter_count"]==6_106_154
          and len(CB59(physical8).handles)==43,
          "model weights must be original rejected but physically authentic S8b")
    guard(len(physical11)==PHYSICAL_BYTES==123 and
          hashlib.sha256(physical11).hexdigest()==S11_ORIGINAL_39_FP16_SHA
          and PaidThreeWayRouter.from_physical(physical11).physical_bytes()==physical11,
          "locked S11 39 FP16 weights cannot be retuned or re-exported")
    args.out.mkdir(parents=True,exist_ok=True)
    lib=args.out/"lib_original_CB59_c_native_s12c.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(lib)],
                    check=True)
    v2=NRA3NativePackedCPU(physical8,NRA3Config(
        parallel_hypotheses=3,parallel_selector_version=2),lib).eval()
    v3=NRA3NativePackedCPU(physical8,NRA3Config(
        parallel_hypotheses=3,parallel_selector_version=3),lib,
        paid_router_physical=physical11).eval()
    guard(not list(v2.parameters()) and
          len([z for z in v3.parameters()])==2 and
          all(not z.requires_grad for z in v3.parameters()) and
          v2.source_sha256==v3.source_sha256==S8b_ORIGINAL_FROZEN_43_CB59_SHA
          and v2.native_cpu_weights_dense_resident is False
          and v3.native_cpu_weights_dense_resident is False and
          v2.output_gate==v3.output_gate==0 and
          not v2.candidate_accepted and not v3.candidate_accepted,
          "actual native C original 43 packed matrix C source must be immutable")
    # Forward BEFORE reading any EVAL labels, on all 16 prompts.
    predictions=[]
    with torch.inference_mode():
        for sample in samples:
            x=sample.hidden320.unsqueeze(0)
            a=v2(x)
            b=v3(x)
            ix=torch.tensor([w.last_token for w in sample.aligned.word_alignment])
            ga=a.parallel_hypothesis_gates[0,ix,:]
            gb=b.parallel_hypothesis_gates[0,ix,:]
            guard(ga.shape==gb.shape==(len(ix),3) and
                  torch.allclose(ga.sum(-1),torch.ones(len(ix)),atol=3e-5) and
                  torch.allclose(gb.sum(-1),torch.ones(len(ix)),atol=3e-5) and
                  torch.isfinite(a.relations).all() and
                  torch.isfinite(b.relations).all(),
                  "real original 43M native C EVAL 3way score nonfinite")
            # S11 router is attached AFTER both grammar and semantic layers:
            # the architectural limitation is exact, not an empirical win.
            guard(torch.equal(a.grammar,b.grammar) and
                  torch.equal(a.semantics,b.semantics) and
                  torch.equal(a.upos_logits,b.upos_logits) and
                  torch.equal(a.semantic_role_logits,b.semantic_role_logits) and
                  torch.equal(a.dep_logits,b.dep_logits),
                  "S11 changed upstream grammar or semantic heads unexpectedly")
            pos_a=pure_inference_word_grammar_score(v2,a,sample.aligned)
            pos_b=pure_inference_word_grammar_score(v3,b,sample.aligned)
            guard(all(torch.equal(u,v) for u,v in zip(pos_a,pos_b)),
                  "S11 relative to S8b MUST be identical upstream UD roles")
            predictions.append((ga.clone(),gb.clone(),pos_a,pos_b,
                                float((a.relations-b.relations).norm()),sample))
    # Gold EVAL opened ONLY after all 16 original native CPU forwards, no
    # optimizer steps or model re-entry after this point.
    gate0=[];gate1=[];teacher=[]
    pos_correct=role_correct=dep_correct=words=0
    per_group=[]
    for gates0,gates1,parts0,parts1,diff,sample in predictions:
        row=sample.row
        y=weak_eval_labels_posthoc(row,sample.aligned)
        gate0.append(gates0)
        gate1.append(gates1)
        teacher.append(y)
        p,r,d=parts0
        ud=torch.tensor([UPOS.index(w.upos) for w in row.words])
        semantic=torch.tensor([SEMANTIC_ROLES.index(v) for v in row.roles])
        head=torch.tensor([w.head-1 if w.head else len(row.words)
                           for w in row.words])
        guard(len(y)==len(ud)==len(p)==len(r)==len(d),
              "original UD word count/granularity mismatch")
        pc=int((p==ud).sum());rc=int((r==semantic).sum())
        dc=int((d==head).sum())
        words+=len(ud)
        pos_correct+=pc;role_correct+=rc;dep_correct+=dc
        per_group.append({
            "construction":row.construction,
            "word_count":len(ud),
            "UPOS_correct":pc,
            "semantic_roles_correct":rc,
            "dependency_head_correct":dc,
            "mean_relation_state_delta_norm":diff,
            "S8b_hard_selection":gates0.argmax(-1).tolist(),
            "S11_hard_selection":gates1.argmax(-1).tolist(),
        })
    g0=torch.cat(gate0);g1=torch.cat(gate1);y=torch.cat(teacher)
    guard(words==len(y) and len(per_group)==16
          and all(torch.isfinite(s).all() for s in (g0,g1))
          and set(y.tolist())=={0,1,2},
          "posthoc EVAL16 original CPU ground truth label validity")
    b0=g0.argmax(-1);b1=g1.argmax(-1)
    weak_before=float((b0==y).float().mean())
    weak_after=float((b1==y).float().mean())
    ce_before=float(F.nll_loss(g0.clamp_min(1e-8).log(),y))
    ce_after=float(F.nll_loss(g1.clamp_min(1e-8).log(),y))
    branches=("GLOBAL","LOCAL","PREDICTED_ROLE")
    result={
      "stage":"NRA3_S12C_ONE_TIME_NOVEL_RUSSIAN_EVAL16_FROZEN_S8B_VS_S11_NATIVE_C_AUDIT",
      "true_43M_GT_layer8_16_EVAL_feature_sha256":S12B_EVAL16_EXPECTED_FP16_SHA,
      "complete_EVAL16_annotation_preregistered_SHA":EVAL16_FULL_CANONICAL_SHA,
      "S12_prereg_completed_before_first_EVAL_model_run":True,
      "S11_model_router_train_frozen_before_EVAL":True,
      "S11_frozen_model_router_SHA":S11_ORIGINAL_39_FP16_SHA,
      "S8b_original_43_CB59_matrix_SHA":S8b_ORIGINAL_FROZEN_43_CB59_SHA,
      "eval_sentences":len(samples),
      "eval_total_words":words,
      "actual_original_Calibri_43M_CPU_after_GT_layer8_evaluated":True,
      "both_S8b_and_S11_full_native_C_original_43_matrices_executed":True,
      "all_16_original_hidden_tested_without_filtering":True,
      "S11_39_FP16_physically_charged_bytes":len(physical11),
      "original_NRA3_CB59_bytes_unchanged":len(physical8),
      "zero_eval_optimizer_steps":0,
      "no_eval_model_selection_or_hyperparameter_changes":True,
      "S8b_hard_selection_counts":{branches[k]:int((b0==k).sum()) for k in range(3)},
      "S11_hard_selection_counts":{branches[k]:int((b1==k).sum()) for k in range(3)},
      "weak_heuristic_EVAL16_targets_counts":{branches[k]:int((y==k).sum()) for k in range(3)},
      "S8b_weak_routing_heldout_accuracy":weak_before,
      "S11_weak_routing_heldout_accuracy":weak_after,
      "S8b_weak_routing_heldout_crossentropy":ce_before,
      "S11_weak_routing_heldout_crossentropy":ce_after,
      "S11_heldout_weak_selection_accuracy_change_pp":100*(weak_after-weak_before),
      "router_weak_heuristic_is_not_actual_semantic_truth":True,
      "UPOS_heldout_word_accuracy_same_baseline_and_S11":pos_correct/words,
      "semantic_role_heldout_word_accuracy_same_baseline_and_S11":role_correct/words,
      "dependency_head_heldout_word_accuracy_same_baseline_and_S11":dep_correct/words,
      "upstream_heads_unchanged_by_3way_router_by_architecture":True,
      "original_616_J7_frozen_baseline_matches_S194":True,
      "S11_active_original_616_J7_generation_measured":False,
      "full_answer_EOS_or_LIVE_communication_measured":False,
      "full_parent_43M_plus_GT_NRA_CB59_CPU_speed_ratio_measured":False,
      "whole_parent_GT_NRA_inclusive_bpp_le_point7_measured":False,
      "raw_standalone_NRA_plus_paid_router_bpp":8*(len(physical8)+len(physical11))/(6_106_154+SOURCE_PARAMETERS),
      "true_broad_language_understanding_or_answer_quality_proven":False,
      "five_previous_EVAL128_untouched":True,
      "original_sixth_S3CI_EVAL128_untouched":True,
      "no_ordinary_Jessica_chat_or_LIVE_or_Render_or_Supabase_or_APK":True,
      "output_gate":0,
      "candidate_accepted":False,
      "activation_allowed":False,
      "per_sentence_readonly_report":per_group,
    }
    out=args.out/"NRA3_S12C_ONE_TIME_NEW_RU16_EVAL_S8B_VS_S11_NO_PROMOTION.json"
    out.write_text(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False)+"\n",
                   encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items()
                      if k!="per_sentence_readonly_report"},
                      ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
