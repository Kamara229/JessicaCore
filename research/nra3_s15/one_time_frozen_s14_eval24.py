"""NRA2 3.0 S15c: FIRST and ONLY true 43M CPU Russian EVAL24 S14 audit.

This is a precommitted, entirely frozen physically paid 43 CB59 native-C
and 39-parameter S11-router plus S14 1,266-FP16 relation readout test.
No learning, optimizer, checkpoint selection, output activation, chat,
S12c EVAL16 reuse, old EVAL128 or reserved S3CI data.

CRITICAL: S12c earlier UPOS/role scores added end-of-sentence grammar
to earlier words and were NOT causal. S15c instead evaluates *only*
original native readout logits at each word's own LAST true Calibri
token. Dependency head only when actual head precedes dependent; the
native causal T×T logits do NOT have a trained ROOT class. Do not
mistake these restricted-head metrics for complete syntax parsing.

The previous S14 24 internal rows were exposed to S11 training, and
the new 24 EVAL sentences are small hand-authored examples. Even a
positive measured gain is not certified broad natural language.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from safetensors.torch import load_file

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config,UPOS,SEMANTIC_ROLES
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord,CausalAlignment,
)
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s13.paid_relation_feedback import (
    PaidRelationFeedback,PHYSICAL_BYTES,PARAMETERS,UPOS_DIM,
)
from research.nra3_s15.new_ru_eval24_prereg import (
    EVAL24_CANONICAL_SHA,new_russian_eval24,sealed_new_eval24,SPLIT,
)

S15B_SOURCE_RUN=43742040678
S15B_TRUE43M_HIDDEN_FP16_SHA="fc4b438929d90e56ce8edc8743836fdb8f609bfaea228944d71723453227b9a5"
S15_SOURCE_RUN=43741334688
S14_FROZEN_RUN=43740415883
S14_PAID_FP16_SHA="5b96e2174e33da5232fcc2a4a397b6fab4370b6879adb684d75fea8a503f1279"
S8B_CB59_SHA="d710aed710a44719b6003434a2dd83150861928ce167364cb92329beabf7be97"
S11_ROUTER_SHA="05ce14e20c9f4e7e09f85dbd9bd46f0dc0a700d72f1a96ffa921b2e822f235c3"
S15_EVAL24_SHA="640bdfb0d817d1467eaf308fc1ba5dd9043361721af293b9666d89f4a2c7276d"
S194_ORIGINAL_CHECKPOINT_SHA="b1e58fde43bc9d4593f6db78f1c9b55d556255f4dcac7c96bce09302b68b34c2"


def guard(yes,reason:str):
    if not bool(yes):
        raise ValueError("NRA3 S15c original43M one-time EVAL24 fail closed: "+reason)


def sha_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Input24:
    row:object
    hidden320:torch.Tensor
    alignment:CausalAlignment


def read_frozen_original43m_eval24(hidden:Path,report:Path)->tuple[tuple[Input24,...],dict]:
    guard(EVAL24_CANONICAL_SHA==S15_EVAL24_SHA,
          "old S15 complete UD/roles/negation/coref prereg changed")
    guard(hidden.is_file() and report.is_file(),
          "real original Calibri43M EVAL24 physical inputs required")
    guard(sha_bytes(hidden.read_bytes())==S15B_TRUE43M_HIDDEN_FP16_SHA,
          "true original43M+GT native-CPU post-layer8 FP16 SHA changed")
    r=json.loads(report.read_text(encoding="utf-8"))
    locked={
      "stage":"NRA3_S15B_TRUE_FROZEN_CALIBRI_43M_POST_GT_LAYER8_NEW_RUSSIAN_EVAL24_READONLY",
      "actual_fp16_feature_sha256":S15B_TRUE43M_HIDDEN_FP16_SHA,
      "S15_RUSSIAN_UD_EVAL24_CANONICAL_SHA256":EVAL24_CANONICAL_SHA,
      "S15_PREREG_PHYSICAL_RUN_ID":S15_SOURCE_RUN,
      "S15_EVAL24_PREREG_BEFORE_MODEL":True,
      "S15_S14_FROZEN_BEFORE_EVAL":True,
      "S15_S14_FROZEN_1266FP16_READOUT_SHA256":S14_PAID_FP16_SHA,
      "S15_S14_NRA_READOUT_NOT_LOADED":True,
      "S15_model_selection_on_EVAL24":False,
      "S15_coreference_event_role_gold_NEVER_passed_to_model":True,
      "S15_eval_labels_loaded_into_Calibri_forward":False,
      "S15_eval24_scored":False,
      "S15_original_43M_EVAL24_only_after_SHA_pinning":True,
      "S15_all_24_prompts_are_new_EVAL_only":True,
      "S194_ORIGINAL_CHECKPOINT_SHA256":S194_ORIGINAL_CHECKPOINT_SHA,
      "all_616_native_CPU_generated_token_digests_unchanged":True,
      "original_456_full_EOS":456,
      "original_new128_full_EOS":128,
      "real_source_eval24_count":24,
      "actual_genuine_43M_CPU_prefills":24,
      "observed_layer_index_zero_based":7,
      "GT_residuals_before_readport_in_same_active_S93_scope":True,
      "NRA3_output_gate":0.,
      "NRA3_candidate_accepted":False,
      "true_real_Calibri_token_alignment_all_24":True,
      "no_training_or_optimizer_for_original_43M_and_GT":True,
      "all_gold_labels_excluded_from_frozen_decoder":True,
      "five_consumed_independent_EVAL128_not_loaded":True,
      "sixth_reserved_S3CI_EVAL128_not_loaded":True,
      "no_Render_Supabase_APK_LIVE_or_Jessica_chat":True,
    }
    guard(all(r.get(k)==v for k,v in locked.items()),
          "S15b CPU 616 J7, S15 locked precommit, model or gold provenance drift")
    guard(sealed_new_eval24()[
       "complete_UD_semantic_role_coreference_event_gold_sha256"]==S15_EVAL24_SHA,
       "post-S14 frozen complete 24 row prereg changed")
    tensor=load_file(str(hidden),device="cpu")
    guard(set(tensor)=={"token_hidden320","input_ids","token_mask","token_lengths"},
          "gold/source label hidden in original43M model physical input")
    h=tensor["token_hidden320"]; ids=tensor["input_ids"]
    masks=tensor["token_mask"]; lengths=tensor["token_lengths"]
    guard(h.ndim==3 and h.shape[0]==24 and h.shape[2]==320 and
          h.dtype==torch.float16 and bool(torch.isfinite(h).all())
          and ids.dtype==torch.int32 and masks.dtype==torch.bool
          and ids.shape==masks.shape==h.shape[:2]
          and lengths.shape==(24,) and
          list(h.shape)==r["actual_hidden320_shape"] and
          lengths.tolist()==r["true_eval24_token_counts"],
          "original CPU hidden320/token-ID physical ABI incorrect")
    rows=new_russian_eval24()
    maps=r["wordpiece_alignment"]
    guard(len(rows)==len(maps)==24 and all(row.split==SPLIT for row in rows),
          "wrong new EVAL24 text set or reused old heldout")
    result=[];all_ids=[]
    for i,(row,mapping) in enumerate(zip(rows,maps)):
        n=int(lengths[i])
        guard(1<n<=h.shape[1] and
              torch.equal(masks[i],torch.arange(h.shape[1])<n) and
              bool((ids[i,:n]>=0).all()) and
              bool((ids[i,n:]==-1).all()) and
              bool((h[i,n:]==0).all()),
              "noncanonical native Calibri token ID/padding")
        true_ids=tuple(int(v) for v in ids[i,:n].tolist())
        all_ids.append(list(true_ids))
        guard(mapping["text_sha256"]==row.sha256() and
              mapping["token_count"]==n and
              len(mapping["word_token_positions"])==
              len(mapping["word_last_token_positions"])==len(row.words),
              "SHA-locked new EVAL24 exact wordpiece offset identity changed")
        aligned=[];used=set()
        for j,(pieces,last) in enumerate(zip(mapping["word_token_positions"],
                                             mapping["word_last_token_positions"])):
            guard(type(pieces) is list and pieces and
                  type(last) is int and last==max(pieces)
                  and all(type(q) is int and 0<=q<n and q not in used
                          for q in pieces),
                  "noncausal/overlapping original wordpiece mapping")
            used.update(pieces)
            aligned.append(AlignedWord(j+1,tuple(pieces),last))
        guard(all(aligned[j].last_token<aligned[j+1].last_token
                  for j in range(len(aligned)-1)),
              "incorrect original left-to-right wordpiece token order")
        a=CausalAlignment(row.sha256(),n,true_ids,
                          tuple(aligned),n-1,True)
        result.append(Input24(row,h[i,:n].float().contiguous(),a))
    packed=json.dumps(all_ids,sort_keys=True,separators=(",",":")).encode()
    guard(sha_bytes(packed)==r["actual_token_ids_lengths_sha256"],
          "genuine original CPU 24 EVAL input token ID SHA changed")
    return tuple(result),r


def physical_sources(cb59:bytes,router:bytes,adapter:bytes):
    guard(len(cb59)==653798 and sha_bytes(cb59)==S8B_CB59_SHA,
          "source original 43 CB59 physical 5bit matrices changed")
    orig=CB59(cb59)
    guard(len(orig.handles)==43 and
          orig.meta["original_unique_parameter_count"]==6_106_154,
          "missing original true 6M C kernels")
    guard(len(router)==123 and sha_bytes(router)==S11_ROUTER_SHA and
          PaidThreeWayRouter.from_physical(router).physical_bytes()==router,
          "original S11 frozen 39-FP16 3way selector mutated")
    guard(len(adapter)==PHYSICAL_BYTES==2581 and
          sha_bytes(adapter)==S14_PAID_FP16_SHA,
          "S14 rank8 1,266-FP16 model changed AFTER EVAL source prereg")
    fb=PaidRelationFeedback.from_physical(adapter).eval()
    guard(sum(p.numel() for p in fb.parameters())==PARAMETERS==1266
          and all(not p.requires_grad for p in fb.parameters())
          and bool((fb.output_weight[:UPOS_DIM]==0).all())
          and bool((fb.output_bias[:UPOS_DIM]==0).all())
          and fb.output_gate==0 and fb.candidate_accepted is False,
          "S14 physically serialized zero-UPOS preservation not exact")
    return fb


def word_gold(row,positions:list[int],pre):
    """Use ONLY after all new EVAL24 native forward passes complete.

    Past-only dependency model has no ROOT/future-governor class. Gold
    for those must not be treated as predictions, hence denominator
    records the actual causally observable subset only.
    """
    gold_u=torch.tensor([UPOS.index(w.upos) for w in row.words])
    gold_r=torch.tensor([SEMANTIC_ROLES.index(s) for s in row.roles])
    guard(len(gold_u)==len(gold_r)==len(positions)==len(pre),
          "UD word/role/native inference output row mismatch")
    candidate=[]
    for i,w in enumerate(row.words):
        if 0<w.head<=i:
            candidate.append((i,w.head-1))
    return gold_u,gold_r,candidate


def one_time_metrics(predictions)->dict:
    """Only once after all 24 forwards; no further model inference."""
    base_u=base_r=s14_u=s14_r=0
    words=past_count=base_h=s14_h=0
    s8b_semantic_same=True
    per_case=[]
    by_tag={}
    for item in predictions:
        row=item["row"];positions=item["positions"]
        gold_u,gold_r,observed=word_gold(row,positions,item["base_role"])
        bu=item["base_upos"].argmax(-1)
        br=item["base_role"].argmax(-1)
        au=item["adapter_upos"].argmax(-1)
        ar=item["adapter_role"].argmax(-1)
        guard(torch.equal(item["base_upos"],item["adapter_upos"]),
              "S14 exact physical zero UPOS must preserve every score")
        guard(torch.equal(item["base_upos"],item["S8b_upos"]) and
              torch.equal(item["base_role"],item["S8b_role"]),
              "frozen selector cannot alter prior grammar or semantics")
        n=len(gold_u)
        bu_ok=int((bu==gold_u).sum())
        br_ok=int((br==gold_r).sum())
        au_ok=int((au==gold_u).sum())
        ar_ok=int((ar==gold_r).sum())
        guard(bu_ok==au_ok,"upos-locked must yield exact baseline hard predictions")
        base_u+=bu_ok;base_r+=br_ok;s14_u+=au_ok;s14_r+=ar_ok
        words+=n
        this_base_head=this_s14_head=0
        for i,prior in observed:
            assert prior<i
            a=item["base_dependency"][i,positions[:i+1]]
            b=item["adapter_dependency"][i,positions[:i+1]]
            this_base_head+=int(a.argmax().item()==prior)
            this_s14_head+=int(b.argmax().item()==prior)
        past_count+=len(observed)
        base_h+=this_base_head;s14_h+=this_s14_head
        record={
            "construction":row.construction,
            "new_eval_one_time_words":n,
            "true_original_Calibri_last_token_wordpieces":positions,
            "baseline_upos_correct":bu_ok,
            "S14_upos_correct":au_ok,
            "baseline_semantic_roles_correct":br_ok,
            "S14_semantic_roles_correct":ar_ok,
            "past_governor_visible_head_cases":len(observed),
            "baseline_past_governors_correct":this_base_head,
            "S14_past_governors_correct":this_s14_head,
            "gold_negation_future_event_truth_not_predicted_by_S14":row.event_truth,
            "gold_coreference_edges_not_predicted_by_S14_count":len(row.antecedents)
        }
        per_case.append(record)
        tag=by_tag.setdefault(row.event_truth,{"sentences":0,"words":0,
               "baseline_role_correct":0,"S14_role_correct":0})
        tag["sentences"]+=1;tag["words"]+=n
        tag["baseline_role_correct"]+=br_ok
        tag["S14_role_correct"]+=ar_ok
    guard(len(per_case)==24 and words>=100 and past_count>=20,
          "all EVAL24 gold/language scenarios must be scored without filtering")
    return {
       "eval24_words":words,
       "causally_observable_past_governor_edges_only":past_count,
       "baseline_UPOS_correct":base_u,
       "S14_UPOS_correct":s14_u,
       "baseline_UPOS_accuracy":base_u/words,
       "S14_UPOS_accuracy":s14_u/words,
       "S14_UPOS_delta_correct":s14_u-base_u,
       "baseline_semantic_roles_correct":base_r,
       "S14_semantic_roles_correct":s14_r,
       "baseline_semantic_roles_accuracy":base_r/words,
       "S14_semantic_roles_accuracy":s14_r/words,
       "S14_roles_delta_correct":s14_r-base_r,
       "baseline_past_governors_correct":base_h,
       "S14_past_governors_correct":s14_h,
       "baseline_past_governor_accuracy":base_h/past_count,
       "S14_past_governor_accuracy":s14_h/past_count,
       "S14_past_governor_delta_correct":s14_h-base_h,
       "event_truth_gold_groups_not_predicted":by_tag,
       "per_sentence_readonly_gold_analysis":per_case
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--eval24-hidden",required=True,type=Path)
    ap.add_argument("--eval24-report",required=True,type=Path)
    ap.add_argument("--s8-cb59",required=True,type=Path)
    ap.add_argument("--s11-router",required=True,type=Path)
    ap.add_argument("--s14-adapter",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    guard(not a.out.exists(),
          "S15c one-time scores must never overwrite an existing run")
    torch.set_num_threads(2)
    samples,origin=read_frozen_original43m_eval24(
       a.eval24_hidden,a.eval24_report)
    b=a.s8_cb59.read_bytes()
    c=a.s11_router.read_bytes()
    d=a.s14_adapter.read_bytes()
    fb=physical_sources(b,c,d)
    a.out.mkdir(parents=True,exist_ok=False)
    lib=a.out/"s15c_original_43_CB59_C_native.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
        "research/calibri/direct_kernel_v59.c","-o",str(lib)],check=True)
    baseline=NRA3NativePackedCPU(b,NRA3Config(
        parallel_hypotheses=3,parallel_selector_version=2),lib).eval()
    s11=NRA3NativePackedCPU(b,NRA3Config(
        parallel_hypotheses=3,parallel_selector_version=3),lib,
        paid_router_physical=c).eval()
    guard(not list(baseline.parameters()) and
          not baseline.native_cpu_weights_dense_resident and
          not s11.native_cpu_weights_dense_resident and
          baseline.source_sha256==s11.source_sha256==S8B_CB59_SHA
          and s11.paid_router.physical_bytes()==c and
          baseline.output_gate==s11.output_gate==fb.output_gate==0 and
          not baseline.candidate_accepted and not s11.candidate_accepted
          and not fb.candidate_accepted and
          all(not p.requires_grad for p in s11.parameters()),
          "true original 43 native C CB59 with frozen real paid adapters required")

    # No EVAL labels are read or used in any native/S14 forward. Capture
    # immutable predictions on every one of 24 rows, not a selected subset.
    predictions=[]
    with torch.inference_mode():
        for sample in samples:
            hidden=sample.hidden320.unsqueeze(0)
            old=baseline(hidden)
            new=s11(hidden)
            t=int(hidden.shape[1])
            valid=torch.ones((1,t),dtype=torch.bool)
            z=fb(new.relations,new.upos_logits,new.semantic_role_logits,
                 new.dependency_logits,valid)
            ix=torch.tensor([q.last_token for q in sample.alignment.word_alignment],
                            dtype=torch.long)
            guard(old.upos_logits.shape==new.upos_logits.shape==
                  z.upos_logits.shape==(1,t,len(UPOS)) and
                  old.semantic_role_logits.shape==new.semantic_role_logits.shape==
                  z.semantic_role_logits.shape==(1,t,len(SEMANTIC_ROLES)) and
                  old.dependency_logits.shape==new.dependency_logits.shape==
                  z.dependency_logits.shape==(1,t,t) and
                  bool(torch.isfinite(z.dependency_logits).all())
                  and torch.equal(old.upos_logits,new.upos_logits)
                  and torch.equal(old.semantic_role_logits,new.semantic_role_logits)
                  and torch.equal(z.upos_logits,new.upos_logits),
                  "full frozen original S8b/S11 grammar/readout or zero-UPOS wrong")
            predictions.append({
                "row":sample.row,   # gold properties touched only after ALL forwards
                "positions":ix.tolist(),
                "S8b_upos":old.upos_logits[0,ix].clone(),
                "S8b_role":old.semantic_role_logits[0,ix].clone(),
                "base_upos":new.upos_logits[0,ix].clone(),
                "base_role":new.semantic_role_logits[0,ix].clone(),
                "adapter_upos":z.upos_logits[0,ix].clone(),
                "adapter_role":z.semantic_role_logits[0,ix].clone(),
                "base_dependency":new.dependency_logits[0,ix,:].clone(),
                "adapter_dependency":z.dependency_logits[0,ix,:].clone(),
            })
    guard(len(predictions)==24,"must forward all precommitted 24 once")
    # Gold starts HERE; no optimizer, no another model run after opening gold.
    metrics=one_time_metrics(predictions)
    guard(sha_bytes(b)==S8B_CB59_SHA and
          sha_bytes(c)==S11_ROUTER_SHA and
          sha_bytes(d)==S14_PAID_FP16_SHA and
          fb.physical_bytes()==d and
          a.s8_cb59.read_bytes()==b and
          a.s11_router.read_bytes()==c and
          a.s14_adapter.read_bytes()==d,
          "no physical model bytes may change during EVAL24")
    report={
       "stage":"NRA3_S15C_ONE_TIME_TRUE43M_EVAL24_FROZEN_S14_VS_S11_S8B_CAUSAL",
       "original_Calibri43M_GT_EVAL24_CPU_layer8_hidden_SHA256":
           S15B_TRUE43M_HIDDEN_FP16_SHA,
       "S15b_authentic_readonly_616_J7_run":S15B_SOURCE_RUN,
       "S15_new_Russian24_full_coreference_event_UD_precommitted_SHA256":
           S15_EVAL24_SHA,
       "S15_source_only_SHA_lock_preceded_first_EVAL_forward":True,
       "S14_full_PHYSICAL_model_frozen_before_S15_eval_gold":True,
       "S14_locked_rank8_FP16_physical_SHA256":S14_PAID_FP16_SHA,
       "S11_locked_39param_router_physical_SHA256":S11_ROUTER_SHA,
       "S8b_unchanged_original_43_native_C_CB59_sha256":S8B_CB59_SHA,
       "actual_original_Calibri43M_GT_CPU_after_layer8_all24_scored_once":True,
       "true_original_43_CB59_native_C_6M_matrices_unchanged":True,
       "S11_frozen_39_FP16_6cycle_router_attached":True,
       "S14_frozen_1266_FP16_rank8_causal_relation_readout_attached":True,
       "model_native_sources_sha_locked_and_rechecked_after_eval":True,
       "S14_UPOS_zero_correction_proves_exact_logits_parity":True,
       "S15_word_predictions_strictly_causal_own_token_no_future_sentence_summary":True,
       "S15_dependency_only_past_governors_no_root_future_gold_leak":True,
       "original_616_J7_baseline_digests_unchanged_from_S15b":True,
       "protected_616_J7_with_ACTIVE_S14_generation_performed":False,
       "new_answer_fullEOS_quality_measured":False,
       "full_Calibri43M_GT_NRA3_CPU_speed_ge90percent_certified":False,
       "full_parent_and_nra_bpp_le_point7_certified":False,
       "raw_NRA_43CB59_plus_router_plus_feedback_bpp":8*(len(b)+len(c)+len(d))/(6_106_154+39+PARAMETERS),
       "native_russian_negation_event_truth_prediction_measured":False,
       "native_russian_coreference_antecedent_prediction_measured":False,
       "new_heldout24_hand_authored_not_representative_Russian_generalization":True,
       "any_EVAL24_training_or_adapter_reselection":False,
       "EVAL24_optimizer_steps":0,
       "no_prior_consumed_S12c_EVAL16_reopened":True,
       "no_old_five_EVAL128_or_reserved_sixth_S3CI_used":True,
       "no_LIVE_Jessica_chat_Render_Supabase_APK":True,
       "research_acceptance":False,
       "candidate_accepted":False,
       "output_gate":0,
       "activation_allowed":False,
       **metrics,
    }
    path=a.out/"NRA3_S15C_ONE_TIME_NEW_RUSSIAN24_FROZEN_S14_RESEARCH_AUDIT_NO_PROMOTION.json"
    path.write_text(json.dumps(report,sort_keys=True,indent=2,
       ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items()
       if k not in ("per_sentence_readonly_gold_analysis",)},
       ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
