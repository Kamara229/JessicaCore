"""S53 first auditable manual-gold evaluation of EXISTING S26 paid graph on NEW24.

The user-approved research question: do original Calibri43M+GT native
representations plus frozen NRA semantic graph identify the actual
predicate, agent and explicitly REJECTED dative recipient? S53 captures
real model features in a separate CPU script. This script loads human
labels only AFTER the entire frozen S26 native graph inference finishes.

S53 source and manually labeled graph were authored and committed before
the S53 model-capture/alignment/evaluator logic. This is a small,
researcher-designed diagnostic and NOT a random independent EVAL. S53
does NO training or threshold selection. S38/S47/S48 HELD26 remain
sealed; no LIVE/model activation or modifications.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s19.s19f_real43m_train9_graph_targets import ROLE,STATUS
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,prediction_totals,
)
from research.nra3_s20.s53_precommitted_negation_quote24_plain import (
    SOURCE_ROWS,text_only_inputs,text_sha256,
)
from research.nra3_s19.s19g_paid_semantic_graph import HEADER as PAID_GRAPH_HEADER

STAGE="NRA3_S53_REAL43M_GT_NATIVE24_MANUAL_NEGATION_RECIPIENT_FIRST_CPU"
NATIVE_FILE="NRA3_S53_GENUINE43M_GT_NEGATION24_WORD339_AND_UNSAFE.safetensors"
NATIVE_MANIFEST="NRA3_S53_GENUINE43M_GT_NEGATION24_OFFSET_PROVENANCE.json"
CAPTURE_MANIFEST="NRA3_S53_GENUINE43M_GT_NEGATION24_NATIVE_J7_REPORT.json"
PAID_MODEL="NRA3_S26_TRUE43M_GT_TRAIN256_GROUPED40_EVENT_GRAPH_12384_FP16.n3s26"
PAID_MANIFEST="NRA3_S26_REAL43M_GT_TRAIN256_GROUPED40_MATCHED_DIAGNOSTICS.json"
P24_SOURCE_COMMIT="475b914d0390771ee6a3e5ee36aacc70f6966980"
P24_SOURCE_BLOB="4b62a96f827728c8e9e8c4424e568031cbddc090"
G24_SOURCE_COMMIT="f769b365f6cdf13aaddb0892e2294cb8d022cfc0"
G24_SOURCE_BLOB="a488142beea69cb87ca8ef0785dfd56afc3bff51"


def require(ok,reason):
    if not bool(ok):
        raise ValueError("S53 actual43M GT new24 manual TRAIN-development audit: "+reason)


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def count(pred,truth):
    tp=int((pred&truth).sum())
    fp=int((pred&~truth).sum())
    fn=int((~pred&truth).sum())
    return {"TP":tp,"FP":fp,"FN":fn,
            "precision":tp/(tp+fp) if tp+fp else 0.,
            "recall":tp/(tp+fn) if tp+fn else 0.,
            "F1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}


def word_idx(word_tokens,word):
    matches=[i for i,t in enumerate(word_tokens) if t.form==word]
    require(len(matches)==1,
            "separately manually labeled gold anchor cannot match exact single raw token "+str(word))
    return matches[0]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--native",type=Path,required=True)
    p.add_argument("--native-manifest",type=Path,required=True)
    p.add_argument("--capture-manifest",type=Path,required=True)
    p.add_argument("--paid-model",type=Path,required=True)
    p.add_argument("--paid-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"first audit must not overwrite existing outputs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    require(all(z.is_file() for z in (
        a.native,a.native_manifest,a.capture_manifest,a.paid_model,a.paid_manifest)),
        "actual original43M CPU-native features or earlier paid S26 model not downloaded")
    require((a.native.name,a.native_manifest.name,a.capture_manifest.name,
             a.paid_model.name,a.paid_manifest.name)==(
             NATIVE_FILE,NATIVE_MANIFEST,CAPTURE_MANIFEST,PAID_MODEL,PAID_MANIFEST),
            "source archive filenames did not match frozen S53/S26")
    rawmanifest=json.loads(a.native_manifest.read_text(encoding="utf-8"))
    capture=json.loads(a.capture_manifest.read_text(encoding="utf-8"))
    paidmanifest=json.loads(a.paid_manifest.read_text(encoding="utf-8"))
    require(rawmanifest["stage"]=="NRA3_S53_GENUINE43M_GT_NEGATION24_BPE_SOURCE_ONLY"
            and rawmanifest["frozen_S53_NEW24_raw_source_sha256"]==text_sha256()
            and rawmanifest["derived_REAL43M_WORD339_filename"]==a.native.name
            and rawmanifest["derived_REAL43M_WORD339_SHA256"]==sha(a.native)
            and rawmanifest["human_manual_graph_gold_N53_NOT_IMPORTED"]
            and rawmanifest["actual_original43M_12layer_GT_post616J7"]
            and rawmanifest["original_616_J7_replayed_before_read_only_quotes"]
            and rawmanifest["output_gate"]==0
            and not rawmanifest["candidate_accepted"]
            and not rawmanifest["activation_allowed"],
            "S53 original24 CPU native gold-free provenance failure")
    require(capture["stage"]=="NRA3_S53_GENUINE43M_GT_NEGATION24_NATIVE_CPU_FIRST_PASS"
            and capture["S53_PRECOMMITTED_NEW24_PLAINTEXT_SHA256"]==text_sha256()
            and capture["S53_RAW24_SOURCE_COMMIT"]==P24_SOURCE_COMMIT
            and capture["S53_RAW24_SOURCE_BLOB"]==P24_SOURCE_BLOB
            and capture["all_616_native_CPU_generated_token_digests_unchanged"]
            and capture["actual_genuine_43M_CPU_new_negation24_TRAIN_prefills"]==24
            and capture["original_456_full_EOS"]==456
            and capture["original_new128_full_EOS"]==128
            and capture["original_new32_full_EOS"]==0
            and capture["S53_manual_gold_SEPARATE_NOT_imported_during_model_or_adapter_forward"]
            and capture["no_optimizer_or_model_selection"]
            and capture["NRA3_output_gate"]==0,
            "genuine original 43M 616 J7 preserved source conditions drift")
    require(paidmanifest["stage"]=="NRA3_S26_REAL43M_GT_NEW256_SYNTHETIC_TRAIN_ONLY_GROUPED40_GRAPH"
            and paidmanifest["physical_FP16_filename"]==PAID_MODEL
            and paidmanifest["physical_FP16_candidate_SHA256"]==sha(a.paid_model)
            and paidmanifest["full_physical_FP16_parameters"]==12384
            and paidmanifest["output_gate"]==0
            and not paidmanifest["candidate_accepted"],
            "original trained S26 paid graph physical model missing")
    native=load_file(str(a.native),device="cpu")
    require(set(native)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S53_NEW24"},
        "S53 true native features contain unknown/gold fields")
    x=native["original_real43m_gt_word339_READONLY"].detach()
    valid=native["original_native_word_valid_mask"].detach()
    unsafe=native["native_BPE_causally_unsafe_word_mask"].detach()
    ids=native["source_row_indices_in_frozen_S53_NEW24"]
    require(x.ndim==3 and x.shape[0]==24 and x.shape[-1]==339
            and x.dtype==torch.float32 and valid.shape==unsafe.shape==x.shape[:2]
            and valid.dtype==unsafe.dtype==torch.bool
            and ids.tolist()==list(range(24)),
            "S53 native 24 real post-GT aligned word339 + BPE unsafe flags invalid")
    raw_full_sha=tensor_sha(x,valid,unsafe)
    torch.manual_seed(531118)
    paid_file=a.paid_model.read_bytes()
    model=EventGroundedInversionGraph.from_physical(paid_file)
    require(all(not p.requires_grad for p in model.parameters())
            and model.output_gate==model.parent.output_gate==0
            and not model.candidate_accepted and not model.activation_allowed,
            "original S26 paid 12384 model not safely frozen")
    paid_param_sha=tensor_sha(*(p.detach() for p in model.parameters()))
    parsed=PAID_GRAPH_HEADER.unpack_from(paid_file)
    semantic_original=paid_file[PAID_GRAPH_HEADER.size+parsed[4]:-32]
    require(model.head_physical()==semantic_original,
            "original S26 trained semantic head did not restore exact paid FP16")
    with torch.inference_mode():
        unmasked=model(x[...,:338],valid)
    observer=S52QuoteAwareFrozenPaidGraphBridge(model)
    with torch.inference_mode():
        watched=observer.observe(x,valid,unsafe)
    for key in ("event_status","role","argument_polarity","coreference","event_relation"):
        require(torch.equal(getattr(unmasked,key),
                            getattr(watched.original_paid_graph_logits,key)),
                "new S53 quote/BPE bridge altered existing S26 five-head logits")
    require(tensor_sha(x,valid,unsafe)==raw_full_sha
            and tensor_sha(*(p.detach() for p in model.parameters()))==paid_param_sha
            and observer.output_gate==model.output_gate==model.parent.output_gate==0,
            "new S53 original 43M native source or prior S26 paid model modified")

    # CRITICAL: Only NOW load human manual semantic/factuality annotations.
    # Entire original Calibri43M and S26 graph forward is already complete,
    # no additional inference or training is made after gold import.
    from research.nra3_s20.s53_separate_manual_gold_negation24 import (
        gold24,gold_sha256,
    )
    gt=gold24()
    require(len(gt)==len(text_only_inputs())==24
            and all(g.row_id==SOURCE_ROWS[i][0] for i,g in enumerate(gt)),
            "hand gold source row order changed")
    b,w=valid.shape
    gold_predicate=torch.zeros((b,w),dtype=torch.bool)
    gold_event_negated=torch.zeros((b,w),dtype=torch.bool)
    gold_recipient=torch.zeros((b,w,w),dtype=torch.bool)
    gold_rejected=torch.zeros((b,w,w),dtype=torch.bool)
    gold_agent=torch.zeros((b,w,w),dtype=torch.bool)
    pred_gold_per_row=[]
    safe_gold_rejected=0
    safe_gold_recipient=0
    for i,(row,g) in enumerate(zip(SOURCE_ROWS,gt)):
        toks=raw_surface_grammar(row[2])
        n=int(valid[i].sum())
        require(len(toks)==n,"native original S53 word offset vs handwritten gold mismatch")
        event_i=word_idx(toks,g.predicate)
        agent_i=word_idx(toks,g.agent)
        rec_i=word_idx(toks,g.recipient) if g.recipient is not None else None
        require(toks[event_i].form==g.predicate
                and bool(valid[i,event_i]) and bool(valid[i,agent_i]),
                "explicit human predicate/agent not in native original tokens")
        if rec_i is not None:
            require(bool(valid[i,rec_i]),"human directed recipient missing from native source")
        gold_predicate[i,event_i]=True
        if g.event_status=="NEGATED":gold_event_negated[i,event_i]=True
        gold_agent[i,event_i,agent_i]=True
        if rec_i is not None:
            gold_recipient[i,event_i,rec_i]=True
            if g.recipient_polarity=="REJECTED":gold_rejected[i,event_i,rec_i]=True
            if bool(watched.eligible_directed_pair_mask[i,event_i,rec_i]):
                safe_gold_recipient+=1
                safe_gold_rejected+=int(g.recipient_polarity=="REJECTED")
        pred=STATUS[int(unmasked.event_status[i,event_i].argmax())]
        role_at_gold=ROLE[int(unmasked.role[i,event_i,rec_i].argmax())] if rec_i is not None else None
        polar_at_gold=("NONE","POSITIVE","REJECTED")[
            int(unmasked.argument_polarity[i,event_i,rec_i].argmax())] if rec_i is not None else None
        pred_gold_per_row.append({
            "id":g.row_id,"category":row[1],
            "text_SOURCE":row[2],
            "manual_main_predicate_token":g.predicate,
            "manual_main_agent_token":g.agent,
            "manual_recipient_token":g.recipient,
            "manual_event_factuality":g.event_status,
            "manual_recipient_polarity":g.recipient_polarity,
            "existing_S26_predicted_predicate_status":pred,
            "existing_S26_predicted_gold_pair_role":role_at_gold,
            "existing_S26_predicted_gold_pair_polarity":polar_at_gold,
            "manual_main_event_predicate_native_word_eligible":bool(watched.eligible_word_mask[i,event_i]),
            "manual_directed_recipient_gold_pair_eligible":(
                bool(watched.eligible_directed_pair_mask[i,event_i,rec_i])
                if rec_i is not None else None),
            "manual_gold_entered_only_AFTER_physical_original_and_S26_forward":True,
        })
    # No oracle-provided predicate is ever passed into neural forward.
    # Predictions below use ALL native nodes/pairs, not a handpicked pair
    # to artificially suppress FP.
    active_word=watched.eligible_word_mask
    active_pair=watched.eligible_directed_pair_mask
    base_pair=valid.unsqueeze(2)&valid.unsqueeze(1)
    base_pair=base_pair&~torch.eye(w,dtype=torch.bool).unsqueeze(0)
    pred_event=unmasked.event_status.argmax(-1)
    pred_role=unmasked.role.argmax(-1)
    pred_polarity=unmasked.argument_polarity.argmax(-1)
    predicted_event_any=(pred_event!=0)&valid
    predicted_event_neg=(pred_event==STATUS.index("NEGATED"))&valid
    predicted_recipient=(pred_role==ROLE.index("RECIPIENT"))&base_pair
    predicted_rejected=(pred_role==ROLE.index("RECIPIENT"))&(
        pred_polarity==2)&base_pair
    predicted_agent=(pred_role==ROLE.index("AGENT"))&base_pair
    stats={
        "main_event_node_any_unmasked":count(predicted_event_any,gold_predicate),
        "main_event_node_any_safe":count(predicted_event_any&active_word,gold_predicate),
        "main_event_negated_unmasked":count(predicted_event_neg,gold_event_negated),
        "main_event_negated_safe":count(predicted_event_neg&active_word,gold_event_negated),
        "directed_recipient_unmasked":count(predicted_recipient,gold_recipient),
        "directed_recipient_safe":count(predicted_recipient&active_pair,gold_recipient),
        "directed_REJECTED_unmasked":count(predicted_rejected,gold_rejected),
        "directed_REJECTED_safe":count(predicted_rejected&active_pair,gold_rejected),
        "directed_AGENT_unmasked":count(predicted_agent,gold_agent),
        "directed_AGENT_safe":count(predicted_agent&active_pair,gold_agent),
    }
    positive_id=list(range(6))
    negated_postposed=list(range(6,12))
    negated_preposed=list(range(12,18))
    quoted_id=list(range(18,24))
    groups={
       "positive":positive_id,
       "negated_postposed":negated_postposed,
       "negated_preposed":negated_preposed,
       "quoted_report":quoted_id,
    }
    group_stats={}
    for category,idx in groups.items():
        idx=torch.tensor(idx,dtype=torch.long)
        bpred=predicted_rejected.index_select(0,idx)
        bgold=gold_rejected.index_select(0,idx)
        bmask=active_pair.index_select(0,idx)
        group_stats[category]={
            "unmasked_REJECTED_gold_vs_prediction":count(bpred,bgold),
            "safe_REJECTED_gold_vs_prediction":count(bpred&bmask,bgold),
            "gold_manually_written_directed_REJECTED":int(bgold.sum()),
            "original_true43M_native_unsafe_words":int(unsafe.index_select(0,idx).sum()),
            "original_S48_quote_content_words":int((x.index_select(0,idx)[...,338]).sum()),
        }

    report={
        "stage":STAGE,
        "S53_SEPARATELY_PRECOMMITTED_PLAIN24_COMMIT":P24_SOURCE_COMMIT,
        "S53_SEPARATELY_PRECOMMITTED_PLAIN24_BLOB":P24_SOURCE_BLOB,
        "S53_SEPARATELY_PRECOMMITTED_GOLD24_COMMIT":G24_SOURCE_COMMIT,
        "S53_SEPARATELY_PRECOMMITTED_GOLD24_BLOB":G24_SOURCE_BLOB,
        "plain24_source_sha256":text_sha256(),
        "manual_gold24_source_sha256":gold_sha256(),
        "original_true43M_616J7_CPU_replay_at_S53_capture":True,
        "original_true43M_plus_GT_12layer_real24_native_words":int(valid.sum()),
        "actual_original_true43M_F16_S53_capture_file_sha256":sha(a.native),
        "real_S53_original_input_source_manifest_sha256":sha(a.native_manifest),
        "original_prior_S26_paid12384_FP16_physical_sha256":sha(a.paid_model),
        "original_prior_S26_semantic_head_exact_paid_FP16":True,
        "original_S26_neural_graph_all_five_heads_bitwise_identical_pre_post_quote_BPE_mask":True,
        "original_S26_12384_FP16_loaded_parameters_unchanged":True,
        "original_true43M_native_word339_sha256_unchanged":True,
        "human_manual_gold_labels_IMPORTED_AFTER_every_model_forward":True,
        "human_manual_gold_predicate_agent_recipient_labels_NEVER_passed_into_ANY_model":True,
        "predeclared_reference_gold24_is_researcher_authored_not_random_or_independent_generalization":True,
        "s53_native_eligible_directed_recipient_golds":safe_gold_recipient,
        "s53_native_eligible_directed_REJECTED_golds":safe_gold_rejected,
        "S53_original_S26_frozen_graph_manual_gold_metrics_NOT_unbiased_EVAL":stats,
        "S53_24case_manual_gold_per_family":group_stats,
        "S53_all24_individual_gold_and_after_inference_diagnostics":pred_gold_per_row,
        "prior_S26_frozen_raw_head_argmax_descriptive_not_truth":prediction_totals(watched),
        "original_model_and_decoder_not_integrated_no_new_616J7_bpp_CB59":True,
        "no_training_finetuning_threshold_tuning_or_S53_gold_selection":True,
        "original_S26_bidirectional_context_not_online_causally_safe":True,
        "S51_BPE_shared_unsafe_pairs_abstained_not_causal_certification":True,
        "S38_S47_S48_HELD26_and_independent_EVAL32_EVAL128_not_imported":True,
        "LIVE_Render_Supabase_Jessica_APK_unchanged":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"NRA3_S53_REAL43M_GT_NEW24_MANUAL_NEGATION_GRAPH_FIRST_CPU.json").write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8")
    print("S53_GENUINE43M_GT_MANUAL_NEGATION24_FROZEN_S26_GRAPH",
          json.dumps({
            "real_original_native_words":int(valid.sum()),
            "manual_gold_directed_rejected":int(gold_rejected.sum()),
            "safe_gold_directed_rejected":safe_gold_rejected,
            "original_S26_REJECTED":stats["directed_REJECTED_unmasked"],
            "S52_masked_REJECTED":stats["directed_REJECTED_safe"],
            "original_S26_NEGATED_node":stats["main_event_negated_unmasked"],
            "original_S26_RECIPIENT":stats["directed_recipient_unmasked"],
            "precommitted_gold_after_forward":True,
            "gate":0,
          },sort_keys=True),flush=True)
    print("S53_GOLD24_FIRST_REAL43M_GT_GROUP_BREAKDOWN",
          json.dumps(group_stats,sort_keys=True),flush=True)
    print("S53_MANUAL_AUTHOR_DIAGNOSTIC_NOT_INDEPENDENT_EVAL_NOT_ACCEPTED",flush=True)


if __name__=="__main__":
    main()
