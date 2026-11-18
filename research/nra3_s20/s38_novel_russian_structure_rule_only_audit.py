"""S38: execute frozen S35→S37 RAW decoder on newly committed Russian texts.

This is an INPUT-ONLY decoder eligibility challenge using synthetic
all-candidate-high REJECTED logits; it measures structural policy
coverage, NOT neural false-positive rates or true43M+GT accuracy.
It FAILS CLOSED if one even implies these 32 are independently scored
native CPU representations. No model is loaded or trained.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    raw_surface_grammar, FEATURES, DIM,
)
from research.nra3_s19.s19g_paid_semantic_graph import GraphLogits
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import text_only_inputs
from research.nra3_s20.s35_real43m_local_negation_scope_trainonly import (
    apply_local_ne_dative_scope,verify_logits_unmodified,
)
from research.nra3_s20.s37_real43m_source_verb_scope_trainonly import (
    apply_raw_source_verb_scope,assert_only_ineligible_rejected_masked,
    PAST,FUTURE,
)
from research.nra3_s20.s38_novel_russian_structure_challenge_source import (
    ROWS, validate_challenge, challenge_digest, STAGE,
)

CHALLENGE_FROZEN_SOURCE_BLOB_GIT_SHA="d1c5be4402fb456adab896a828e742f5f34c48dd"
REPORT_STAGE="NRA3_S38_NEW_RUSSIAN_ADVERSARIAL_RAW_RULE_ONLY_ELIGIBILITY_AUDIT_V1"

def require(ok,why):
    if not bool(ok):
        raise ValueError("S38 new Russian source-only decoder stress test: "+why)

def raw_input_only_batch(rows=ROWS):
    """No human role labels or predicate annotations enter raw features."""
    words=[raw_surface_grammar(r.text) for r in rows]
    w=max(map(len,words))
    features=torch.zeros((len(rows),w,338),dtype=torch.float32)
    mask=torch.zeros((len(rows),w),dtype=torch.bool)
    for i,tokens in enumerate(words):
        for j,t in enumerate(tokens):
            features[i,j,320:]=torch.tensor(t.features,dtype=torch.float32)
        mask[i,:len(tokens)]=True
    require(DIM==18 and len(FEATURES)==18
            and torch.isfinite(features).all()
            and torch.all(features[...,:320]==0)
            and features.shape==(len(rows),w,338),
            "raw18 INPUTS changed or fake hidden320 zero invariant broken")
    return features,mask,words

def all_candidate_upper_bound(b,w):
    """Deliberately non-neural saturated REJECTED scores, not Calibri!

    Saturate all directed pairs so any policy *allowed* candidate can
    emit REJECTED. On a negative quotation this is only a POTENTIAL FP,
    never evidence that the actual neural model would hallucinate it.
    """
    pol=torch.zeros((b,w,w,3),dtype=torch.float32)
    pol[...,0]=1.
    pol[...,2]=12.
    return GraphLogits(
        torch.zeros((b,w,6)),
        torch.zeros((b,w,w,9)),
        pol,
        torch.zeros((b,w,w,2)),
        torch.zeros((b,w,w,5)),
    )

def score_policy(rows=ROWS):
    require(validate_challenge()["count"]==32 and
            challenge_digest()==validate_challenge()["canonical_source_and_label_sha256"],
            "source challenge changed after model development")
    old_texts=set(text_only_inputs())
    require(len(old_texts)==256
            and all(r.text not in old_texts for r in rows),
            "new challenge text duplicates original TRAIN256")
    x,valid,toks=raw_input_only_batch(rows)
    out=all_candidate_upper_bound(*valid.shape)
    with torch.inference_mode():
        scope35=apply_local_ne_dative_scope(out,x,valid)
        verify_logits_unmodified(out,scope35,x,valid)
        scope37=apply_raw_source_verb_scope(scope35,x,valid)
        assert_only_ineligible_rejected_masked(scope35,scope37,x,valid)
    # Human references are used ONLY AFTER all raw+decoder predictions
    # have been computed from plain text, for offline pair-level scoring.
    grouped={}
    all_rows=[]
    for i,r in enumerate(rows):
        sources=[j for j,t in enumerate(toks[i]) if t.form==r.predicate]
        targets=[j for j,t in enumerate(toks[i]) if t.form==r.recipient]
        require(len(sources)==len(targets)==1
                and sources[0]!=targets[0], "challenge human pair ambiguous")
        src,tgt=sources[0],targets[0]
        allow35=int(scope35.argument_polarity[i,src,tgt].argmax())==2
        allow37=int(scope37.argument_polarity[i,src,tgt].argmax())==2
        require(not allow37 or allow35,"S37 source restriction enlarged S35 REJECTED eligibility")
        bits=toks[i][src].features
        target_bits=toks[i][tgt].features
        s={
            "id":r.key,
            "phenomenon":r.phenomenon,
            "raw_plain_text":r.text,
            "offline_reference_predicate":r.predicate,
            "offline_reference_recipient":r.recipient,
            "offline_semantically_rejected":r.rejected,
            "S35_source_only_policy_allows_REJECTED":allow35,
            "S37_source_only_policy_allows_REJECTED":allow37,
            "source_past_verb_raw18":int(bits[PAST]),
            "source_future_verb_raw18":int(bits[FUTURE]),
            "recipient_previous_NE_raw18":int(target_bits[FEATURES.index("previous_word_negative_particle")]),
            "recipient_dative_raw18":int(target_bits[FEATURES.index("ending_dative_possible")]),
            "neural_logits_are_artificial_saturated_NOT_real_model":True,
            "offline_pair_indices_used_after_decoder_only":[src,tgt],
        }
        all_rows.append(s)
        grouped.setdefault(r.phenomenon,[]).append(s)
    stats={}
    for category,records in grouped.items():
        stat={"count":len(records),"positive":sum(z["offline_semantically_rejected"] for z in records)}
        for policy in ("S35","S37"):
            observed=[z[policy+"_source_only_policy_allows_REJECTED"] for z in records]
            stat[policy+"_policy_eligible"]=sum(observed)
            stat[policy+"_policy_true_pair_supported"]=sum(
                pred and row["offline_semantically_rejected"]
                for pred,row in zip(observed,records))
            stat[policy+"_policy_true_pair_blocked"]=sum(
                not pred and row["offline_semantically_rejected"]
                for pred,row in zip(observed,records))
            stat[policy+"_policy_false_pair_allowed"]=sum(
                pred and not row["offline_semantically_rejected"]
                for pred,row in zip(observed,records))
        stats[category]=stat
    def aggregate(key):
        eligible=sum(r[key+"_source_only_policy_allows_REJECTED"] for r in all_rows)
        positive=sum(r["offline_semantically_rejected"] for r in all_rows)
        supported=sum(r["offline_semantically_rejected"] and
                      r[key+"_source_only_policy_allows_REJECTED"] for r in all_rows)
        false_allowed=eligible-supported
        return {
            "total":len(all_rows),"reference_positive":positive,
            "reference_negative":len(all_rows)-positive,
            "true_rejected_pairs_eligible":supported,
            "true_rejected_pairs_blocked":positive-supported,
            "negatives_with_potential_gate_false_eligibility":false_allowed,
            "policy_pair_support_rate":supported/positive if positive else None,
            "not_model_TP_FP_FN":True,
        }
    s35,s37=aggregate("S35"),aggregate("S37")
    require(len(grouped)==4 and all(len(rows)==8 for rows in grouped.values()),
            "four challenge structures not all represented")
    # Negative-case eligibility is a possible-model-FP only; do not
    # misrepresent S35/S37 or even a fully trained frozen43M performance.
    return {
        "stage":REPORT_STAGE,
        "frozen_source_stage":STAGE,
        "frozen_S38_source_sha256":challenge_digest(),
        "frozen_source_git_blob_sha":CHALLENGE_FROZEN_SOURCE_BLOB_GIT_SHA,
        "source_commit_before_any_S38_evaluator":
            "61ee235ea9e7ebfe4d19ba13be0dd0516afba231",
        "source_fixed_before_any_S38_model_predictions":True,
        "source_texts_disjoint_from_prior_synthetic_TRAIN256":True,
        "new_challenge_design_is_adversarial_and_NOT_statistically_blind":True,
        "independent_EVAL128_EVAL32_not_used":True,
        "genuine_original43M_GT_embeddings_NOT_computed_on_S38":True,
        "frozen_original_43M_GT_weights_and_616J7_NOT_loaded":True,
        "NO_NRA3_FP16_model_loaded_or_trained":True,
        "all_candidates_have_deliberately_saturated_FAKE_REJECTED_logits":True,
        "scores_are_policy_eligibility_NOT_NEURAL_precision_recall":True,
        "no_change_to_S35_or_S37_decoders":True,
        "all_role_event_relation_coref_logits_unchanged":True,
        "S35_eligibility_counts":s35,
        "S37_eligibility_counts":s37,
        "by_predeclared_new_structure":stats,
        "case_level_offline_details":all_rows,
        "known_risks_include_nonlocal_negation_passive_infinitive_and_quoted_mention":True,
        "rule_is_NOT_accepted_as_robust_Russian_grammar":True,
        "do_not_train_or_tune_on_these_NEW_held_cases":True,
        "future_real43M_test_requires_new_separately_frozen_CPU_capture":True,
        "LIVE_Render_Supabase_Jessica_APK_NRA2_active_Calibri_unchanged":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists(),"never overwrite challenge report")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    d=score_policy()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"NRA3_S38_NEW_RUSSIAN_STRUCTURE_32_RULE_ELIGIBILITY_STRESS.json").write_text(
        json.dumps(d,sort_keys=True,ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8")
    print("S38_FROZEN_SOURCE_SHA256",d["frozen_S38_source_sha256"],flush=True)
    print("S38_SOURCE_ONLY_NEW_RUSSIAN_STRUCTURE_POLICY_ELIGIBILITY",
          json.dumps({"S35":d["S35_eligibility_counts"],
                      "S37":d["S37_eligibility_counts"],
                      "by_structure":d["by_predeclared_new_structure"]},
                     ensure_ascii=False,sort_keys=True),flush=True)
    print("S38_NO_ORIGINAL43M_CPU_FEATURES_AND_NO_MODEL_CANDIDATE_CERTIFICATION",flush=True)

if __name__=="__main__":
    main()
