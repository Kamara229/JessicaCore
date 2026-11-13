"""S185: prompt-only FIRST/EOS guard feasibility on frozen 456 real CPU J7.

All 456 proposals are computed BEFORE looking at S184's 2 posthoc error
identities; no reference tokens, answer keys, QA/TEST or model weights loaded.
Inspect only immutable original S179 generated tokens and TRAIN source prompt.
Not an integrated J7 model, not an accepted result.
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

from calibri2.s164.owner_binding_offline_probe import owner_bound_copy
from calibri2.s181.source_grammar_shadow import shadow_candidate
from calibri2.s181.full_456_tokenized_shadow_replay import source_questions_456, hashed


def token_spans(tok, question):
    """Identical normalized source span path to frozen S182 parallel guard."""
    from calibri2.s160.real_parallel_slots_j7 import normalized_candidate_spans, prompt_tokens
    spans = normalized_candidate_spans(tok, prompt_tokens(tok, question))
    return {tuple(x.tolist()) for x in spans}


def source_only_hypotheses(questions, traces, tok):
    """Absolutely no SHA-indexed error labels or reference answers as input."""
    proposals = {}
    for key, question in sorted(questions.items()):
        sha = hashed(question)
        if sha not in traces:
            raise AssertionError("S185 original 456 prompt hash absent in frozen decoder")
        decoded = list(traces[sha]["full_generated_ids"])
        if not decoded or not all(isinstance(x,int) and x>=0 for x in decoded):
            raise AssertionError("S185 original source decoder tokens invalid")
        canonical = token_spans(tok, question)
        phrase1 = shadow_candidate(question)
        phrase2 = owner_bound_copy(question)
        if phrase1 is not None and phrase2 is not None and phrase1 != phrase2:
            raise AssertionError("S185 overlapping independently parsed prompt policies disagree")
        phrase = phrase1 if phrase1 is not None else phrase2
        copied = tok.encode(phrase, add_special_tokens=False) if phrase is not None else []
        lexical_proposal = (len(copied)==2 and tuple(copied) in canonical
                            and tok.eos_token_id not in copied)
        eos_proposal = (len(decoded)==2 and tok.eos_token_id not in decoded
                        and tuple(decoded) in canonical)
        proposals[sha] = {
            "source_family": key.split("/")[0],
            "source_identity": key,
            "original_full_length": len(decoded),
            "original_terminates_in_EOS": decoded[-1]==tok.eos_token_id,
            "prompt_only_copy_proposal_admissible": bool(lexical_proposal),
            "unchanged_model_first_token_vs_source_proposal": (
                "same" if lexical_proposal and decoded[0]==copied[0] else
                "different" if lexical_proposal else "not-applicable"),
            "strict_source_span_two_token_EOS_append_admissible": bool(eos_proposal),
        }
    if len(proposals)!=456:
        raise AssertionError("S185 frozen 456 source cohort missing or duplicated")
    return proposals


def classify_after_prompt_only_proposals(proposals, posthoc, frozen_trace):
    """Join S184 *after* all candidate prompts have been evaluated."""
    if (posthoc.get("event")!="S184_LAST_TWO_SOURCE456_FROZEN_J7_POSTHOC_SHA_ONLY"
            or posthoc.get("remaining_incorrect") != 2
            or posthoc.get("model_accepted") is not False
            or posthoc.get("no_labels_or_answers_enter_model_or_selector") is not True):
        raise AssertionError("S185 frozen S184 posthoc classifier provenance changed")
    if (frozen_trace.get("stage")!="S179_TRACE_ONLY"
            or frozen_trace.get("labels_used") is not False
            or frozen_trace.get("number")!=456):
        raise AssertionError("S185 immutable original S179 source trace absent")
    if len(proposals)!=456 or len(posthoc.get("remaining_cases_SHA_only", []))!=2:
        raise AssertionError("S185 all 456 prompt-only preclassification required")
    rows = []
    for case in posthoc["remaining_cases_SHA_only"]:
        sha=case["prompt_sha256"]
        if sha not in proposals:
            raise AssertionError("S185 S184 unresolved SHA not found in source-only 456 preclassification")
        p=proposals[sha]
        rows.append({
            "source_SHA":sha,"source_family":p["source_family"],
            "failure_family_posthoc":case["failure_family_posthoc"],
            "original_full_length":p["original_full_length"],
            "original_terminates_in_EOS":p["original_terminates_in_EOS"],
            "prompt_only_copy_proposal_admissible":p["prompt_only_copy_proposal_admissible"],
            "model_first_token_vs_prompt_only_copy":p["unchanged_model_first_token_vs_source_proposal"],
            "EOS_append_only_if_canonical_2token_source_span":
                p["strict_source_span_two_token_EOS_append_admissible"],
            "original_guard_events":case["original_preexisting_guard_events"],
        })
    counts=Counter(x["source_family"] for x in proposals.values())
    if sum(counts.values())!=456:
        raise AssertionError("S185 source-only family count incomplete")
    report={
        "event":"S185_FIRST_AND_EOS_SOURCE_ONLY_FROZEN456_FEASIBILITY",
        "source_S179_run":43323981995,
        "source_S184_pass_run":43334612878,
        "all_456_source_only_preclassified_before_posthoc_error_join":True,
        "all_456_source_families":dict(sorted(counts.items())),
        "two_residual_source_diagnostics_SHA_only":rows,
        "candidate_selection_by_gold_or_SHA":False,
        "no_new_neural_inference":True,
        "requires_new_REAL_J7_456plus128_before_any_acceptance":True,
        "model_accepted":False,"no_QA_TEST":True,
        "LIVE_Jessica_Render_Supabase_APK_unchanged":True
    }
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("frozen_s179_trace",type=Path)
    parser.add_argument("s184_posthoc",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError("S185 immutable SHA-only output must not be overwritten")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL, MODEL_SHA
    from calibri2.s87.guard_all_456_full_cpu import families
    t=json.loads(args.frozen_s179_trace.read_text())
    if t.get("stage")!="S179_TRACE_ONLY" or t.get("number")!=456 or t.get("labels_used") is not False:
        raise AssertionError("S185 original 456 label-free traces missing")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    questions=source_questions_456(families(tok))
    # No expected answer tokens enter this source-only gate.
    traces={x["prompt_sha256"]:x for x in t["cases"]}
    if len(traces)!=456:
        raise AssertionError("S185 immutable source traces duplicated")
    proposals=source_only_hypotheses(questions,traces,tok)
    posthoc=json.loads(args.s184_posthoc.read_text())
    report=classify_after_prompt_only_proposals(proposals,posthoc,t)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({
        "event":report["event"],
        "two_SHA_only":report["two_residual_source_diagnostics_SHA_only"],
        "only_offline":True,"model_accepted":False
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
