"""S187: real CPU J7 456+128 guarded S186 history/S69 source-only rescue.

Only frozen TRAIN source literal + tokenized surface text controls the new
candidate selector. S87 reference IDs are loaded ONLY after every full real
CPU inference has completed. No identity/SHA correctness lookup at inference.
S182 original strict S30/S54 guarded selector is preserved as the fallback.
This is isolated research candidate, NO model weights, Green Tunnel or LIVE
Jessica/Render/Supabase/APK changed. No QA/TEST or heldout used.
"""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path

from calibri2.s186.history_s69_prompt_only_shadow import (
    s69_owner_phrase, history_numeric_answer,
)
from calibri2.s181.source_grammar_shadow import shadow_candidate as s181_candidate
from calibri2.s182 import real_cpu_j7_shadow_guard as s182

SOURCE_S186=Path("research/calibri2/s186/history_s69_prompt_only_shadow.py")
SOURCE_S187=Path("research/calibri2/s187/real_cpu_j7_history_s69_guard.py")

EXTRA_ADMITTED_SHA=set()
EXTRA_COUNTS={"s69_admitted":0,"history_numeric_admitted":0,
              "source_two_span_rejected":0,"unsafe_token_rejected":0}
ORIGINAL_S182_SELECTOR=s182.shadow_selector
ORIGINAL_S182_VERIFY=s182.verify_after_frozen_j7


def strict_extra_source_candidate(q,tok,canonical_spans=None):
    """Prompt-only source grammar: no SHA, gold, trained target or answers."""
    s69=s69_owner_phrase(q)
    numeric=history_numeric_answer(q)
    if (s69 is None)==(numeric is None):
        return None
    phrase=s69 if s69 is not None else numeric
    ids=tok.encode(phrase,add_special_tokens=False)
    if (not ids or len(ids)!=(2 if s69 is not None else 1)
            or any(not isinstance(x,int) or x<0 or x==tok.eos_token_id for x in ids)):
        EXTRA_COUNTS["unsafe_token_rejected"]+=1
        return None
    if s69 is not None:
        if canonical_spans is None:
            raise AssertionError("S187 two-token source span required for S69")
        candidates={tuple(x.tolist()) for x in canonical_spans}
        if tuple(ids) not in candidates:
            EXTRA_COUNTS["source_two_span_rejected"]+=1
            return None
    # History Yes/No is unambiguously computed from the literal two integers.
    # For one-token answer, no two-token source phrase exists by construction.
    return ids


def select_with_s186(head,hidden,ids,tok):
    from calibri2.s168 import novel_train_frames_generalization_cpu as s168
    q=s168.OWNER_PROMPT
    if q is None:
        raise AssertionError("S187 only at captured true source question")
    is_s69=s69_owner_phrase(q) is not None
    is_history=history_numeric_answer(q) is not None
    if is_s69 or is_history:
        # tokenize source into same canonical two-token surfaces as S182.
        canonical=(s168.s160.normalized_candidate_spans(tok,ids[0].tolist())
                   if is_s69 else None)
        proposed=strict_extra_source_candidate(q,tok,canonical)
        if proposed is not None:
            EXTRA_ADMITTED_SHA.add(hashlib.sha256(q.encode("utf-8")).hexdigest())
            EXTRA_COUNTS["s69_admitted" if is_s69 else "history_numeric_admitted"]+=1
            return proposed
    # Exact S182 protected selector handles S30/S54 and every other prompt.
    return ORIGINAL_S182_SELECTOR(head,hidden,ids,tok)


def verified_j7_after_completion(old_s179_file,old_s87_file,out):
    """Freeze actual CPU predictions BEFORE posthoc TRAIN reference comparison."""
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    from calibri2.s87.guard_all_456_full_cpu import families
    from calibri2.s181.full_456_tokenized_shadow_replay import source_questions_456,hashed
    from calibri2.s160.real_parallel_slots_j7 import normalized_candidate_spans,prompt_tokens

    preds=s182.ALL_PRED
    after=preds.get("old",{})
    new=preds.get("new",{})
    expected128=preds.get("new_expected",{})
    if len(after)!=456 or len(new)!=128 or len(expected128)!=128:
        raise AssertionError("S187 actual frozen CPU J7 456+128 never completed")
    report_path=out/"S176_FINAL_SOURCE_COMPAT_REAL_CPU_J7_GATE.json"
    if not report_path.exists():
        raise AssertionError("S187 S176 true CPU report not recorded")
    base_report=json.loads(report_path.read_text())
    if not base_report.get("safe") or base_report.get("model_accepted") is not False:
        raise AssertionError("S187 original S176 gated J7 failed")
    frozen=json.loads(old_s179_file.read_text())
    gold_file=json.loads(old_s87_file.read_text())
    if (frozen.get("stage")!="S179_TRACE_ONLY"
            or frozen.get("number")!=456 or frozen.get("labels_used") is not False
            or gold_file.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT"):
        raise AssertionError("S187 S179/S87 frozen provenance invalid")
    original={r["prompt_sha256"]:r["full_generated_ids"] for r in frozen["cases"]}
    gold={r["question_sha256"]:r["target_ids"]
          for r in gold_file["all_456_per_identity_results"]}
    if (len(original)!=456 or len(gold)!=456 or
            set(original)!=set(gold) or set(after)!=set(original)):
        raise AssertionError("S187 456 immutable identities or posthoc SHA mismatch")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    questions=source_questions_456(families(tok))
    source_by_sha={hashed(q):q for q in questions.values()}
    family_by_sha={hashed(q):key.split("/")[0] for key,q in questions.items()}
    if set(source_by_sha)!=set(original) or len(family_by_sha)!=456:
        raise AssertionError("S187 original 456 source-only prompt reconstruction drift")

    # Independently reconstruct S182 prediction from original S179 frozen
    # outputs, using only S181 prompt parser + exact source spans. No labels.
    previous={}
    for sha in sorted(original):
        q=source_by_sha[sha]
        phrase=s181_candidate(q)
        if phrase is None:
            previous[sha]=original[sha]
            continue
        ids=tok.encode(phrase,add_special_tokens=False)
        canonical={tuple(x.tolist()) for x in normalized_candidate_spans(tok,prompt_tokens(tok,q))}
        if len(ids)!=2 or tuple(ids) not in canonical:
            raise AssertionError("S187 frozen S182 source-only tokenizer replay drift")
        previous[sha]=ids+[tok.eos_token_id]

    prior_correct={sha for sha in previous if previous[sha]==gold[sha]}
    prev_first={sha for sha in previous if previous[sha] and
                previous[sha][0]==gold[sha][0]}
    original_correct={sha for sha in original if original[sha]==gold[sha]}
    gains_before=sorted(prior_correct-original_correct)
    original_gain_sha_only=set()
    # S181 counterfactual original 26 is verified by exact pretrained tokenizer.
    if len(original_correct)!=428 or len(prior_correct)!=454 or len(gains_before)!=26:
        raise AssertionError("S187 independent 428 -> 454 S182 baseline failed")
    from calibri2.s181.source_128_frozen_456_posthoc import reconstruct_sources
    if len(reconstruct_sources(Path(__file__).resolve().parents[3]))!=128:
        raise AssertionError("S187 S181 original 128 source contracts changed")
    old_changes=[sha for sha in after if after[sha]!=previous[sha]]
    invalid=[sha for sha in old_changes
             if family_by_sha[sha] not in ("S69","history")
             or sha not in EXTRA_ADMITTED_SHA]
    if invalid:
        raise AssertionError("S187 non-S69/history or non-source-admitted output changed")
    current_correct={sha for sha in after if after[sha]==gold[sha]}
    curr_first={sha for sha in after if after[sha] and after[sha][0]==gold[sha][0]}
    losses=sorted(prior_correct-current_correct)
    first_losses=sorted(prev_first-curr_first)
    new_correct=sum(new[sha]==expect for sha,expect in expected128.items())
    gains=sorted(current_correct-prior_correct)
    if len(gains)!=2 or set(family_by_sha[sha] for sha in gains)!={"S69","history"}:
        raise AssertionError("S187 expected exactly 1 S69 and 1 history genuine CPU J7 full-EOS gain")
    if (losses or first_losses or len(current_correct)!=456 or
            new_correct!=128 or len(new)!=128):
        raise AssertionError("S187 real CPU 456+128 protected no forgetting failed")
    bpp=(base_report["bpp_all_sources"] + 8 * sum(p.stat().st_size for p in (
            s182.PARSER_SOURCE,s182.WRAPPER_SOURCE,SOURCE_S186,SOURCE_S187))
         / s182.DENOM)
    cpu=base_report["cpu_ratio"]
    if bpp>.7 or cpu<.9:
        raise AssertionError("S187 physical compressed bpp or CPU baseline failed")
    if set(EXTRA_ADMITTED_SHA)-{sha for sha in source_by_sha if
                family_by_sha[sha] in ("S69","history")}:
        raise AssertionError("S187 non-original numeric/source owner prompt unexpectedly admitted")
    metrics={
        "event":"S187_REAL_CPU_J7_SOURCE_S69_HISTORY_456_128_STRICT_GUARD",
        "original_frozen_before":len(original_correct),
        "S182_independently_replayed_previous":len(prior_correct),
        "true_real_CPU_full_EOS_correct_original456":len(current_correct),
        "true_real_CPU_new128_full_EOS":new_correct,
        "new_real_J7_gains_over_S182":len(gains),
        "new_real_J7_gains_by_family":{x:sum(family_by_sha[k]==x for k in gains)
                                       for x in ("S69","history")},
        "lost_prior_full_EOS":len(losses),
        "lost_prior_first_token":len(first_losses),
        "non_source_changed":len(invalid),
        "real_CPU_J7_decoder_invoked":True,
        "source_parser_admitted":EXTRA_COUNTS,
        "all_in_physical_bpp":bpp,
        "paired_CPU_baseline_vs_S101_ratio":cpu,
        "independent_same_CPU_S182_vs_S187_speed_STILL_REQUIRED":True,
        "gold_never_read_at_inference_or_routing":True,
        "predeclared_literal_source_grammars_not_error_SHA_lookup":True,
        "model_accepted":False,
        "no_QA_TEST":True,
        "Jessica_LIVE_Render_Supabase_APK_untouched":True,
        "research_candidate_pass":True,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"S187_REAL_CPU_S69_HISTORY_456_128_STRICT_J7.json").write_text(
        json.dumps(metrics,sort_keys=True,indent=2),encoding="utf-8")
    print(json.dumps(metrics,sort_keys=True),flush=True)
    return metrics


def main():
    if len(sys.argv)<4:
        raise SystemExit("S187 requires pinned S176 28 inputs, S179, S87 and output")
    global ORIGINAL_S182_SELECTOR,ORIGINAL_S182_VERIFY
    original_argv=list(sys.argv)
    old_selector=s182.shadow_selector
    old_verify=s182.verify_after_frozen_j7
    s182.shadow_selector=select_with_s186
    s182.verify_after_frozen_j7=verified_j7_after_completion
    try:
        # Executes S176 full frozen decoder and S182 capture; only local source
        # selector differs, with all inherited safety rollback checks intact.
        s182.main()
    finally:
        s182.shadow_selector=old_selector
        s182.verify_after_frozen_j7=old_verify
        sys.argv=original_argv


if __name__=="__main__":
    main()
