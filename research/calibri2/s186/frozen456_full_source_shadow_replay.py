"""S186: frozen 456 posthoc tokenizer replay of prompt-only 1- or 2-word owner answers.

Critical gate: ALL source grammar answers and token IDs constructed from
literal user prompts BEFORE S87 gold/S184 posthoc metadata can be loaded.
Existing successful S182 is reconstructed independently from S179 outputs +
S181 source proposals, checked against the immutable successful S182 SHA
gain list. S186 then overlays source-only history/S69 proposals; no new
model calls. A true full 456+128 real CPU J7 remains mandatory later.
"""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from calibri2.s181.source_128_frozen_456_posthoc import source_ast_literal
from calibri2.s181.full_456_tokenized_shadow_replay import source_questions_456,hashed
from calibri2.s181.source_grammar_shadow import shadow_candidate as old_s181
from calibri2.s186.history_s69_prompt_only_shadow import (
    source_candidate, S69_TEMPLATES,
)


def all_source_proposals(tok,questions):
    """No S87/S184 labels, failure IDs or reference answers in signature."""
    from calibri2.s160.real_parallel_slots_j7 import prompt_tokens, normalized_candidate_spans
    prior,new={},{}
    eligible=Counter()
    for key,q in sorted(questions.items()):
        ids=prompt_tokens(tok,q)
        old=old_s181(q)
        fresh=source_candidate(q)
        if old is not None and fresh!=old:
            raise AssertionError("S186 existing S181 owners must remain unchanged")
        def proposal(phrase):
            if phrase is None:return None
            tokens=tok.encode(phrase,add_special_tokens=False)
            if len(tokens) not in (1,2) or tok.eos_token_id in tokens:
                return None
            if len(tokens)==2:
                canonical={tuple(row.tolist()) for row in normalized_candidate_spans(tok,ids)}
                if tuple(tokens) not in canonical:
                    return None
            else:
                # Literal answer Yes/No must be in original source prompt;
                # require actual yes/no comparator question, never a lookup.
                if not (key.startswith("history/") and
                        "Reply with only Yes or No." in q and
                        phrase in ("Yes","No")):
                    return None
            return tokens+[tok.eos_token_id]
        b=proposal(old)
        n=proposal(fresh)
        if b is not None and n!=b:
            raise AssertionError("S186 must retain old source candidate")
        prior[hashed(q)]=b
        new[hashed(q)]=n
        if n is not None:eligible[key.split("/")[0]]+=1
    if len(prior)!=456 or len(new)!=456:
        raise AssertionError("S186 all 456 source identities required")
    if eligible["S30"]!=64 or eligible["S54"]!=64:
        raise AssertionError("S186 original S181 S30/S54 coverage changed")
    return prior,new,dict(sorted(eligible.items()))


def score_after_predeclared(source_questions,archive,s87,s182,baseline,expanded,coverage):
    """Score all 456 only AFTER source-only proposals/IDs are fully frozen."""
    if archive.get("stage")!="S179_TRACE_ONLY" or archive.get("labels_used") is not False or archive.get("number")!=456:
        raise AssertionError("S186 original S179 trace provenance drift")
    if s87.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT":
        raise AssertionError("S186 original S87 TRAIN-only gold missing")
    if s182.get("event")!="S182_REAL_CPU_J7_456_128_SHADOW_PROTECTED" or s182.get("retention_passed") is not True:
        raise AssertionError("S186 source S182 real J7 retention not certified")
    orig={r["prompt_sha256"]:r["full_generated_ids"] for r in archive["cases"]}
    refs={r["question_sha256"]:r["target_ids"] for r in s87["all_456_per_identity_results"]}
    if (len(orig)!=456 or len(refs)!=456 or len(source_questions)!=456
            or {hashed(q) for q in source_questions.values()}!=set(orig)
            or set(orig)!=set(refs) or set(orig)!=set(baseline) or set(orig)!=set(expanded)):
        raise AssertionError("S186 immutable 456 source SHA gold parity mismatch")
    prior={sha:(baseline[sha] if baseline[sha] is not None else orig[sha]) for sha in orig}
    proposed={sha:(expanded[sha] if expanded[sha] is not None else orig[sha]) for sha in orig}
    correct_o={sha for sha in orig if orig[sha]==refs[sha]}
    correct_p={sha for sha in prior if prior[sha]==refs[sha]}
    correct_n={sha for sha in proposed if proposed[sha]==refs[sha]}
    gained_p=sorted(correct_p-correct_o)
    if (len(correct_o)!=428 or len(correct_p)!=454 or
            gained_p!=s182.get("old_gain_sha_only") or
            not correct_o.issubset(correct_p) or s182.get("old_loss_sha_only")!=[]):
        raise AssertionError("S186 offline S182 replay not equal to true S182 real CPU J7")
    lost=sorted(correct_p-correct_n)
    gained=sorted(correct_n-correct_p)
    first_o={sha for sha in orig if orig[sha] and orig[sha][0]==refs[sha][0]}
    first_p={sha for sha in prior if prior[sha] and prior[sha][0]==refs[sha][0]}
    first_n={sha for sha in proposed if proposed[sha] and proposed[sha][0]==refs[sha][0]}
    if not first_p.issubset(first_n):
        raise AssertionError("S186 new source-only proposals lost previously correct first tokens")
    family={hashed(q):k.split("/")[0] for k,q in source_questions.items()}
    if len(family)!=456:
        raise AssertionError("S186 family mapping mismatch")
    by_family=Counter(family[sha] for sha in gained)
    unchanged_other=[sha for sha in orig
                     if family[sha] not in ("S30","S54","S69","history") and proposed[sha]!=orig[sha]]
    if unchanged_other:
        raise AssertionError("S186 unsupported source family decoding changed")
    rejected=bool(lost or not gained or len(correct_n)<454)
    return {
        "event":"S186_SOURCE_ONLY_S69_HISTORY_456_FROZEN_POSTHOC",
        "source_S179_real_J7_run":43323981995,
        "source_S182_real_J7_run":43331434088,
        "before_original_456":len(correct_o),
        "frozen_S182_full_EOS_correct":len(correct_p),
        "S186_counterfactual_full_EOS_correct":len(correct_n),
        "new_full_gain_over_S182":len(gained),
        "lost_prior_full_EOS":len(lost),
        "lost_prior_first":len(first_p-first_n),
        "new_gain_by_source_family":dict(sorted(by_family.items())),
        "source_only_frozen_prompt_coverage_by_family":coverage,
        "gain_SHA_only":gained, "loss_SHA_only":lost,
        "unsupported_source_changes":len(unchanged_other),
        "gold_never_in_prompt_selector":True,
        "only_frozen_replay_NOT_real_integrated_J7":True,
        "real_456_and_new_128_CPU_bpp_check_still_required":True,
        "offline_gain_no_forgetting_gate":not rejected,
        "model_accepted":False,"no_QA_TEST":True,
        "LIVE_Jessica_Render_Supabase_APK_unchanged":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("frozen_s179",type=Path)
    p.add_argument("frozen_s87",type=Path)
    p.add_argument("passed_s182",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError("S186 immutable replay cannot overwrite result")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    from calibri2.s87.guard_all_456_full_cpu import families
    root=Path(__file__).resolve().parents[3]
    frozen_frames=source_ast_literal(root/"research/calibri2/s69/train_new_contexts_without_old_skills_loss.py","TRAIN_FRAMES")
    if tuple(frozen_frames)!=S69_TEMPLATES:
        raise AssertionError("S186 exact original S69 TRAIN grammar changed")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    questions=source_questions_456(families(tok))
    trace=json.loads(args.frozen_s179.read_text())
    if trace.get("stage")!="S179_TRACE_ONLY" or trace.get("labels_used") is not False or trace.get("number")!=456:
        raise AssertionError("S186 original S179 baseline missing")
    original={r["prompt_sha256"]:r["full_generated_ids"] for r in trace["cases"]}
    if len(original)!=456:
        raise AssertionError("S186 original 456 prediction SHA duplicates")
    # Seal ALL 456 lexical proposals before loading any gold/reference.
    old,new,coverage=all_source_proposals(tok,questions)
    report=score_after_predeclared(
        questions,trace,json.loads(args.frozen_s87.read_text()),
        json.loads(args.passed_s182.read_text()),old,new,coverage)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,sort_keys=True,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if not k.endswith("_SHA_only")},sort_keys=True),flush=True)
    if not report["offline_gain_no_forgetting_gate"]:
        raise AssertionError("S186 rejected offline due to full/EOS forgetting or no true gain")


if __name__=="__main__":
    main()
