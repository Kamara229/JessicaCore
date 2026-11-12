"""S181 original 456 full-source, tokenizer-verified *offline* shadow replay.

No second neural inference is done. Replay saved S179 actual real CPU J7
outputs; only substitute source-derived, prompt-only S30/S54 shadow phrase
encoded with the exact frozen pretrained TOKENIZER (NO model weights).
Build all 456 source questions from authorized TRAIN generators using
questions ONLY, BEFORE reading S87 labels; no QA or heldout. Score AFTER
all outputs freeze. Not integrated J7; no new CPU/bpp/speed, no promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from calibri2.s181.source_grammar_shadow import shadow_candidate
from calibri2.s181.source_128_frozen_456_posthoc import reconstruct_sources


def hashed(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def source_questions_456(families: dict) -> dict[str, str]:
    """Extract literal prompts without carrying gold targets or answers."""
    result = {}
    for family, pairs in families.items():
        for pair in pairs:
            for side in ("left", "right"):
                key = f"{family}/{pair['pair_id']}/{side}"
                question = pair[side]["question"]
                if key in result or not isinstance(question, str):
                    raise AssertionError("S181 malformed frozen 456 source questions")
                result[key] = question
    if len(result) != 456 or len(set(result.values())) != 456:
        raise AssertionError("S181 source generator must cover 456 unique prompts")
    return result


def source_only_proposals(questions: dict[str, str], tok) -> dict[str, list[int] | None]:
    """PROMPT ONLY. No label or reference-correctness arguments."""
    candidates = {}
    for key, prompt in sorted(questions.items()):
        phrase = shadow_candidate(prompt)
        if phrase is None:
            candidates[key] = None
            continue
        ids = tok.encode(phrase, add_special_tokens=False)
        if (not isinstance(ids, list) or len(ids) != 2
                or tok.eos_token_id in ids
                or any(not isinstance(x, int) for x in ids)):
            raise AssertionError("S181 shadow candidate is not exactly 2 tokens")
        candidates[key] = ids + [tok.eos_token_id]
    return candidates


def counterfactual_offline(
    questions: dict[str, str], tok, real_s179: dict, frozen_s87: dict,
    source128: list[tuple[str, str, str]],
) -> dict:
    """Replay frozen old output; no J7 decoder rerun and no selector changes."""
    if len(questions) != 456 or len(set(questions.values())) != 456:
        raise AssertionError("S181 complete source questions required")
    proposals = source_only_proposals(questions, tok)  # BEFORE label access
    # From this point all proposed output IDs are frozen; only scores are posthoc.
    if (real_s179.get("stage") != "S179_TRACE_ONLY"
            or real_s179.get("number") != 456
            or real_s179.get("labels_used") is not False
            or frozen_s87.get("stage") != "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT"):
        raise AssertionError("S181 pinned S179/S87 baseline missing")
    traces = {x["prompt_sha256"]: x for x in real_s179.get("cases", [])}
    references = {x["id"]: x for x in frozen_s87.get("all_456_per_identity_results", [])}
    if len(traces) != 456 or len(references) != 456 or set(references) != set(questions):
        raise AssertionError("S181 immutable label/trace identity drift")
    if {r["question_sha256"] for r in references.values()} != set(traces):
        raise AssertionError("S181 SHA mismatch among 456 frozen CPU outputs")
    source128_keys = {key for key, _question, _answer in source128}
    if len(source128_keys) != 128:
        raise AssertionError("S181 original source128 reconstruction drift")
    source128_hashes = {hashed(q) for _key, q, _answer in source128}
    if len(source128_hashes) != 128:
        raise AssertionError("S181 source128 SHA duplicates")
    for key, prompt, _answer in source128:
        if questions.get(key) != prompt:
            raise AssertionError("S181 source128 literal question ABI drift")
    covered = {key for key, ids in proposals.items() if ids is not None}
    if covered != source128_keys:
        raise AssertionError("S181 shadow must cover exactly 128 original S30/S54 TRAIN prompts")
    if any(hashed(questions[key]) not in source128_hashes for key in covered):
        raise AssertionError("S181 source128 question hashes drift")

    counters = Counter()
    improved_sha_only = []
    forgotten_sha_only = []
    first_lost_sha_only = []
    wrong_pair = []
    for key in sorted(questions):
        prompt_sha = hashed(questions[key])
        r = references[key]
        if prompt_sha != r["question_sha256"]:
            raise AssertionError("S181 original 456 source prompt SHA mismatch")
        original = traces[prompt_sha]["full_generated_ids"]
        gold = r["target_ids"]  # SCORE ONLY, never passed to lexical proposal.
        candidate = proposals[key] if proposals[key] is not None else original
        if not original or not candidate or not gold:
            raise AssertionError("S181 empty generated/target token IDs")
        before_full, after_full = original == gold, candidate == gold
        before_first, after_first = original[0] == gold[0], candidate[0] == gold[0]
        counters["total"] += 1
        counters["shadow_matched" if proposals[key] is not None else "shadow_abstained"] += 1
        counters["before_full_correct"] += int(before_full)
        counters["offline_after_full_correct"] += int(after_full)
        counters["before_first_correct"] += int(before_first)
        counters["offline_after_first_correct"] += int(after_first)
        counters["gained_full"] += int(not before_full and after_full)
        counters["lost_full"] += int(before_full and not after_full)
        counters["gained_first"] += int(not before_first and after_first)
        counters["lost_first"] += int(before_first and not after_first)
        if proposals[key] is None and original != candidate:
            raise AssertionError("S181 incorrectly modified frozen nonmatched output")
        if proposals[key] is not None and key not in source128_keys:
            raise AssertionError("S181 out-of-scope shadow match")
        if not before_full and after_full:
            improved_sha_only.append({"sha": prompt_sha, "family": key.split("/")[0]})
        if before_full and not after_full:
            forgotten_sha_only.append({"sha": prompt_sha, "family": key.split("/")[0]})
        if before_first and not after_first:
            first_lost_sha_only.append(prompt_sha)
        if not after_full:
            wrong_pair.append({"sha": prompt_sha, "family": key.split("/")[0],
                               "was_shadow_matched": proposals[key] is not None})
    if (counters["total"] != 456 or counters["before_full_correct"] != 428
            or counters["shadow_matched"] != 128 or counters["shadow_abstained"] != 328):
        raise AssertionError("S181 old 456 original J7 baseline drift")
    return {
        "stage": "S181_RESEARCH_ONLY_SOURCE_456_OFFLINE_REPLAY_NOT_INTEGRATED",
        "original_s179_real_CPU_run": 43323981995,
        "original_source_S87_456": True,
        "source_only_candidates_frozen_before_labels": True,
        "source_prompts_reconstructed": 456,
        "sources_ast_pinned": 128,
        "metrics": dict(sorted(counters.items())),
        "improved_sha_only": improved_sha_only,
        "forgotten_sha_only": forgotten_sha_only,
        "first_lost_sha_only": first_lost_sha_only,
        "remaining_incorrect_sha_only": wrong_pair,
        "shadow_model_weights_changed": False,
        "same_pretrained_tokenizer_only": True,
        "frozen_saved_neural_outputs_replayed_not_regenerated": True,
        "not_actual_integrated_CPU_J7": True,
        "not_generalization_proof_due_to_train_template": True,
        "cpu_speed_unmeasured": True,
        "physical_bpp_unmeasured": True,
        "new128_and_heldout_unmeasured": True,
        "no_QA_TEST_or_Jessica_chat_test": True,
        "not_model_accepted": True,
        "LIVE_untouched": True,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("s179_trace", type=Path)
    p.add_argument("s87_train", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("S181 full source offline immutable report already exists")
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL, MODEL_SHA
    from calibri2.s87.guard_all_456_full_cpu import families

    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    questions = source_questions_456(families(tok))
    source128 = reconstruct_sources(Path(__file__).resolve().parents[3])
    report = counterfactual_offline(
        questions, tok, json.loads(a.s179_trace.read_text()),
        json.loads(a.s87_train.read_text()), source128)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, sort_keys=True, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "S181_SOURCE456_FROZEN_J7_OFFLINE_REPLAY",
        "metrics": report["metrics"],
        "not_actual_integrated_CPU_J7": True,
        "not_model_accepted": True,
    }), flush=True)


if __name__ == "__main__":
    main()
