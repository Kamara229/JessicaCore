"""S181 source-only paired SHADOW audit of S30/S54 lexical candidates.

Generate 128 source prompts with byte-exact literal templates and parameters
extracted by AST from the original S30/S54 research source, never model code.
Perform prompt-only candidate generation FIRST. Only afterward join S87
frozen 456 source hashes/labels and S179 completed real CPU generated ids.
No model run, selector, answer ID lookup, training or LIVE modifications.
328 other source prompts are not reconstructed here; only their baseline
real CPU outputs are preserved. Thus this is NOT a full 456 parser trial.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from calibri2.s181.source_grammar_shadow import (
    S30_TEMPLATES, S54_TEMPLATES, shadow_candidate,
)

def source_ast_literal(path: Path, name: str, function: str | None = None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = tree.body
    if function:
        candidates = [x for x in nodes if isinstance(x, ast.FunctionDef) and x.name == function]
        if len(candidates) != 1:
            raise AssertionError("S181 source function drift")
        nodes = candidates[0].body
    for node in nodes:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("S181 literal source missing: " + name)

def reconstruct_sources(root: Path) -> list[tuple[str, str, str]]:
    s30 = root / "research/calibri2/s30/matched_dense_fp32_train.py"
    s54 = root / "research/calibri2/s54/diverse_train_preserve_S52_full_sequences.py"
    patterns = source_ast_literal(s30, "patterns", "make_training_pairs")
    palette = source_ast_literal(s30, "TRAIN_ELIGIBLE")
    s54_frames = source_ast_literal(s54, "NEW_TRAIN_FRAMES")
    colors = source_ast_literal(s54, "COLOR_PAIRS")
    owners = source_ast_literal(s54, "OWNERS")
    if (tuple(patterns) != S30_TEMPLATES or tuple(s54_frames) != S54_TEMPLATES
            or len(palette) != 4 or len(colors) != 4 or len(owners) != 8):
        raise AssertionError("S181 S30/S54 literal source drift")
    out = []
    for ci, (color, nouns) in enumerate(palette.items()):
        if len(nouns) != 2 or nouns[0] == nouns[1]:
            raise AssertionError("S181 S30 changed noun pairing")
        a, b = f"FPTrainL{ci:01d}", f"FPTrainR{ci:01d}"
        for i in range(8):
            left, right = f"{color} {nouns[0]}", f"{color} {nouns[1]}"
            template = patterns[(i + ci) % len(patterns)]
            owner, other = f"{a}{i:02d}", f"{b}{i:02d}"
            for side, vleft, vright, gold in (
                    ("left", left, right, left), ("right", right, left, right)):
                q = template.format(i=i, a=owner, b=other, left=vleft, right=vright)
                out.append((f"S30/{5000 + ci * 8 + i}/{side}", q, gold))
    for ci, (color, left, right) in enumerate(colors):
        for i, (a, b) in enumerate(owners):
            template = s54_frames[i]
            pair = 9000 + ci * 8 + i
            vl, vr = f"{color} {left}", f"{color} {right}"
            for side, l, r, gold in (
                    ("left", vl, vr, vl), ("right", vr, vl, vr)):
                q = template.format(i=pair, a=a, b=b, left=l, right=r)
                out.append((f"S54/{pair}/{side}", q, gold))
    if len(out) != 128 or len({q for _, q, _ in out}) != 128:
        raise AssertionError("S181 must regenerate exact unique 128 prompts")
    return out

def sha(q: str) -> str:
    return hashlib.sha256(q.encode("utf-8")).hexdigest()

def offline_posthoc(root: Path, frozen_trace: dict, frozen_s87: dict) -> dict:
    """Freeze all candidate phrases BEFORE accessing offline S87 labels."""
    sources = reconstruct_sources(root)
    proposals = {key: (sha(q), shadow_candidate(q)) for key, q, _gold in sources}
    if any(p is None for _, p in proposals.values()):
        raise AssertionError("S181 source shadow parse abstained unexpectedly")
    # Inference/candidate stage ends here. No gold or correctness was consulted.
    if (frozen_trace.get("stage") != "S179_TRACE_ONLY"
            or frozen_trace.get("number") != 456
            or frozen_trace.get("labels_used") is not False
            or frozen_s87.get("stage") != "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT"):
        raise AssertionError("S181 immutable provenance violated")
    rows = frozen_s87.get("all_456_per_identity_results", [])
    traces = frozen_trace.get("cases", [])
    frozen = {row["id"]: row for row in rows}
    actual = {row["prompt_sha256"]: row for row in traces}
    if len(frozen) != 456 or len(actual) != 456:
        raise AssertionError("S181 source 456 trace coverage missing")
    if {r["question_sha256"] for r in rows} != set(actual):
        raise AssertionError("S181 complete baseline frozen corpus SHA drift")
    if not {key for key, _, _ in sources} <= set(frozen):
        raise AssertionError("S181 missing source identity")
    counts = Counter()
    sha_only = []
    for key, q, gold in sources:
        h, proposal = proposals[key]
        row = frozen[key]
        if h != row["question_sha256"]:
            raise AssertionError("S181 exact source prompt SHA mismatch " + key)
        if proposal != gold:
            raise AssertionError("S181 source prompt-only candidate disagreed with labeled TRAIN answer")
        frozen_decode = actual[h]["full_generated_ids"]
        was_correct = frozen_decode == row["target_ids"]
        counts["source_total"] += 1
        counts["source_S30" if key.startswith("S30/") else "source_S54"] += 1
        counts["source_frozen_correct" if was_correct else "source_frozen_wrong"] += 1
        if not was_correct:
            counts["lexical_source_could_recover_if_integrated"] += 1
            sha_only.append({
                "prompt_sha256": h,
                "family": key.split("/", 1)[0],
                "full_eos_frozen_wrong": True,
                "shadow_phrase_matches_offline_source_answer": True,
                "integrated_model_tested": False,
            })
    if counts["source_total"] != 128:
        raise AssertionError("S181 missing original source records")
    frozen_correct = sum(actual[r["question_sha256"]]["full_generated_ids"] == r["target_ids"] for r in rows)
    if frozen_correct != 428:
        raise AssertionError("S181 real CPU baseline 428/456 drift")
    return {
        "stage": "S181_SHADOW_SOURCE_128_IN_FROZEN_456_OFFLINE",
        "source_s179_real_CPU_run": 43323981995,
        "all_frozen_train_outputs_checked": 456,
        "frozen_full_eos_correct": 428,
        "shadow_source_regenerated_and_SHA_verified": 128,
        "other_source_prompt_texts_not_generated": 328,
        "no_parser_claim_for_other_328": True,
        "shadow_candidate_is_prompt_only_and_offline": True,
        "source_metrics": dict(sorted(counts.items())),
        "potential_recovery_sha_only": sha_only,
        "genuine_integrated_gain": None,
        "genuine_integrated_forgetting": None,
        "gold_join_after_candidate_generation_only": True,
        "model_accepted": False,
        "LIVE_untouched": True,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("frozen_s179_traces", type=Path)
    p.add_argument("frozen_s87_report", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("S181 immutable shadow source report already exists")
    root = Path(__file__).resolve().parents[3]
    result = offline_posthoc(root,
        json.loads(a.frozen_s179_traces.read_text()),
        json.loads(a.frozen_s87_report.read_text()))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"event": "S181_SOURCE_128_FROZEN_456_OFFLINE",
        "source_metrics": result["source_metrics"],
        "all_frozen_456_baseline": 428, "model_accepted": False}), flush=True)

if __name__ == "__main__":
    main()
