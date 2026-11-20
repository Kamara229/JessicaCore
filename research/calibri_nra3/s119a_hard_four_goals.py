"""S119a: irreversible four-goal research constraints for NRA3-R.

RESEARCH ONLY. This is a reusable fail-closed accounting/evidence contract,
NOT an accepted physical Calibri checkpoint or a measured CPU benchmark.
No new model, no gold, no training, no LIVE/Jessica/Render/Supabase.
Only original actual unique TRAINED core weights enter the denominator:
do not silently count deterministic/fixed zero adapters as source weights.

A candidate cannot be considered for activation until all four tests use
physical artifacts: packed <=0.700 bpp, >=0.85x matched CPU throughput
(target >=0.90), exact 616 original J7 token/EOS retention, and a positive
independent sentence-paired understanding gain. Reserve bytes BEFORE training.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import random

SOURCE_UNIQUE_TRAINED_WEIGHTS = 36_380_480
BPP_MAX = 0.700
CPU_MIN = 0.850
CPU_TARGET = 0.900
PROTECTED_J7 = 616
BOOTSTRAP_N = 4000


class GateFailure(ValueError):
    pass


def check(condition: bool, reason: str) -> None:
    if not condition:
        raise GateFailure("NRA3 S119a HARD GATE: " + reason)


def packed_budget_bytes(source_unique_weights: int = SOURCE_UNIQUE_TRAINED_WEIGHTS) -> int:
    check(type(source_unique_weights) is int and
          source_unique_weights == SOURCE_UNIQUE_TRAINED_WEIGHTS,
          "denominator must be unique genuinely trained original source weights; "
          "do not inflate to 43M using fixed/zero structure")
    return int(BPP_MAX * source_unique_weights // 8)


def physical_bundle(root: Path, sha256_inventory: dict[str, str],
                    *, prospective_extra_bytes: int = 0) -> dict:
    """Enumerate ALL runtime model/GT/NRA3/adapter/metadata files from loader inventory.

    Caller must first independently confirm inventory is complete against the
    actual model loader. This function does not itself prove loader completeness.
    """
    check(bool(sha256_inventory), "empty runtime inventory")
    check(type(prospective_extra_bytes) is int and prospective_extra_bytes >= 0,
          "invalid prospective byte reservation")
    root = Path(root).resolve(strict=True)
    check(root.is_dir(), "bundle root must be a real directory")
    seen = set()
    total = 0
    parts = []
    for name, expected in sorted(sha256_inventory.items()):
        check(isinstance(name, str) and name and
              isinstance(expected, str) and len(expected) == 64 and
              all(x in "0123456789abcdef" for x in expected),
              "noncanonical inventory entry")
        rel = Path(name)
        check(not rel.is_absolute() and ".." not in rel.parts and
              len(rel.parts) > 0, "path escape in runtime inventory")
        item = root / rel
        check(not item.is_symlink() and item.is_file(),
              "missing file or symlink in runtime inventory")
        resolved = item.resolve(strict=True)
        check(resolved.is_relative_to(root) and resolved not in seen,
              "duplicate file or root escape")
        seen.add(resolved)
        raw = item.read_bytes()
        check(sha256(raw).hexdigest() == expected, "physical bundle SHA mismatch: " + name)
        check(len(raw) > 0, "empty runtime component")
        total += len(raw)
        parts.append({"file": name, "bytes": len(raw), "sha256": expected})
    projected = total + prospective_extra_bytes
    limit = packed_budget_bytes()
    check(projected * 8 <= SOURCE_UNIQUE_TRAINED_WEIGHTS * BPP_MAX,
          "physical bundle plus reserved adapter exceeds 0.700 bpp")
    return {"physical_bytes": total, "reserved_new_bytes": prospective_extra_bytes,
            "projected_all_in_bpp": 8 * projected / SOURCE_UNIQUE_TRAINED_WEIGHTS,
            "max_physical_bytes": limit, "parts": parts,
            "loader_inventory_completeness_STILL_REQUIRES_physical_audit": True,
            "NOT_model_acceptance": True}


def exact_protected_j7(baseline: dict[str, tuple], candidate: dict[str, tuple]) -> dict:
    """Both dictionaries must be produced by true original CPU full+EOS replay.

    Each tuple = (prompt_SHA256, generated_original_token_ids_including_EOS).
    This deliberately protects even baseline mistakes from accidental drift.
    """
    check(len(baseline) == PROTECTED_J7 and len(candidate) == PROTECTED_J7,
          "616 full protected original J7 records required")
    check(set(baseline) == set(candidate), "J7 identity changed")
    for key in baseline:
        pre = baseline[key]
        post = candidate[key]
        check(isinstance(pre, tuple) and isinstance(post, tuple) and
              len(pre) == len(post) == 2 and
              isinstance(pre[0], str) and len(pre[0]) == 64 and
              isinstance(pre[1], tuple) and len(pre[1]) > 0,
              "invalid original J7 record")
        check(pre == post, "protected original full token/EOS drift: " + key)
    return {"identical_full_token_EOS": 616, "losses": 0,
            "actual_CPU_replay_must_supply_records": True}


def matched_cpu_speed(pairs: list[dict]) -> dict:
    """Matched full decode pairs; seconds are actually measured externally.

    Same prompt, machine, hardware, threads, decoding path, and emitted tokens.
    Minimum 8 prompts, >=3 repeated pairs each, >=24 total runs.
    """
    check(len(pairs) >= 24, "need >=24 paired CPU observations")
    prompts = Counter()
    base_total = cand_total = 0.0
    for item in pairs:
        check(set(item) == {"prompt_sha256", "environment_sha256", "baseline_s",
                            "candidate_s", "baseline_token_ids", "candidate_token_ids"},
              "CPU record must have complete paired provenance")
        prompt, env = item["prompt_sha256"], item["environment_sha256"]
        check(isinstance(prompt, str) and len(prompt) == 64 and
              isinstance(env, str) and len(env) == 64,
              "CPU prompt and environment SHA required")
        b, c = item["baseline_s"], item["candidate_s"]
        check(type(b) in (float, int) and type(c) in (float, int) and
              0 < b < 3600 and 0 < c < 3600,
              "invalid paired full-decode CPU durations")
        check(tuple(item["baseline_token_ids"]) == tuple(item["candidate_token_ids"]) and
              len(item["baseline_token_ids"]) > 0,
              "CPU work must be token-comparable")
        prompts[prompt] += 1
        base_total += b
        cand_total += c
    check(len(prompts) >= 8 and min(prompts.values()) >= 3,
          "need eight source prompts each repeated at least 3 times")
    check(len({x["environment_sha256"] for x in pairs}) == 1,
          "CPU platform and environment drift")
    ratio = base_total / cand_total
    check(ratio >= CPU_MIN - 1e-12,
          "CPU is >15% slower than matched reference")
    return {"paired_full_decode_speed_ratio": ratio, "absolute_floor": CPU_MIN,
            "design_target": CPU_TARGET, "target_met": ratio >= CPU_TARGET,
            "CPU_measured_externally_not_by_this_contract": True}


def independent_relation_gain(source_rows: list[tuple[int, int, int]], *,
                              fixed_seed: int = 11920261121) -> dict:
    """(source-row independent slots, baseline correct, candidate correct).

    Only use on one FIRSTREAD clean cohort after all checkpoints are frozen.
    Bootstrap resamples ORIGINAL SENTENCES, never role edges individually.
    """
    check(len(source_rows) >= 100, "minimum 100 unseen sentence clusters")
    slots = old = new = 0
    for n, a, b in source_rows:
        check(type(n) is int and type(a) is int and type(b) is int and
              n > 0 and 0 <= a <= n and 0 <= b <= n,
              "invalid independent sentence-level score")
        slots += n
        old += a
        new += b
    check(slots >= 200, "insufficient independent relation slots")
    rng = random.Random(fixed_seed)
    diffs = []
    for _ in range(BOOTSTRAP_N):
        num = den = 0
        for __ in range(len(source_rows)):
            n, a, b = source_rows[rng.randrange(len(source_rows))]
            num += b - a
            den += n
        diffs.append(num / den)
    diffs.sort()
    ci_low = diffs[int(.025 * (len(diffs) - 1))]
    ci_high = diffs[int(.975 * (len(diffs) - 1))]
    delta = (new-old) / slots
    check(delta > 0 and ci_low > 0,
          "no statistically supported positive independent understanding gain")
    return {"old": old, "new": new, "slots": slots, "delta_pp": 100*delta,
            "paired_sentence_bootstrap_95pct_pp": [100*ci_low,100*ci_high],
            "firstread_must_be_immutable_and_never_reused_for_tuning": True,
            "NOT_autonomous_event_accuracy": True}


def candidate_acceptance(*_args, **_kwargs):
    """S119a is only synthetic contract code; physical acceptance is forbidden."""
    raise GateFailure("S119a CANNOT activate or accept any research candidate")
