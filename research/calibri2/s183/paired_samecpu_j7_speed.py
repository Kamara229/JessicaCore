"""CALIBRI S183: independent paired actual frozen CPU J7 speed: S176 vs S182.

Runs after S182's real full 456+128 frozen inference while all weights remain
installed. Compare EXACT original S176 selector against S182 source shadow on
the SAME packed model, same S93 route and same literal TRAIN prompts. Alternate
execution order AB/BA; never use gold or correctness for sample selection.
No training, QA/TEST, Jessica chat, LIVE, Render, Supabase or APK changes.
"""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from calibri2.s182 import real_cpu_j7_shadow_guard as s182
from calibri2.s160 import real_parallel_slots_j7 as s160

ORIGINAL_PAIRED_CPU_SPEED = s160.paired_cpu_ratio
BENCHMARK_SOURCE = Path("research/calibri2/s183/paired_samecpu_j7_speed.py")
ALL_TIMES = None
S179_FROZEN_PRED = {}

def sha(q):
    return hashlib.sha256(q.encode("utf-8")).hexdigest()

def chosen_keys(old_ids, routes):
    """Predeclare balanced groups by identity, no gold, no outcomes."""
    def take(prefix, n):
        return [key for key in old_ids if key.startswith(prefix) and routes[key] == "S93"][:n]
    s30, s54 = take("S30/", 12), take("S54/", 12)
    other = [key for key in old_ids
             if not key.startswith(("S30/", "S54/")) and routes[key] == "S93"][:12]
    if len(s30) != 12 or len(s54) != 12 or len(other) != 12:
        raise AssertionError("S183 requires 12+12 matched plus 12 untouched S93 controls")
    return {"S30": s30, "S54": s54, "non_shadow_control": other}

def describe(records):
    """Measured speed ratio = original median elapsed / shadow median elapsed."""
    if len(records) < 2 or any(a <= 0 or b <= 0 for a,b in records):
        raise AssertionError("S183 paired CPU samples missing or invalid")
    base = [a for a,_ in records]
    trial = [b for _,b in records]
    med_base = statistics.median(base)
    med_trial = statistics.median(trial)
    return {"paired_trials": len(records),
            "s176_median_cpu_seconds": med_base,
            "s182_shadow_median_cpu_seconds": med_trial,
            "s182_over_s176_speed": med_base/med_trial}

def independent_speed(cpu, old, old_ids, routes, tok, bank,
                      copy_net, bridge, full, expansion, parent_old, head):
    """Same packed model and profile, real frozen full autoregressive J7."""
    global ALL_TIMES
    if ALL_TIMES is not None:
        raise AssertionError("S183 paired speed cannot be measured twice")
    if len(s182.ALL_PRED.get("old", {})) != 456:
        raise AssertionError("S183 full 456 actual model answers not yet frozen")
    model, sources, first, later, pointer, lm_head, *_ = cpu
    if model.training:
        raise AssertionError("S183 model must be in eval mode")
    if len(S179_FROZEN_PRED) != 456:
        raise AssertionError("S183 missing immutable S179 original CPU baseline")
    groups = chosen_keys(old_ids, routes)
    chosen = [key for group in groups.values() for key in group]
    metrics = {}
    base_guard = s182.REAL_S175_BASE_GUARD
    from calibri2.s175 import negative_owner_guard_real_j7 as s175
    if base_guard is None or s175.OLD_S172 is not s182.shadow_selector:
        raise AssertionError("S183 must benchmark S176/S182 exact selectors")
    try:
        # All target identities predeclared from TRAIN source; no expected IDs.
        samples = []
        for group, keys in groups.items():
            for key in keys:
                q = old[key]["question"]
                key_sha = sha(q)
                if key_sha not in S179_FROZEN_PRED:
                    raise AssertionError("S183 missing baseline original CPU trace")
                samples.append((group, key, q, key_sha))
        # Same route throughout (S93). Profile set once before both variants.
        s160.apply_profile(sources, cpu[6], "S93", bank, cpu[7], cpu[8])

        def call(q, policy):
            s175.OLD_S172 = base_guard if policy == "base" else s182.shadow_selector
            features = torch.as_tensor(s160.prompt_only_vector(tok, q), dtype=torch.float32)
            with torch.inference_mode(), expansion.prefill_only(model), s160.prefill_only_tunnel(model, full):
                with s160.frozen_cpu_phase_scope(
                        model, bridge, "S93", features, phase="prefill_only"):
                    start = time.perf_counter()
                    generated, _ = s160.cached_full_cpu_parallel(
                        model, sources, first, later, lm_head, tok, q,
                        pointer, copy_net, head)
                    elapsed = time.perf_counter() - start
            return list(generated), elapsed
        # Equal warmup on one actual source sample; do not include in results.
        warm = samples[0][2]
        call(warm, "base")
        call(warm, "shadow")
        times = {name: [] for name in groups}
        per_sha = []
        # 2 complete paired sweeps with alternate order, no randomization or
        # calibration. Never adapt sample selection to observed performance.
        for cycle in (0, 1):
            for n, (group, key, q, key_sha) in enumerate(samples):
                order = ("base", "shadow") if (cycle+n)%2 == 0 else ("shadow", "base")
                observations = dict((policy, call(q, policy)) for policy in order)
                expected_baseline = S179_FROZEN_PRED[key_sha]
                expected_shadow = s182.ALL_PRED["old"][key_sha]
                if observations["base"][0] != expected_baseline:
                    raise AssertionError("S183 original unmodified S176 CPU full/EOS changed")
                if observations["shadow"][0] != expected_shadow:
                    raise AssertionError("S183 real S182 CPU shadow full/EOS changed")
                if group == "non_shadow_control" and observations["base"][0] != observations["shadow"][0]:
                    raise AssertionError("S183 control had non-shadow answer drift")
                base_t = observations["base"][1]
                shadow_t = observations["shadow"][1]
                times[group].append((base_t, shadow_t))
                per_sha.append({"prompt_sha256": key_sha,
                                "group": group, "cycle": cycle,
                                "s176_cpu_seconds": base_t,
                                "s182_cpu_seconds": shadow_t,
                                "order": list(order)})
        all_records = [pair for pairs in times.values() for pair in pairs]
        metrics = {name: describe(records) for name, records in times.items()}
        metrics["all_36_source_cases"] = describe(all_records)
        if not (len(per_sha) == 72 and
                all(metrics[k]["paired_trials"] == 24 for k in groups)):
            raise AssertionError("S183 no full paired 36x2 CPU coverage")
        ALL_TIMES = {"groups": metrics, "paired_SHA_only": per_sha,
                     "s176_s182_outputs_verified_against_real_frozen_J7": True,
                     "s93_route_only": True, "candidate_selection_without_gold": True}
    finally:
        s175.OLD_S172 = s182.shadow_selector
    return ALL_TIMES

def hook(*args, **kwargs):
    # Preserve original S168 old-S101 speed certification. Independently
    # compare S176 same J7 vs S182 on the exact same CPU *afterward*.
    old_result = ORIGINAL_PAIRED_CPU_SPEED(*args, **kwargs)
    independent_speed(*args, **kwargs)
    return old_result

def main():
    global S179_FROZEN_PRED
    if len(sys.argv) < 5:
        raise SystemExit("S183 requires original S182 frozen J7 input args")
    source = Path(sys.argv[-3])
    out = Path(sys.argv[-1])
    if (out/"S183_PAIRED_SAME_CPU_SOURCE_SHADOW_BENCHMARK.json").exists():
        raise FileExistsError("S183 shadow benchmark immutable output")
    frozen = json.loads(source.read_text())
    if frozen.get("stage")!="S179_TRACE_ONLY" or frozen.get("labels_used") is not False or frozen.get("number")!=456:
        raise AssertionError("S183 original label-free S179 real CPU source missing")
    S179_FROZEN_PRED = {r["prompt_sha256"]: r["full_generated_ids"] for r in frozen["cases"]}
    if len(S179_FROZEN_PRED)!=456:
        raise AssertionError("S183 456 baseline trace coverage failed")
    s160.paired_cpu_ratio = hook
    try:
        s182.main()   # FULL original physical real CPU J7; no model updates.
    finally:
        s160.paired_cpu_ratio = ORIGINAL_PAIRED_CPU_SPEED
    if ALL_TIMES is None:
        raise AssertionError("S183 no real independent same-CPU timing collected")
    s182_report=json.loads((out/"S182_REAL_SOURCE_SHADOW_456_128_GUARD.json").read_text())
    base_bpp=s182_report["all_in_bpp_charge_new_source"]
    all_bpp=base_bpp + 8 * BENCHMARK_SOURCE.stat().st_size/s182.DENOM
    groups=ALL_TIMES["groups"]
    matched=[x for x in ALL_TIMES["paired_SHA_only"] if x["group"]!="non_shadow_control"]
    matched_ratio=describe([(x["s176_cpu_seconds"],x["s182_cpu_seconds"]) for x in matched])["s182_over_s176_speed"]
    passed=(s182_report["retention_passed"] and
            s182_report["old_original_train_after"]==454 and
            s182_report["new_train128_actual_full_EOS_correct"]==128 and
            s182_report["old_losses"]==0 and s182_report["first_losses"]==0 and
            all_bpp <= .7 and matched_ratio >= .9 and
            groups["all_36_source_cases"]["s182_over_s176_speed"] >= .9 and
            groups["non_shadow_control"]["s182_over_s176_speed"] >= .9)
    report={
        "event":"S183_REAL_CPU_S176_VS_S182_PAIRED_SAME_CPU_SHADOW_J7",
        "parent_S182_run":43331434088,
        "same_real_43M_cpu_compressed_weights":True,
        "same_S93_route_and_36_frozen_source_prompts":True,
        "two_paired_AB_BA_sweeps":True,
        "no_training_or_extra_model_weights":True,
        "baseline_S176_vs_shadow_S182":groups,
        "matched_combined_speed_ratio":matched_ratio,
        "physical_bpp_including_benchmark_code":all_bpp,
        "correct_456":s182_report["old_original_train_after"],
        "correct_new128":s182_report["new_train128_actual_full_EOS_correct"],
        "old_losses":s182_report["old_losses"],
        "independent_speed_gate_pass":passed,
        "model_accepted":False,
        "LIVE_Jessica_Render_Supabase_unchanged":True,
        "per_prompt_SHA_only":ALL_TIMES["paired_SHA_only"],
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"S183_PAIRED_SAME_CPU_SOURCE_SHADOW_BENCHMARK.json").write_text(
        json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="per_prompt_SHA_only"},sort_keys=True),flush=True)
    if not passed:
        raise AssertionError("S183 reject paired S176/S182 speed or protected 456+128/bpp")
if __name__=="__main__":
    main()
