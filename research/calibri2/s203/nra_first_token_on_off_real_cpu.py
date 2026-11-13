"""S203: read-only genuine CPU J7 paired NRA-ON/OFF first-token ablation.

The four prompts are previously *precommitted original S73 TRAIN* cases from
S202, identified solely to decide which EXTRA diagnostic decodes to capture.
No gold, expected IDs, QA, or answer-specific selector is used during inference.
The returned original ON prediction is unmodified; OFF is a separately executed
counterfactual with identical physically loaded S200 weights and all original
routing, CPU caches, neural head, Green Tunnel and input contexts unchanged.
No candidate accepted, no training, no LIVE.
"""
from __future__ import annotations

from collections import Counter

import torch
import hashlib
import json
from pathlib import Path
import sys

from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s168 import novel_train_frames_generalization_cpu as s168
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea
from calibri2.s202 import four_lost_real_cpu_j7_trace as s202

STAGE = "S203_ACTUAL_NATIVE_CPU_FIRST_TOKEN_NRA_ON_OFF_S200_DIAGNOSTIC"
# Exact immutable original TRAIN prompt hashes from successful S202 CPU J7.
# These select only the ADDITIONAL read-only ablation, NOT the model's output.
ORIGINAL_S73_TRAIN_SHA = {
    "f83736e83814ae27a25ba558e8b937d92b0460c67e947ab5f372c9254eca9a7f": "S73/73012/left",
    "f45f58b356add1302eb1502ae5f9e88100854293503dcb33ca986968125a7fa8": "S73/73012/right",
    "f21dfbced10f22a3da43267210beeaa627fa6caecf52fff845aa1586db318aa0": "S73/73015/left",
    "4597a12d3d9ca11061369fcd9c9147872e7fa076f9ee7b3875285d0d59a4cda1": "S73/73015/right",
}


def first_token_observation(on, off):
    """Compare genuinely generated ON and OFF answer IDs, without labels."""
    if not on or not off:
        raise AssertionError("S203 empty real generated answer")
    return {
        "NRA_ON_full_generated_ids": list(on),
        "NRA_OFF_full_generated_ids": list(off),
        "NRA_ON_first_token": int(on[0]),
        "NRA_OFF_first_token": int(off[0]),
        "first_token_changed_by_NRA_activation": int(on[0]) != int(off[0]),
        "full_generated_sequence_changed": list(on) != list(off),
    }


def run_paired_ablation():
    """Run unmodified guarded S202 J7; add only a separate NRA-OFF decode."""
    if len(sys.argv) < 18:
        raise SystemExit("S203 needs STEP and exact native CPU 616 J7 CLI")
    step = int(sys.argv[1])
    if step not in s202.LOSSES:
        raise AssertionError("S203 requires original immutable S200 2/4/8 checkpoint")
    out = Path(sys.argv[-1])
    original_install = NeuroRepresentationArea.install
    original_decoder = s160.cached_full_cpu_parallel
    original_s168_decoder = s168.ORIGINAL_DECODER
    if original_decoder is not original_s168_decoder:
        raise AssertionError("S203 fresh unchanged base CPU decoder required")
    installed = []
    paired = {}
    on_first = {}
    original_audit = s160.audit_cpu_outputs

    def register_area(area, model):
        result = original_install(area, model)
        installed.append(area)
        return result

    def instrument_only_four_original_TRAIN(*args, **kwargs):
        question = args[6] if len(args) > 6 else kwargs["question"]
        identity = ORIGINAL_S73_TRAIN_SHA.get(
            hashlib.sha256(question.encode("utf-8")).hexdigest())
        # ALWAYS return the unaltered original CPU decoder result.
        # The prompt digest only gates the extra, discarded diagnostic pass.
        if identity is None or not installed or not installed[0]._active:
            return original_decoder(*args, **kwargs)
        on_result = original_decoder(*args, **kwargs)
        if identity in on_first:
            if list(on_result[0]) != on_first[identity]:
                raise AssertionError("S203 repeated original NRA-ON CPU decode drift")
        else:
            on_first[identity] = list(on_result[0])
        return on_result

    def audit_and_ablate_after_phase_scope(*args, **kwargs):
        # The S115 KV-phase hook requires ONE full generation per context:
        # a second rollout inside the same scope would mislabel its prefill
        # as a continuation token. Perform read-only OFF controls AFTER the
        # original full original-TRAIN J7 audit has exited that hook.
        result = original_audit(*args, **kwargs)
        cpu, old, new, old_ids, new_ids, routes, tok, bank, copy_net, bridge, full, expansion, parent_old, parent_new, head = args
        if len(old) != 456 or len(new) != 128:
            return result
        if set(on_first) != set(ORIGINAL_S73_TRAIN_SHA.values()) or len(installed) != 1:
            raise AssertionError("S203 original full guarded J7 missing four real NRA-ON predictions")
        area = installed[0]
        if not area._active:
            raise AssertionError("S203 original physical candidate NRA must be ON")
        model, sources, first, later, pointer, lm_head, *rest = cpu
        # The extra controls are separate, discarded CPU rollouts under
        # their own fresh causal S115 scope, with the original owner selector.
        # Restore the original ending source profile for all later J7 probes.
        ending_id = new_ids[-1] if new_ids else old_ids[-1]
        ending_route = routes[ending_id]
        try:
            for key in sorted(on_first):
                if key not in old:
                    raise AssertionError("S203 ablation keyed to non-original TRAIN")
                row = old[key]
                route = routes[key]
                s160.apply_profile(sources, cpu[6], route, bank, cpu[7], cpu[8])
                features = torch.as_tensor(
                    s160.prompt_only_vector(tok, row["question"]),
                    dtype=torch.float32,
                ) if route == "S93" else None
                counters = Counter(s168.COUNTERS)
                area._active = False
                try:
                    import torch
                    with torch.inference_mode(), expansion.prefill_only(model), s160.prefill_only_tunnel(model, full):
                        with s160.frozen_cpu_phase_scope(model, bridge, route, features, phase="prefill_only"):
                            off_result = original_decoder(
                                model, sources, first, later, lm_head, tok,
                                row["question"], pointer, copy_net, head)
                finally:
                    area._active = True
                    s168.COUNTERS.clear()
                    s168.COUNTERS.update(counters)
                note = first_token_observation(on_first[key], off_result[0])
                note["original_TRAIN_id"] = key
                note["immutable_prompt_sha256"] = hashlib.sha256(
                    row["question"].encode("utf-8")).hexdigest()
                note["extra_OFF_decode_discarded_not_used_for_routing"] = True
                paired[key] = note
        finally:
            s160.apply_profile(sources, cpu[6], ending_route, bank, cpu[7], cpu[8])
        return result

    NeuroRepresentationArea.install = register_area
    s160.cached_full_cpu_parallel = instrument_only_four_original_TRAIN
    s168.ORIGINAL_DECODER = instrument_only_four_original_TRAIN
    s160.audit_cpu_outputs = audit_and_ablate_after_phase_scope
    try:
        # S202 verifies the unchanged strict real S201 456+128 protected J7
        # and its original S176 rejection. Never remove either guard.
        s202.main()
    finally:
        s160.cached_full_cpu_parallel = original_decoder
        s168.ORIGINAL_DECODER = original_s168_decoder
        s160.audit_cpu_outputs = original_audit
        NeuroRepresentationArea.install = original_install

    raw = out / f"S202_STEP_{step}_FOUR_ORIGINAL_TRAIN_CPU_TRACE.json"
    if not raw.is_file():
        raise AssertionError("S203 no actual S202 guarded 616 CPU trace")
    guard = json.loads(raw.read_text())
    if (set(paired) != set(ORIGINAL_S73_TRAIN_SHA.values())
        or guard["stage"] != s202.STAGE
        or guard["S200_CPU_gradient_steps"] != step
        or guard["model_accepted"] is not False
        or guard["S201_guarded_original_CPU_456"] != 456 - len(s202.LOSSES[step])
        or guard["S201_guarded_new128_CPU"] != 128):
        raise AssertionError("S203 missing actual protected original TRAIN ablations")
    samples = {row["id"]: row for row in guard["samples"]}
    for identity, observed in paired.items():
        row = samples[identity]
        if (observed["NRA_ON_full_generated_ids"] != row["candidate_generated_ids"]
            or observed["NRA_OFF_full_generated_ids"] != row["original_parent_generated_ids"]
            or observed["immutable_prompt_sha256"] != row["question_sha256"]):
            raise AssertionError(
                "S203 actual NRA-OFF control differs from frozen parent or NRA-ON J7")
    report = {
        "stage": STAGE,
        "S202_real_native_CPU_J7_run": 43379019263,
        "S200_gradient_step_snapshot": step,
        "full_native_J7_original_456": guard["S201_guarded_original_CPU_456"],
        "full_native_J7_new128": guard["S201_guarded_new128_CPU"],
        "previous_original_answers_lost": guard["loss_IDs"],
        "same_loaded_weights_except_NRA_activation": True,
        "same_original_real_native_CPU_decoder_and_routes": True,
        "original_NRA_ON_predictions_returned_unchanged": True,
        "all_OFF_counterfactual_predictions_discarded": True,
        "original_TRAIN_cases_only": True,
        "no_gold_used_to_choose_inference_route": True,
        "full_protected_J7_original_rejection_preserved": True,
        "no_QA_TEST_or_LIVE": True,
        "model_accepted": False,
        "cases": [paired[k] for k in sorted(paired)],
    }
    dest = out / f"S203_STEP_{step}_REAL_CPU_NRA_ON_OFF_FIRST_TOKEN.json"
    if dest.exists():
        raise FileExistsError("S203 may not overwrite diagnostic proof")
    dest.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({
        "event": STAGE, "snapshot": step,
        "NRA_ON_OFF_first_token_changes": {
            k: paired[k]["first_token_changed_by_NRA_activation"]
            for k in sorted(paired)},
        "model_accepted": False,
        "LIVE_unchanged": True,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    run_paired_ablation()
