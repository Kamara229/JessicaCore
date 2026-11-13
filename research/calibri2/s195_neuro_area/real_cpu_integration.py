"""S195-NRA real frozen 43M CPU J7 *integration* proof.

Attach the learnable 21,900-weight neural area to ALL actual decoder layers,
enable it in actual S190 cached decoding, serialize every new weight as FP16,
and verify old456/new128/new32 generation against frozen S194. No training,
model acceptance, inference-time answers, or LIVE writes. Existing S194
strict bpp <=.7 remains binding: an over-budget initial FP16 region is
reported as REJECTED, never smuggled into an accepted model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from safetensors.torch import load_file

from calibri2.s195_neuro_area.representation_area import (
    NeuroRepresentationArea, PARENT_PARAMS, DEPTH,
)
from calibri2.s194.actual_cpu_j7_bridge import (
    real_j7_baseline, make_real_584_certificate, actual_fp16_checkpoint_sha,
)


def main() -> None:
    if len(sys.argv) < 15:
        raise SystemExit("S195 requires S194 exact full 43M S190 CPU J7 CLI")
    args = list(sys.argv)
    out = Path(args[-1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S195 separate immutable result directory required")

    baseline_file = Path("inputs/s194/S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json")
    if not baseline_file.is_file():
        raise AssertionError("S195 must pin independently successful S194 CPU J7")
    pinned = json.loads(baseline_file.read_text())
    if (pinned.get("stage") != "S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER"
            or pinned.get("candidate_weight_update_executed") is not False
            or pinned.get("physical_FP16_reload_verified") is not True
            or pinned.get("model_accepted") is not False
            or pinned.get("J7Retention584_frozen_baseline", {}).get(
                "old_full_eos_correct") != 456
            or pinned.get("J7Retention584_frozen_baseline", {}).get(
                "new_full_eos_correct") != 128):
        raise AssertionError("S195 S194 physical parent provenance mismatch")
    # S190 -> S187 -> S182 -> S176 -> S168 is the genuine frozen J7
    # entrypoint. S168 imports its own load_cpu_original from S112;
    # patch THAT bound import, not unused s160.load_cpu_original.
    from calibri2.s168 import novel_train_frames_generalization_cpu as s168

    original_load = s168.load_cpu_original
    installed = []

    def load_with_region(*a, **kw):
        cpu = original_load(*a, **kw)
        if installed:
            raise AssertionError("S195 exact one neural 43M core under test")
        area = NeuroRepresentationArea().eval().install(cpu[0])
        # The entire frozen real J7 run uses the region as a physical
        # computational module, not a post-hoc router or label lookup.
        area._active = True
        installed.append(area)
        return cpu

    s168.load_cpu_original = load_with_region
    try:
        (old, new, expected128, old_targets, probes,
         probe_targets, eos, source) = real_j7_baseline(
            args, old_s87=Path(args[-2]))
    finally:
        s168.load_cpu_original = original_load
        for region in installed:
            region._active = False
    # A few J7 questions are answered by an existing prompt-only selector
    # without invoking the neural core. Count *actual* neural forwards,
    # not the number of scored questions; all 616 source outputs are still
    # captured and checked below against the frozen S194 digests.
    if len(installed) != 1 or installed[0].forward_calls < 1:
        raise AssertionError("S195 area never ran inside the genuine 43M neural CPU")
    area = installed[0]
    if area.last_coverage != tuple(range(DEPTH)):
        raise AssertionError("S195 partial Green Tunnel / region layer installation")
    if source["original_real_CPU_FULL_EOS_456"] != 456 or source[
            "new128_real_CPU_FULL_EOS"] != 128:
        raise AssertionError("S195 real frozen CPU J7 old+new retention failure")

    out.mkdir(parents=True, exist_ok=True)
    sidecar = area.save_fp16(out/"S195_NEURAL_REPRESENTATION_AREA_UNTRAINED_FP16.safetensors")
    physically_reloaded = NeuroRepresentationArea()
    physically_reloaded.load_fp16(sidecar)
    for key, value in physically_reloaded.fp16_state().items():
        if not (value == load_file(str(sidecar))[key]).all():
            raise AssertionError("S195 physical FP16 region reload changed neural weights")

    source_paths = (
        "research/calibri2/s195_neuro_area/representation_area.py",
        "research/calibri2/s195_neuro_area/test_representation_area.py",
        "research/calibri2/s195_neuro_area/real_cpu_integration.py",
    )
    area_bpp = area.all_in_bpp(
        parent_bpp=float(pinned["physical_bpp_including_S194_source"]),
        physical_area_bytes=sidecar.stat().st_size,
        additional_source_bytes=sum(Path(p).stat().st_size for p in source_paths))
    certificate = make_real_584_certificate(
        old_pred=old, old_target=old_targets, new_pred=new,
        new_target=expected128, probe_pred=probes, probe_target=probe_targets,
        eos_id=eos, source_bpp=area_bpp,
        speed_ratio=1.,  # UNTRAINED model: NOT a same-CPU candidate speed proof.
        checkpoint_sha256=actual_fp16_checkpoint_sha(
            [Path(args[-7]), Path(args[-5])]),
        physically_reloaded=True, frozen_parent=True, captured_before_gold=True)
    if (certificate.old_full_eos_correct != 456
            or certificate.new_full_eos_correct != 128
            or certificate.train32_full_eos_correct != 0
            or certificate.train32_first_correct != 16):
        raise AssertionError("S195 unchanged region accidentally altered real J7")
    if (certificate.protected_584_generation_digest != pinned[
            "J7Retention584_frozen_baseline"]["protected_584_generation_digest"]
            or certificate.train32_generation_digest != pinned[
                "J7Retention584_frozen_baseline"]["train32_generation_digest"]):
        raise AssertionError("S195 new region did not preserve exact S194 tokens")

    # An initialized region is NOT trained. Check all-inclusive budget openly;
    # failure here is expected and MUST block acceptance and further promotion.
    report = {
        "stage": "S195_NEURAL_REPRESENTATION_AREA_REAL_CPU_UNTRAINED_PROTOTYPE",
        "parent_S194_run": 43345267876,
        "real_43M_CPU_forward_neural_area_enabled": True,
        "all_12_layers_visited": list(area.last_coverage),
        "actual_CPU_forward_calls": area.forward_calls,
        "parent_43M_weights_unchanged": True,
        "new_neural_area_trainable_parameters": area.parameter_count(),
        "expanded_core_architecture_parameter_count": PARENT_PARAMS + area.parameter_count(),
        "new_area_fp16_file_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "physical_area_sidecar_bytes": sidecar.stat().st_size,
        "physical_bpp_charging_new_area_and_source": area_bpp,
        "physical_bpp_gate_le_0_7": area_bpp <= .7,
        "candidate_train_steps": 0,
        "candidate_real_43M_j7_training_done": False,
        "new32_full_EOS_improvement": 0,
        "same_CPU_candidate_speed_independently_measured": False,
        "identical_S194_full_generated_584_plus32": True,
        "old456_full_EOS": certificate.old_full_eos_correct,
        "new128_full_EOS": certificate.new_full_eos_correct,
        "new32_full_EOS": certificate.train32_full_eos_correct,
        "new32_first": certificate.train32_first_correct,
        "region_accepted": False,
        "model_accepted": False,
        "LIVE_Render_Supabase_APK_untouched": True,
    }
    result_path = out/"S195_REAL_43M_NEURO_AREA_UNTRAINED_RESEARCH.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
