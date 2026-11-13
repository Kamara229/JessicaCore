"""S196: grow the *main* CALIBRI 2.0 architecture to physical S135 46.70M
and add S195 learnable neural representation area, with HONEST <=0.7 bpp.

The physical S135 artifact already proves five completed J7 transactions and
zero-initialized rank 1344 growth, but 0/5 candidate weight updates committed.
This stage validates and *repackages* that physically existing growth model,
instead of re-labelling a 43M file as 46M.

One and only one S135 original sidecar can be replaced by a losslessly
compressed sidecar. All new NRA tensors and source must be physically paid;
zeros are preserved exactly, not discarded from the neural architecture.
Two parameter counts are separately disclosed: architectural vs learned.
No J7 quality, speed, or learning claim follows from the bpp result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zlib

import torch
from safetensors.torch import load, load_file

from calibri2.s132.growth_schedule import architecture_params, MAX_RANK, TARGET
from calibri2.s132.progressive_core import (
    ProgressiveRouteCore, initial_active_indices, IDS_KEY, WEIGHTS_KEY,
)
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea

S135_RUN = 42990346609
S135_PARAMS = architecture_params(1344)     # 46,702,400 main-core parameters
PREVIOUS_PARAMS = architecture_params(896)  # 43,261,760
AREA_PARAMS = 21_900
NEXT_TOTAL_PARAMS = S135_PARAMS + AREA_PARAMS
MAX_BPP = .7


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_real_growth_provenance(report_path: Path, physical_path: Path):
    """Exactly matching successful S135 CPU+MPS artifacts, no forged pseudo-growth."""
    data = json.loads(Path(report_path).read_text())
    raw = Path(physical_path).read_bytes()
    if (data.get("stage") != "S135_STAGE2_FIVE_CYCLES_THEN_ZERO_GROWTH"
            or data.get("next_rank") != 1344
            or data.get("previous_core_architecture") != PREVIOUS_PARAMS
            or data.get("next_core_architecture") != S135_PARAMS
            or data.get("new_channels_trained") is not False
            or data.get("new_architecture_initial_up_exact_zero") is not True
            or data.get("old_456_exact_EOS_parity") is not True
            or data.get("new_128_exact_EOS_parity") is not True
            or data.get("physical_sha256") != sha256(raw)
            or data.get("newly_learned_params_from_structural_growth") != 0
            or data.get("CPU_speed_ratio", 0) < .9
            or not 0 < data.get("rate_bpp_all_runtime_included", 1) <= MAX_BPP):
        raise AssertionError("S196 requires real successful S135 46.70M physical provenance")
    records = data.get("five_J7_cycles_verified")
    if (not isinstance(records, list) or len(records) != 5
            or [r.get("cycle") for r in records] != [1,2,3,4,5]
            or any(r.get("lost_previously_correct_full") != 0
                   or r.get("lost_previously_correct_first") != 0
                   or r.get("physical_reloaded") is not True
                   or r.get("source_immutable") is not True
                   or not (r.get("transaction_committed") is True
                           or r.get("transaction_rollback_verified") is True)
                   for r in records)):
        raise AssertionError("S196 physical J7 five-cycle evidence missing or unsafe")
    original = load_file(str(physical_path), device="cpu")
    if IDS_KEY not in original or WEIGHTS_KEY not in original:
        raise AssertionError("S196 no physically inherited 46.70M tensors")
    core = ProgressiveRouteCore(rank=1344)
    core.load_active_fp16({IDS_KEY: original[IDS_KEY], WEIGHTS_KEY: original[WEIGHTS_KEY]})
    if core.architectural_new_parameter_count() != S135_PARAMS - 36_380_480:
        raise AssertionError("S196 main core not physically expanded to rank1344")
    ids = initial_active_indices(1344)
    old_ids = initial_active_indices(896)
    if (tuple(core.active_indices) != ids or len(ids) != 72
            or len(old_ids) != 48 or ids[:48] != old_ids
            or tuple(ids[-24:]) != tuple(range(896,920))):
        raise AssertionError("S196 48 inherited and 24 new active channels changed")
    with torch.no_grad():
        if int(torch.count_nonzero(core.up[:, :, 896:]).item()) != 0:
            raise AssertionError("S196 unauthorized learned weights in fresh growth channels")
    # Check same previously trained FP16 sidecar roundtrips bit-for-bit.
    got = core.export_active_fp16()
    if any(not torch.equal(got[k], original[k]) for k in (IDS_KEY, WEIGHTS_KEY)):
        raise AssertionError("S196 physical S135 FP16 growth persistence mismatched")
    return data, len(raw), original


def pack_lossless(*, physical_path: Path, area: NeuroRepresentationArea,
                  physical_out: Path, region_out: Path):
    """Lossless runtime packages: canonical stored payloads with exact SHA roundtrip."""
    if physical_out.exists() or region_out.exists():
        raise FileExistsError("S196 must not overwrite a physical experiment")
    uncompressed = Path(physical_path).read_bytes()
    packed = zlib.compress(uncompressed, level=9)
    physical_out.parent.mkdir(parents=True, exist_ok=True)
    physical_out.write_bytes(packed)
    if zlib.decompress(physical_out.read_bytes()) != uncompressed:
        raise AssertionError("S196 growth archive decompression not bit-identical")
    # Deterministic region tensors; include all *21,900* values even if zero.
    from safetensors.torch import save
    region_plain = save(area.fp16_state())
    region_packed = zlib.compress(region_plain, level=9)
    region_out.write_bytes(region_packed)
    region_back = load(zlib.decompress(region_out.read_bytes()))
    region_before = area.fp16_state()
    if (set(region_back) != set(region_before)
            or any(not torch.equal(region_back[k], region_before[k])
                   for k in region_before)):
        raise AssertionError("S196 NRA learned weights did not reload exactly")
    return {
        "growth_plain_bytes": len(uncompressed),
        "growth_packed_bytes": len(packed),
        "growth_original_sha256": sha256(uncompressed),
        "growth_payload_sha256": sha256(packed),
        "region_plain_bytes": len(region_plain),
        "region_packed_bytes": len(region_packed),
        "region_original_sha256": sha256(region_plain),
        "region_payload_sha256": sha256(region_packed),
    }


def physical_budget(*, parent_bpp: float, old_physical_bytes: int,
                    packed_growth_bytes: int, packed_region_bytes: int,
                    new_source_bytes: int, extra_manifest_bytes: int = 0):
    """Replace ONE counted S135 physical file, never discount unknown artifacts.

    All source charges are additive. A future 135M parameter denominator is
    NEVER used to excuse the current package. A separate conservative
    denominator excludes all 3.44M newly zero structural channels.
    """
    from math import isfinite
    numbers = (old_physical_bytes, packed_growth_bytes, packed_region_bytes,
               new_source_bytes, extra_manifest_bytes)
    if (not isfinite(parent_bpp) or not 0 < parent_bpp <= MAX_BPP
            or any(type(x) is not int or x < 0 for x in numbers)
            or old_physical_bytes <= 0 or packed_growth_bytes <= 0
            or packed_region_bytes <= 0):
        raise ValueError("S196 missing measured physical runtime budget")
    # Parent bpp numerator explicitly includes one S135 uncompressed sidecar.
    bits = (parent_bpp * S135_PARAMS
            - 8 * old_physical_bytes
            + 8 * (packed_growth_bytes + packed_region_bytes
                   + new_source_bytes + extra_manifest_bytes))
    if bits <= 0:
        raise AssertionError("S196 impossible physical archive accounting")
    return {
        "physical_bits": bits,
        "architectural_bpp": bits / NEXT_TOTAL_PARAMS,
        "conservative_no_zero_growth_bpp": bits / (PREVIOUS_PARAMS + AREA_PARAMS),
        "architecture_gate_le_0_7": bits / NEXT_TOTAL_PARAMS <= MAX_BPP,
        "no_zero_growth_gate_le_0_7": bits / (PREVIOUS_PARAMS + AREA_PARAMS) <= MAX_BPP,
    }


def plan_to_135m():
    """Plan MAIN-core growth only. NRA independently adds 21,900 parameters."""
    rank = 1344
    stages = []
    while rank < MAX_RANK:
        nxt = min(MAX_RANK, rank + 448)
        stages.append({
            "from_rank": rank,
            "to_rank": nxt,
            "new_main_core_params": architecture_params(nxt),
            "new_channels_initially_zero_and_not_claimed_trained": True,
            "requires_five_physically_audited_J7_cycles_before_this_growth": True,
        })
        rank = nxt
    if architecture_params(rank) != 134_999_360:
        raise AssertionError("S196 135M original main-core target drift")
    return stages


def main():
    import argparse
    a = argparse.ArgumentParser()
    a.add_argument("s135_growth_report", type=Path)
    a.add_argument("s135_growth_fp16", type=Path)
    a.add_argument("out", type=Path)
    args = a.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError("S196 cannot overwrite previous physical result")
    parent, old_size, _ = verify_real_growth_provenance(
        args.s135_growth_report, args.s135_growth_fp16)
    torch.manual_seed(196013)
    area = NeuroRepresentationArea()
    pack = pack_lossless(
        physical_path=args.s135_growth_fp16,
        area=area,
        physical_out=args.out / "S196_MAIN_CORE_46M_LOSSLESS.safetensors.zlib",
        region_out=args.out / "S196_NRA_ALL_21900_WEIGHTS_FP16.safetensors.zlib")
    source_files = [
        "research/calibri2/s195_neuro_area/representation_area.py",
        "research/calibri2/s195_neuro_area/real_cpu_integration.py",
        "research/calibri2/s195_neuro_area/test_representation_area.py",
        "research/calibri2/s196/core_growth_budget.py",
        "research/calibri2/s196/test_core_growth_budget.py",
    ]
    source_bytes = sum(Path(p).stat().st_size for p in source_files)
    # Include the manifest itself in the invoice; iterate exactly until its
    # byte size stabilizes. Canonical filenames/data are fixed in every pass.
    report = {
        "stage": "S196_REAL_S135_46M_LOSSLESS_RUNTIME_BPP_PLUS_195_NRA",
        "source_S135_pinned_GitHub_run": S135_RUN,
        "main_core_params": S135_PARAMS,
        "neural_area_params": AREA_PARAMS,
        "architectural_total_params": NEXT_TOTAL_PARAMS,
        "learned_new_growth_channels": 0,
        "new_neural_area_training_steps": 0,
        "source_model_S135_456": parent["old_456_full_correct"],
        "source_model_S135_new128": parent["new_128_full_correct"],
        "S194_584_full_J7_after_growth_not_yet_run": True,
        "physical_S135_CPU_speed": parent["CPU_speed_ratio"],
        "physical_source_files_bytes": source_bytes,
        "all_incremental_physical_runtime_file_bytes_included": True,
        "lossless_reconstructed_source_file_sha256": pack["growth_original_sha256"],
        "lossless_reconstructed_NRA_sha256": pack["region_original_sha256"],
        "old_S135_uncompressed_growth_file_replaced_not_double_counted": True,
        "not_accepted": True,
        "no_QA_TEST_no_LIVE": True,
        "proposed_further_stages": plan_to_135m(),
        **pack,
    }
    manifest_size = 0
    for _ in range(30):
        report["budget"] = physical_budget(
            parent_bpp=parent["rate_bpp_all_runtime_included"],
            old_physical_bytes=old_size,
            packed_growth_bytes=pack["growth_packed_bytes"],
            packed_region_bytes=pack["region_packed_bytes"],
            new_source_bytes=source_bytes,
            extra_manifest_bytes=manifest_size)
        body = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True).encode()
        if len(body) == manifest_size:
            break
        manifest_size = len(body)
    else:
        raise AssertionError("S196 physical manifest byte accounting did not converge")
    if not report["budget"]["architecture_gate_le_0_7"]:
        raise AssertionError("S196 FULL 46.72M architectural physical bpp exceeds 0.7")
    (args.out / "S196_46M_PHYSICAL_BPP_REBUDGET.json").write_bytes(body)
    print(json.dumps({k:v for k,v in report.items()
                      if k != "proposed_further_stages"}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
