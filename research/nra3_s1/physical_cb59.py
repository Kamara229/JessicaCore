"""NRA2 3.0 S1: genuine CALIBRI CB59_K32V8 physical archive bridge.

Uses the SAME experimental direct_codebook_v59 5-bit/8-vector quantizer,
not a bit-rate estimate. Serialized bytes include metadata, 1D source
weights, row scales, FP16 codebooks, CRC and SHA256. Quantized quality is
NOT guaranteed, and this reference loader cold-reconstructs dense FP32
for testing only: it is NOT native compressed inference or a CPU speed
qualification. No training or existing model weights are edited.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

# Explicit required path: PYTHONPATH=.:research:research/calibri
from direct_codebook_v59 import CB59, encode, unpack5


def _require(condition, reason):
    if not condition:
        raise ValueError("NRA3 CB59 " + reason)


def unique_state(area):
    state = area.state_dict()
    _require(state and all(v.is_floating_point() and v.dtype == torch.float32
                           and torch.isfinite(v).all() and v.is_contiguous()
                           for v in state.values()),
             "requires finite complete dense float32 pre-quantization state")
    # Require all matrices to be genuine paid CB59 operators: every 2D
    # dimension must have ncols multiple of 8. The all-zero case is NOT
    # silently free. Dense full runtime is training-only.
    for name, value in state.items():
        _require(value.ndim in (1, 2), "unsupported 0D/3D operator: " + name)
        if value.ndim == 2:
            _require(value.shape[1] % 8 == 0
                     and bool((value != 0).any().item()),
                     "non-CB59 or degenerate 2D source: " + name)
    _require(sum(v.numel() for v in state.values()) == area.parameter_count(),
             "source parameter count mismatch")
    return {k: v.detach().cpu().contiguous() for k, v in state.items()}


def encode_area(area):
    source = unique_state(area)
    archive, meta, diagnostics = encode(source)
    saved = CB59(archive, expected_keys=source)
    _require(saved.hash() == hashlib.sha256(archive).hexdigest(),
             "physical archive SHA mismatch")
    _require(meta == saved.meta and
             saved.meta["original_unique_parameter_count"] == area.parameter_count(),
             "physical metadata identity/count mismatch")
    _require(all(d["dtype"] == "CB59_K32V8"
                 for d in saved.meta["tensor_descriptors"].values()
                 if len(d["shape"]) == 2),
             "some matrix weights were not physically CB59")
    _require(len(saved.meta["tensor_descriptors"]) == len(source),
             "missing unique parameter tensor")
    return archive, meta, diagnostics


def reconstruct_state(archive: bytes, area):
    """Slow dense reference only; no native C kernel or speed claim."""
    _require(isinstance(archive, bytes), "immutable physical bytes required")
    saved = CB59(archive, expected_keys=area.state_dict().keys())
    _require(saved.meta["original_unique_parameter_count"] == area.parameter_count(),
             "physical parameter count incorrect")
    state = {}
    for name, reference in area.state_dict().items():
        if name in saved.raw:
            rebuilt = np.array(saved.raw[name], dtype=np.float32, copy=True)
        elif name in saved.handles:
            handle = saved.handles[name]
            num = int(np.prod(handle.shape)) // 8
            codes = unpack5(handle.packed.tobytes(), num)
            rebuilt = (handle.centroids[codes].reshape(handle.shape)
                       * handle.row_scales[:, None]).astype(np.float32)
        else:
            raise ValueError("NRA3 CB59 missing source " + name)
        _require(tuple(rebuilt.shape) == tuple(reference.shape) and
                 np.isfinite(rebuilt).all(), "invalid decoded weight " + name)
        state[name] = torch.from_numpy(rebuilt.copy())
    _require(set(state) == set(area.state_dict()), "incomplete reconstructed state")
    return state


def save_and_validate(area, path: Path):
    """Archive must be sealed, physically reopened, and zero-gate verified."""
    path = Path(path)
    _require(not path.exists(), "never overwrite prior research artifact")
    archive, metadata, diagnostics = encode_area(area)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(archive)
    physical = path.read_bytes()
    _require(physical == archive, "physical disk reload mismatch")
    reference = reconstruct_state(physical, area)
    # Evidence only: do not load quantized weights into LIVE or Calibri.
    source = area.state_dict()
    errors = []
    for k in source:
        orig = source[k].detach().cpu().float()
        decoded = reference[k]
        energy = (orig**2).sum().item()
        squared = ((orig - decoded)**2).sum().item()
        errors.append((k, squared, energy))
    num = area.parameter_count()
    report = {
        "stage": "NRA2_3_S1_CB59_PHYSICAL_RESEARCH_UNTRAINED",
        "physical_archive": path.name,
        "sha256": hashlib.sha256(physical).hexdigest(),
        "complete_new_area_bytes": len(physical),
        "unique_new_area_parameters": num,
        "actual_new_area_bpp": 8 * len(physical) / num,
        "all_matrix_operators_CB59_K32V8": True,
        "complete_source_tensor_count": len(source),
        "matrix_tensor_count": len(CB59(physical).handles),
        "max_relative_matrix_rmse": max((math.sqrt(s/e) for _, s, e in errors if e),
                                        default=0.0),
        "full_weight_relative_rmse": math.sqrt(sum(s for _,s,_ in errors) /
                                               max(1e-30,sum(e for _,_,e in errors))),
        "source_only_codebook_fitted": True,
        "grammar_trained": False,
        "semantic_trained": False,
        "relation_trained": False,
        "native_packed_CPU_kernel_verified": False,
        "native_full_Calibri_CPU_616_J7_verified": False,
        "green_tunnel_weights_in_this_archive": False,
        "candidate_accepted": False,
        "output_gate": 0,
        "not_a_full_parent_plus_GT_budget_certificate": True,
        "untrained_quantization_not_language_generalization": True,
    }
    return report


def budget_for_paid_components(*, parent_physical_bytes: bytes,
                               parent_unique_parameters: int,
                               nra_physical_bytes: bytes,
                               existing_gt_physical_bytes: bytes | None = None,
                               parent_includes_gt: bool = True,
                               package_manifest_bytes: bytes = b""):
    """Audits supplied REAL source/GT/NRA bytes; never estimated tensor RAM.

    If parent_includes_gt, passing separate GT bytes is forbidden to avoid
    double counting. If GT not in parent, its actual physical bytes required.
    Caller must verify the supplied parent's checkpoint SHA separately;
    this function does not authenticate parent architecture/quality.
    """
    _require(isinstance(parent_physical_bytes, bytes) and
             len(parent_physical_bytes)>0 and
             isinstance(parent_unique_parameters, int) and
             parent_unique_parameters>0, "missing physical parent")
    _require(isinstance(nra_physical_bytes, bytes), "missing NRA physical bytes")
    # Count the PHYSICALLY STORED variant, not both raw and entropy files.
    # S61-style envelopes are validated and decoded only for metadata;
    # the stored byte denominator remains the actual envelope file size.
    if nra_physical_bytes.startswith(b"N3CB61A1"):
        from research.nra3_s1.entropy_envelope import restore_lossless_cb59
        nra = CB59(restore_lossless_cb59(nra_physical_bytes))
    else:
        nra = CB59(nra_physical_bytes)
    _require(isinstance(parent_includes_gt, bool), "GT membership unknown")
    if parent_includes_gt:
        _require(existing_gt_physical_bytes is None,
                 "GT already charged inside parent; must not double-count")
        gt_bytes = 0
    else:
        _require(isinstance(existing_gt_physical_bytes,bytes)
                 and len(existing_gt_physical_bytes)>0,
                 "GT physical sidecar required when not in parent")
        gt_bytes=len(existing_gt_physical_bytes)
    _require(isinstance(package_manifest_bytes, bytes), "invalid manifest bytes")
    total=len(parent_physical_bytes)+len(nra_physical_bytes)+gt_bytes+len(package_manifest_bytes)
    n=parent_unique_parameters+nra.meta["original_unique_parameter_count"]
    return {
        "physical_parent_sha256": hashlib.sha256(parent_physical_bytes).hexdigest(),
        "physical_new_nra_sha256": hashlib.sha256(nra_physical_bytes).hexdigest(),
        "physical_existing_gt_sha256": (
            hashlib.sha256(existing_gt_physical_bytes).hexdigest()
            if existing_gt_physical_bytes is not None else None),
        "whole_package_bytes":total,
        "actual_whole_bpp":8*total/n,
        "rate_le_0_7":(8*total/n)<=.7,
        "green_tunnel_accounted_exactly_once":True,
        "weight_quality_and_native_inference_not_proven": True,
    }
