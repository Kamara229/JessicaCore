"""S87-J3: balanced six-family TRAIN-only real-MPS joint TRAIN of paid CB59 core dictionaries and internal tunnel.

Scientific feasibility pilot; no model promotion, no independent QA/test,
no FP32-pretrained teacher, no new LIVE service. A saved prototype is NOT
an accepted Calibri version until 456 TRAIN retention, full CPU/MPS parity,
speed, and a new preregistered independent evaluation have passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct
import time
import zlib

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS, MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from v186.pairs import build_pairs
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87.internal_green_tunnel import (
    IntegratedGreenTunnel, select_paid_core_codebooks)
from calibri2.s87.aligned_loss import aligned_joint_step
from v185.codec import replace_discrete

SEED = 870087
TUNNEL_LAYERS = (3, 9)
TUNNEL_RANK = 8
MIN_TRAIN_STEPS = 1
MAX_TRAIN_STEPS = 64
MAX_BPP = 0.7


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def lossless_cb61(new_cb59: bytes):
    """Rebuild CB61ZL01 from changed source; old S61 encoder pins original SHA."""
    header = struct.Struct("<8sQI32s")
    packed = header.pack(b"CB61ZL01", len(new_cb59),
                         zlib.crc32(new_cb59), hashlib.sha256(new_cb59).digest())
    packed += zlib.compress(new_cb59, 6)
    expected_len, expected_crc, expected_sha = header.unpack_from(packed)[1:]
    restored = zlib.decompress(packed[header.size:])
    if (restored != new_cb59 or len(restored) != expected_len or
            zlib.crc32(restored) != expected_crc or sha(restored) != expected_sha.hex()):
        raise AssertionError("Changed physically packed CB59 not exactly recoverable")
    return packed


def _pointer(weights):
    ptr = s82.GenericContextPointer()
    ptr.load_state_dict({k.removeprefix("generic_pointer."): v for k, v in
                         weights.items() if k.startswith("generic_pointer.")},
                        strict=True)
    return ptr.eval().requires_grad_(False)


def _original_cpu(model, sources, weights, tok, question, ptr):
    first, later = s78.controllers(weights)
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)
    head = model.lm_head.source().detach().cpu().contiguous()
    return s82.full_cpu_generate(model, sources, first, later,
                                 head, tok, question, ptr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("original_cb59", type=Path)
    p.add_argument("original_cb61", type=Path)
    p.add_argument("S82_controller", type=Path)
    p.add_argument("S82_paid_manifest", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--steps", type=int, default=64)
    a = p.parse_args()
    if not MIN_TRAIN_STEPS <= a.steps <= MAX_TRAIN_STEPS:
        raise ValueError("Bounded S87 feasibility TRAIN only")
    if not torch.backends.mps.is_available():
        raise RuntimeError("S87 requires REAL Apple Silicon MPS GPU")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw = a.original_cb59.read_bytes()
    source = a.original_cb61.read_bytes()
    old_side = a.S82_controller.read_bytes()
    old_manifest = json.loads(a.S82_paid_manifest.read_text())
    if (len(raw) != SOURCE_BYTES or sha(raw) != s65.S45_SHA or
            sha(source) != old_manifest["original_S61_physically_compressed_source_SHA256"] or
            sha(old_side) != old_manifest["S82_full_new_controller_SHA256"] or
            int(old_manifest["all_inclusive_physical_bytes"]) !=
            len(source)+len(old_side)+a.S82_paid_manifest.stat().st_size or
            float(old_manifest["all_inclusive_bpp"]) > MAX_BPP):
        raise AssertionError("Immutable S82 paid source/controller provenance mismatch")
    weights = load_file(str(a.S82_controller))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    s30.TOK = tok
    pairs = s30.make_training_pairs(tok)
    new_pairs, _ = s82.newly_authored_train(tok)
    # Only previously authorized TRAIN data. Do not load any inspected QA.
    old_rows = [pairs[i][side] for i in (0, 1) for side in ("left", "right")]
    new_rows = [new_pairs[i][side] for i in (0, 1) for side in ("left", "right")]
    cpu, cpu_src = create_small_paid_model(CB59(raw), build_config(MODEL_SPEC))
    cpu.eval().requires_grad_(False)
    ptr = _pointer(weights)
    before = {i: _original_cpu(cpu, cpu_src, weights, tok, r["question"], ptr)
              for i, r in enumerate(old_rows+new_rows)}
    integrated_cpu = IntegratedGreenTunnel(layers=TUNNEL_LAYERS,
                                            rank=TUNNEL_RANK).eval()
    integrated_cpu.install(cpu)
    unchanged = {i: _original_cpu(cpu, cpu_src, weights, tok, r["question"], ptr)
                 for i, r in enumerate(old_rows+new_rows)}
    if before != unchanged:
        raise AssertionError("Zero-initialized internal tunnel modified old S82 outputs")
    integrated_cpu.uninstall()
    del cpu, cpu_src

    student, sources = create_small_paid_model(CB59(raw), build_config(MODEL_SPEC))
    student.to("mps").eval().requires_grad_(False)
    inner = IntegratedGreenTunnel(layers=TUNNEL_LAYERS, rank=TUNNEL_RANK).to("mps")
    inner.install(student)
    paid_core = select_paid_core_codebooks(sources, TUNNEL_LAYERS)
    parameters = [src.centroids for src in paid_core.values()] + list(inner.parameters())
    # S87-J3: train through the *actual inference scoring path*, not only raw LM CE.
    # All three reference components are frozen S82 tensors. We differentiate
    # through their input hidden states but never update their parameters.
    mps_first, mps_later = s78.controllers(weights)
    mps_first.to("mps").eval().requires_grad_(False)
    mps_later.to("mps").eval().requires_grad_(False)
    mps_pointer = _pointer(weights).to("mps")
    # J3: fixed six-family *TRAIN-only* replay, preregistered before any new QA.
    # 64 steps cover all 256 S82 new cases and all 64 S30/S54 cases once;
    # each step includes historical, S69 and S73 safety rehearsal.
    # Prior inspected QA sets, including J2 fresh64, are fully embargoed.
    def flatten(rows):
        return [p[side] for p in rows for side in ("left", "right")]
    cohort = {
        "S30": flatten(pairs),
        "S54": flatten(s54.build_new_train(tok)),
        "history": flatten(build_pairs(tok, "TRAIN")[:4]),
        "S69": flatten(s69.build_new_train(tok)),
        "S73": flatten(s73.new_s73_train(tok)),
        "S82_new": flatten(new_pairs),
    }
    if [len(v) for v in cohort.values()] != [64, 64, 8, 32, 32, 256]:
        raise AssertionError("Authorized TRAIN six-family replay cohort changed")
    optimizer = torch.optim.Adam(parameters, lr=0.00006)
    steps = []
    for step in range(a.steps):
        # Nine real TRAIN examples per update; never reference independent QA.
        # At 64 steps, each S82 example is used exactly once, and 5 other
        # TRAIN families receive protected rehearsals in every update.
        batch = [
            cohort["S82_new"][(4 * step + j) % 256] for j in range(4)
        ] + [
            cohort["S30"][step % 64],
            cohort["S54"][step % 64],
            cohort["history"][step % 8],
            cohort["S69"][step % 32],
            cohort["S73"][step % 32],
        ]
        cases = [(r["question"], r["answer"]) for r in batch]
        report = aligned_joint_step(student, sources, inner, paid_core, tok,
                                    cases, optimizer, mps_first, mps_later,
                                    mps_pointer, step=step)
        steps.append(report.__dict__)
        print(json.dumps({"event":"S87_J3_MPS_ALIGNED_FULL_EOS_STEP",
                          "step":step+1, **report.__dict__}), flush=True)
    del mps_first, mps_later, mps_pointer
    inner.uninstall()
    inner.to("cpu")
    student.to("cpu")
    modified, mutation = replace_discrete(CB59(raw), sources, strict=False)
    if len(modified) != SOURCE_BYTES or CB59(modified).hash() != sha(modified):
        raise AssertionError("Repacked changed paid model failed integrity")
    packed = lossless_cb61(modified)
    packed_path = a.out/"S87_UNACCEPTED_NEW_SOURCE_CB61.s61"
    packed_path.write_bytes(packed)
    if packed_path.read_bytes() != packed:
        raise AssertionError("Changed physical source did not round-trip")

    # Diagnose post-training behavior against the actual serialized CB59
    # CPU source, not the transient GPU training weights. This is a small
    # TRAIN-only safety preflight; it is NOT full 456-prompt acceptance.
    cpu_joint, joint_sources = create_small_paid_model(
        CB59(modified), build_config(MODEL_SPEC))
    cpu_joint.eval().requires_grad_(False)
    joint_inner = IntegratedGreenTunnel(
        layers=TUNNEL_LAYERS, rank=TUNNEL_RANK).eval()
    joint_inner.load_state_dict(inner.state_dict(), strict=True)
    joint_inner.install(cpu_joint)
    cpu_first, cpu_later = s78.controllers(weights)
    cpu_first.eval().requires_grad_(False)
    cpu_later.eval().requires_grad_(False)
    cpu_head = cpu_joint.lm_head.source().detach().cpu().contiguous()
    after_native = {}
    protected_lost_full = []
    protected_lost_first = []
    newly_full = []
    for i, row in enumerate(old_rows + new_rows):
        generated = s82.full_cpu_generate(cpu_joint, joint_sources,
                                          cpu_first, cpu_later, cpu_head,
                                          tok, row["question"], ptr)
        gold = tok.encode(row["answer"], add_special_tokens=False) + [tok.eos_token_id]
        was = before[i]
        prior_full = was == gold
        now_full = generated == gold
        prior_first = bool(was) and was[0] == gold[0]
        now_first = bool(generated) and generated[0] == gold[0]
        if prior_full and not now_full:
            protected_lost_full.append(i)
        if prior_first and not now_first:
            protected_lost_first.append(i)
        if now_full and not prior_full:
            newly_full.append(i)
        after_native[str(i)] = {
            "TRAIN_question_sha256": sha(row["question"].encode()),
            "expected_ids": gold, "original_S82_ids": was,
            "joint_compressed_CPU_ids": generated,
            "original_full": prior_full, "new_full": now_full,
            "original_first": prior_first, "new_first": now_first,
        }
    joint_inner.uninstall()
    side = dict(weights)
    for name, tensor in inner.state_dict().items():
        side["internal_tunnel."+name] = tensor.detach().cpu().contiguous()
    prototype_side = a.out/"S87_UNACCEPTED_JOINT_PROTOTYPE.safetensors"
    save_file(side, str(prototype_side))
    if any(not torch.equal(weights[k], load_file(str(prototype_side))[k])
           for k in weights):
        raise AssertionError("Frozen S82 tensor modified")
    proposed_meta = {
        "stage":"S87_J3_UNACCEPTED_CONTROLLER_ALIGNED_JOINT_CORE_INTERNAL_TUNNEL",
        "training_objective":"balanced replay 6 TRAIN families via real frozen S82 first/later/pointer logits + full EOS",
        "training_provenance":"six previously authorized TRAIN families S30/S54/history/S69/S73/S82",
        "candidate_accepted":False,
        "source_sha256":sha(modified), "source_unchanged_bytes":len(modified),
        "source_CB61_physical_sha256":sha(packed),
        "old_S82_sidecar_sha256":sha(old_side),
        "prototype_sidecar_sha256":sha(prototype_side.read_bytes()),
        "internal_tunnel_rank":TUNNEL_RANK,
        "internal_layers":TUNNEL_LAYERS,
        "trainable_paid_core_operators":list(paid_core),
        "trainable_internal_tunnel_parameters":inner.unique_parameter_count(),
        "original_8_train_outputs_identical_at_zero_initialization":True,
        "only_previously_authorized_TRAIN":True,
        "all_previous_inspected_QA_permanently_embargoed":True,
        "never_promote_unchecked_candidate":True,
        "GPU_training_real":True,
        "actual_gradients":steps,
        "small_8_TRAIN_actual_serialized_CPU_outputs":after_native,
        "pilot_8_TRAIN_lost_full_correct":protected_lost_full,
        "pilot_8_TRAIN_lost_first_correct":protected_lost_first,
        "pilot_8_TRAIN_new_full_correct":newly_full,
        "pilot_8_TRAIN_guard_passed":not protected_lost_full and not protected_lost_first,
        "physical_cb61_source_file":packed_path.name,
        "native_ALL_456_TRAIN_old_skills_guard_completed":False,
        "full_CPU_vs_GPU_post_training_parity_completed":False,
        "full_CPU_speed_guard_completed":False,
        "independent_QA_completed":False,
        "physical_original_CB59_mutation":mutation,
        "original_S82_was_not_modified":True,
        "main_LIVE_Jessica_Render_Supabase_APK_untouched":True,
    }
    for _ in range(30):
        manifest = json.dumps(proposed_meta, indent=2, sort_keys=True).encode()
        total = len(packed)+prototype_side.stat().st_size+len(manifest)
        fields = {"physical_source_bytes":len(packed),
                  "full_controller_bytes":prototype_side.stat().st_size,
                  "manifest_bytes":len(manifest),
                  "all_inclusive_physical_bytes":total,
                  "all_inclusive_physical_bpp":f"{8*total/SOURCE_PARAMS:.12f}",
                  "within_0_7_bpp":8*total/SOURCE_PARAMS <= MAX_BPP}
        if all(proposed_meta.get(k) == v for k,v in fields.items()): break
        proposed_meta.update(fields)
    result_json = a.out/"S87_UNACCEPTED_JOINT_PHYSICAL_PROTOTYPE_REPORT.json"
    result_json.write_text(json.dumps(proposed_meta,indent=2,ensure_ascii=False))
    print(json.dumps({"event":"S87_UNACCEPTED_PROTOTYPE_END",
                      "joint_gradient_steps":len(steps),
                      "physical_bpp":proposed_meta["all_inclusive_physical_bpp"],
                      "within_bpp_limit":proposed_meta["within_0_7_bpp"],
                      "pilot_8_TRAIN_protected_lost_full":len(protected_lost_full),
                      "pilot_8_TRAIN_protected_lost_first":len(protected_lost_first),
                      "pilot_8_TRAIN_new_full":len(newly_full),
                      "candidate_accepted":False}),flush=True)


if __name__ == "__main__":
    main()
