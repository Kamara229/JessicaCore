"""CALIBRI2 S5: function-preserving physical 320x6 -> 320x12 depth growth.
Do not start 135M from scratch again: transplant ALL 22M paid packed operator
payloads byte-for-byte, create six near-identity residual decoder layers and
verify real answer regression BEFORE any further TRAIN updates.
No pretrained original FP32 weights or teacher, no QA/VALID/TEST for selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
import struct
import zlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59, encode, HEADER
from calibri2.s0.bootstrap import CANDIDATES, build_config, native_forward_test
from calibri2.s0.train import create_small_paid_model
from calibri2.s4.last_mile import PARAMS, START_BYTES, START_SHA
from v186.pairs import build_pairs
from v200.rollout import probe_greedy_train
from v174.run import save

SMALL_SPEC = next(s for s in CANDIDATES if s[0] == "small_320x6")
GROWN_SPEC = ("grow_320x12", 320, 896, 12, 5, 1)
SEED = 2050
OLD_LAYERS = 6
NEW_LAYERS = 12
CAP = .700

def data_body(blob):
    _, _, mlen, _ = HEADER.unpack_from(blob)
    return HEADER.size + mlen

def transplant_old_paid_payloads(small_blob, generated_big_blob):
    small, large = CB59(small_blob), CB59(generated_big_blob)
    sb, lb = data_body(small_blob), data_body(generated_big_blob)
    a = small.meta["tensor_descriptors"]
    b = large.meta["tensor_descriptors"]
    target = bytearray(generated_big_blob)
    transfer = []
    if small.meta["source_alias_keys"] != large.meta["source_alias_keys"]:
        raise AssertionError("Tied aliases changed during depth growth")
    for key, descriptor in a.items():
        if key not in b:
            raise AssertionError("Original operator vanished: " + key)
        current = b[key]
        if descriptor["shape"] != current["shape"] or descriptor["dtype"] != current["dtype"]:
            raise AssertionError("Old packed 22M geometry changed for " + key)
        for part, (start, length) in descriptor["parts"].items():
            newstart, newlength = current["parts"][part]
            if newlength != length:
                raise AssertionError("Cannot pay-copy old 5-bit segment " + key)
            target[lb+newstart:lb+newstart+newlength] = \
                small_blob[sb+start:sb+start+length]
        transfer.append(key)
    target[-36:-32] = struct.pack("<I", zlib.crc32(target[:-36]))
    target[-32:] = hashlib.sha256(target[:-32]).digest()
    final = bytes(target)
    archive = CB59(final)
    if len(transfer) != len(a):
        raise AssertionError("Not all old compressed physical operators transferred")
    for key, desc in a.items():
        for part, (start, length) in desc["parts"].items():
            next_start, next_len = archive.meta["tensor_descriptors"][key]["parts"][part]
            assert next_len == length
            if (final[lb+next_start:lb+next_start+next_len] !=
                small_blob[sb+start:sb+start+length]):
                raise AssertionError("Physical source bitwise preservation failed " + key)
    return final, {"byte_identical_source_descriptors": len(transfer),
                   "same_tied_aliases": True, "all_physical_source_parts_unchanged": True}

def freeze_old_train_responses(model, sources, pairs):
    r = probe_greedy_train(model, sources, pairs, max_new_tokens=10)
    rows = []
    for p in pairs:
        for side in ("left", "right"):
            sample = r[p["pair_id"]]["sides"][side]
            rows.append({"pair": p["pair_id"], "domain": p["domain"], "side": side,
                         "gold": p[side]["answer"], "ids": sample["generated_tokens"],
                         "strict_full_EOS": sample["exact_with_EOS"]})
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--expect-sha", default=START_SHA)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(2)
    small_blob = args.source.read_bytes()
    if len(small_blob) != START_BYTES or hashlib.sha256(small_blob).hexdigest() != args.expect_sha:
        raise AssertionError("Growth requires exact sealed small model")
    small = CB59(small_blob)
    if small.meta["original_unique_parameter_count"] != PARAMS:
        raise AssertionError("Unexpected source size, never grow wrong model")
    cfg = build_config(GROWN_SPEC)
    fresh = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32).cpu().eval()
    grown_n = sum(x.numel() for x in fresh.parameters())
    if not PARAMS < grown_n < 134515008:
        raise AssertionError("Depth growth is not actually a 22M->~30M intermediary")
    # Residual x+attention(x), x+MLP(x): tiny o_proj and down_proj make
    # each new block approximately an identity WITHOUT any dense sidecar.
    with torch.no_grad():
        for layer in fresh.model.layers[OLD_LAYERS:NEW_LAYERS]:
            layer.self_attn.o_proj.weight.mul_(1.e-6)
            layer.mlp.down_proj.weight.mul_(1.e-6)
    big, meta, diag = encode(fresh.state_dict())
    if meta["original_unique_parameter_count"] != grown_n:
        raise AssertionError("Encoded source lost tied embedding/head")
    # New decoder matrices are random (not from pretrained FP32); the existing
    # 22M source stays physically intact. No decoding/re-quantization of it.
    migrated, transplant = transplant_old_paid_payloads(small_blob, big)
    grown = CB59(migrated)
    if grown.meta["original_unique_parameter_count"] != grown_n:
        raise AssertionError("New 12-layer physical original count disagrees")
    bpp = 8*len(migrated)/grown_n
    if bpp > CAP:
        raise AssertionError("Depth grown checkpoint exceeds full 0.7 bpp")
    if len(grown.meta["tensor_descriptors"]) <= len(small.meta["tensor_descriptors"]):
        raise AssertionError("Did not physically pay for new decoder layers")
    path = args.out/"CALIBRI2_S5_320x12_PHYSICAL_GROWN.cb59"
    path.write_bytes(migrated)
    sha = hashlib.sha256(migrated).hexdigest()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    before_model, before_src = create_small_paid_model(small, build_config(SMALL_SPEC))
    after_model, after_src = create_small_paid_model(grown, cfg)
    # Identical TRAIN-only prompts; NO QA checkpoint/model/architecture selection.
    pairs = build_pairs(tok, "TRAIN")
    selected = pairs[:4] + pairs[6:8] + pairs[18:20]
    before = freeze_old_train_responses(before_model, before_src, selected)
    after = freeze_old_train_responses(after_model, after_src, selected)
    unchanged = sum(x["ids"] == y["ids"] for x, y in zip(before, after))
    lost = sum(x["strict_full_EOS"] and not y["strict_full_EOS"]
               for x, y in zip(before, after))
    new = sum(not x["strict_full_EOS"] and y["strict_full_EOS"]
              for x, y in zip(before, after))
    native = native_forward_test(grown, cfg, args.out)
    report = {
        "stage": "S5_FUNCTION_PRESERVING_320x6_TO_320x12",
        "source_sha": args.expect_sha, "grown_sha": sha,
        "old_unique_params": PARAMS, "new_unique_params": grown_n,
        "new_layer_count": NEW_LAYERS, "old_layer_count": OLD_LAYERS,
        "new_paid_operator_count": sum(d["dtype"]=="CB59_K32V8"
             for d in grown.meta["tensor_descriptors"].values()),
        "old_paid_operator_count": sum(d["dtype"]=="CB59_K32V8"
             for d in small.meta["tensor_descriptors"].values()),
        "physical_bytes": len(migrated), "physical_bpp": bpp,
        "original_physical_segments": transplant,
        "additional_layers_near_identity": True,
        "new_layer_weights_from_random_seed_not_original_FP32": SEED,
        "native_C_forward": native,
        "old_TRAIN_answers": before,
        "after_growth_BEFORE_training_TRAIN_answers": after,
        "identical_actual_full_greedy_sequences": unchanged,
        "compared_actual_responses": len(before),
        "old_correct_answers_lost": lost,
        "new_correct_answers": new,
        "TRAIN_answer_preservation_strict_pass": unchanged==len(before) and lost==0,
        "no_QA_VALID_TEST_or_FP32_teacher_used": True,
        "not_an_Fp32_or_native_CPU_speed_parity_claim": True}
    save(args.out/"ACTUAL_CALIBRI2_S5_DEPTH_GROWTH_REPORT.json", report)
    print(json.dumps({"stage": report["stage"], "n_source_params": PARAMS,
                      "n_grown_params": grown_n, "bpp": bpp, "SHA": sha,
                      "paid_operators": report["new_paid_operator_count"],
                      "preserved_greedy": unchanged, "total": len(before),
                      "old_correct_lost": lost,
                      "strict_train_preservation": report["TRAIN_answer_preservation_strict_pass"]}), flush=True)
    if not report["TRAIN_answer_preservation_strict_pass"]:
        raise AssertionError("Growth FAILED preservation: do not promote or train larger model")

if __name__ == "__main__":
    main()
