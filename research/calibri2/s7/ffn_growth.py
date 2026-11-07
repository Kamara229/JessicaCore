"""CALIBRI2 S7: learnable width growth of the 12-layer paid 29M model.

Expand ONLY the SwiGLU intermediate channels 896 -> 1536 while retaining
hidden_size=320, all 12 layers, original tokenizer/head/attention/RoPE.
Each existing 5-bit index, codebook and scale is transferred bit-for-bit.
Extra gate/up channels start at physically paid minimum FP16 row scale and
extra down columns start with a codeword from the existing paid dictionary.
No external FP32, FP32 teacher, external train artifact, or unpaid sidecar.

Use independently verified S5 29M source for this engineering preflight.
Once S6 is sealed, reapply same migration to its final trained archive.
The TRAIN-only response comparison checks representation, not accuracy parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59, HEADER, encode
from calibri2.s0.bootstrap import build_config, native_forward_test
from calibri2.s0.train import create_small_paid_model
from calibri2.s3.diverse_task_replay import make_new_pairs
from v186.pairs import build_pairs
from v200.rollout import probe_greedy_train
from v174.run import save

OLD_SOURCE_SHA = "7fbb42a6bd364527b6766fa1e81b0e5301767ea5baab49e044fd9f2012de8718"
OLD_SOURCE_BYTES = 2534757
OLD_PARAMS = 29007680
NEW_PARAMS = 36380480
OLD_FFN = 896
NEW_FFN = 1536
HIDDEN = 320
LAYERS = 12
EXPECTED_MATRICES = 1 + 7 * LAYERS
SEED = 2067
BPP_MAX = 0.700
# Original codebooks/row scales never change. This affects only NEW rows.
SMALLEST_NORMAL_FP16 = np.float16(2.0 ** -14)
SPEC_SMALL = ("small_320x12", HIDDEN, OLD_FFN, LAYERS, 5, 1)
SPEC_BIG = ("large_ffn_320x12", HIDDEN, NEW_FFN, LAYERS, 5, 1)


def body_start(blob):
    magic, version, meta_len, body_len = HEADER.unpack_from(blob)
    if magic != b"CB59" or version != 1:
        raise AssertionError("Invalid CB59 header")
    return HEADER.size + meta_len


def extract(blob, data_start, desc, part):
    at, size = desc["parts"][part]
    return blob[data_start+at:data_start+at+size]


def paid_growth(old_blob, random_big_blob):
    old = CB59(old_blob)
    new = CB59(random_big_blob)
    small_data = body_start(old_blob)
    large_data = body_start(random_big_blob)
    old_desc = old.meta["tensor_descriptors"]
    big_desc = new.meta["tensor_descriptors"]
    if set(old_desc) != set(big_desc):
        raise AssertionError("Growth changed the required original key names")
    if old.meta["source_alias_keys"] != new.meta["source_alias_keys"]:
        raise AssertionError("Tied embedding/head changed")
    out = bytearray(random_big_blob)
    checked = 0
    unchanged = 0
    expanded = 0
    copies = []
    for name in sorted(old_desc):
        source, target = old_desc[name], big_desc[name]
        if source["dtype"] != target["dtype"]:
            raise AssertionError("Encoder changed operator dtype: " + name)
        if source["shape"] == target["shape"]:
            for segment, (off, count) in source["parts"].items():
                target_off, target_count = target["parts"][segment]
                if target_count != count:
                    raise AssertionError("Unchanged operator changed physical field size")
                out[large_data+target_off:large_data+target_off+count] = \
                    old_blob[small_data+off:small_data+off+count]
                checked += 1
            unchanged += 1
            continue
        allowed = (
            name.startswith("model.layers.") and ".mlp." in name and
            name.endswith("_proj.weight") and
            (".gate_proj." in name or ".up_proj." in name or ".down_proj." in name)
        )
        if not allowed or source["dtype"] != "CB59_K32V8":
            raise AssertionError("Unknown new shape: " + name)
        if name.endswith("down_proj.weight"):
            if source["shape"] != [HIDDEN, OLD_FFN] or target["shape"] != [HIDDEN, NEW_FFN]:
                raise AssertionError("Mismatched FFN down width")
            # 112 original 8D groups/row = 70 original 5-bit bytes/row.
            # 192 final groups/row = 120 bytes/row. Both byte-aligned.
            old_nbytes = (OLD_FFN // 8) * 5 // 8
            new_nbytes = (NEW_FFN // 8) * 5 // 8
            assert (old_nbytes, new_nbytes) == (70, 120)
            original = extract(old_blob, small_data, source, "packed")
            extended = bytearray(new_nbytes * HIDDEN)
            for row in range(HIDDEN):
                extended[row*new_nbytes:row*new_nbytes+old_nbytes] = \
                    original[row*old_nbytes:(row+1)*old_nbytes]
                # New 5-bit vectors are index=0; old codebook is preserved.
            pieces = {
                "packed": bytes(extended),
                "row_scales": extract(old_blob, small_data, source, "row_scales"),
                "centroids": extract(old_blob, small_data, source, "centroids"),
            }
        else:
            if source["shape"] != [OLD_FFN, HIDDEN] or target["shape"] != [NEW_FFN, HIDDEN]:
                raise AssertionError("Mismatched FFN gate/up width")
            # No change to old rows/codes/centroids. New 640 output neurons
            # get the existing codeword 0 and normal, non-zero minimum FP16
            # representable scales. New neurons are ~zero, not hard-frozen.
            previous_codes = extract(old_blob, small_data, source, "packed")
            new_count = ((NEW_FFN-OLD_FFN) * HIDDEN // 8) * 5 // 8
            previous_scales = extract(old_blob, small_data, source, "row_scales")
            added_scales = np.full(NEW_FFN-OLD_FFN, SMALLEST_NORMAL_FP16, dtype="<f2").tobytes()
            pieces = {
                "packed": previous_codes + bytes(new_count),
                "row_scales": previous_scales + added_scales,
                "centroids": extract(old_blob, small_data, source, "centroids"),
            }
        for part, payload in pieces.items():
            target_off, target_len = target["parts"][part]
            if len(payload) != target_len:
                raise AssertionError(f"Wrong grown CB59 bytes for {name}.{part}")
            out[large_data+target_off:large_data+target_off+target_len] = payload
            checked += 1
        expanded += 1
        copies.append(name)
    if expanded != 3 * LAYERS or unchanged + expanded != len(old_desc):
        raise AssertionError("Not all paid weight matrices were migrated")
    out[-36:-32] = struct.pack("<I", zlib.crc32(out[:-36]))
    out[-32:] = hashlib.sha256(out[:-32]).digest()
    sealed = bytes(out)
    grown = CB59(sealed)
    # Check every unchanged physical part and old logical vector code.
    for name, desc in old_desc.items():
        other = big_desc[name]
        if desc["shape"] == other["shape"]:
            for part in desc["parts"]:
                if extract(sealed, large_data, other, part) != \
                   extract(old_blob, small_data, desc, part):
                    raise AssertionError("Old full operator corrupted: " + name)
        elif name.endswith("down_proj.weight"):
            old_g = np.frombuffer(extract(old_blob,small_data,desc,"packed"),dtype=np.uint8).reshape(HIDDEN,70)
            big_g = np.frombuffer(extract(sealed,large_data,other,"packed"),dtype=np.uint8).reshape(HIDDEN,120)
            if not np.array_equal(old_g, big_g[:,:70]):
                raise AssertionError("Old FFN down 5-bit codes were not preserved")
        else:
            original_codes = extract(old_blob,small_data,desc,"packed")
            extended_codes = extract(sealed,large_data,other,"packed")
            if extended_codes[:len(original_codes)] != original_codes:
                raise AssertionError("Old FFN up/gate 5-bit codes were not preserved")
        for part in ("centroids",):
            if part in desc["parts"]:
                if extract(sealed,large_data,other,part) != extract(old_blob,small_data,desc,part):
                    raise AssertionError("Old 32x8 paid dictionary was altered")
    return sealed, {"old_unchanged_operators": unchanged,
                    "grown_mlp_operators": expanded,
                    "physical_segments_transferred": checked,
                    "old_matrix_bits_and_dictionary_preserved": True,
                    "added_gate_up_scales_fp16": float(SMALLEST_NORMAL_FP16)}


def training_pairs(tok):
    old = build_pairs(tok, "TRAIN")
    further = make_new_pairs(tok)
    by_domain = {}
    for p in further:
        by_domain.setdefault(p["domain"], []).append(p)
    # 48 old + 48 added exact training examples, including multi-token.
    chosen = old + (
        by_domain["s3_folder_training"][:8] +
        by_domain["s3_compare_training"][:8] +
        by_domain["s3_two_word_training"][:4] +
        by_domain["s3_sentence_training"][:4]
    )
    if len(chosen) != 48 or len(set(p["pair_id"] for p in chosen)) != 48:
        raise AssertionError("S7 TRAIN regression pair selection wrong")
    return chosen


def real_answers(model, sources, pairs):
    tested = probe_greedy_train(model, sources, pairs, max_new_tokens=14)
    data = []
    for p in pairs:
        for side in ("left", "right"):
            r = tested[p["pair_id"]]["sides"][side]
            data.append({
                "pair": p["pair_id"], "domain": p["domain"], "side": side,
                "gold": p[side]["answer"],
                "actual_generated_ids": r["generated_tokens"],
                "strict_full_and_EOS": bool(r["exact_with_EOS"]),
            })
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--source-sha", default=OLD_SOURCE_SHA)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw = args.old.read_bytes()
    if len(raw) != OLD_SOURCE_BYTES or hashlib.sha256(raw).hexdigest() != args.source_sha:
        raise AssertionError("S7 only accepts an explicitly sealed and paid 320x12 source")
    source = CB59(raw)
    if source.meta["original_unique_parameter_count"] != OLD_PARAMS:
        raise AssertionError("Old model parameter count drift")
    cfg = build_config(SPEC_BIG)
    init = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32).cpu().eval()
    n = sum(int(x.numel()) for x in init.parameters())
    if n != NEW_PARAMS:
        raise AssertionError(f"Incorrect new unique 36M geometry: {n}")
    fresh_blob, _, _ = encode(init.state_dict())
    trained_blob, proof = paid_growth(raw, fresh_blob)
    cb = CB59(trained_blob)
    if cb.meta["original_unique_parameter_count"] != NEW_PARAMS:
        raise AssertionError("Packed grown model unique parameter count drift")
    bpp = 8 * len(trained_blob) / NEW_PARAMS
    if bpp > BPP_MAX:
        raise AssertionError("Full grown CB59 archive exceeds physical 0.700 bpp")
    if sum(v["dtype"] == "CB59_K32V8" for v in cb.meta["tensor_descriptors"].values()) != EXPECTED_MATRICES:
        raise AssertionError("Lost one of 85 physically paid operators")
    path = args.out/"CALIBRI2_S7_320x12_FFN1536_PHYSICAL_GROWN.cb59"
    path.write_bytes(trained_blob)
    sha = hashlib.sha256(trained_blob).hexdigest()

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    train = training_pairs(tokenizer)
    a, src_a = create_small_paid_model(source, build_config(SPEC_SMALL))
    b, src_b = create_small_paid_model(cb, cfg)
    before = real_answers(a, src_a, train)
    after = real_answers(b, src_b, train)
    compared = len(before)
    identical = sum(x["actual_generated_ids"] == y["actual_generated_ids"] for x,y in zip(before,after))
    regressed = sum(x["strict_full_and_EOS"] and not y["strict_full_and_EOS"] for x,y in zip(before,after))
    gained = sum(not x["strict_full_and_EOS"] and y["strict_full_and_EOS"] for x,y in zip(before,after))
    native = native_forward_test(cb, cfg, args.out)
    result = {
        "stage": "S7_FFN_CHANNEL_GROWTH_29M_TO_36M",
        "source_SHA256": args.source_sha, "grown_SHA256": sha,
        "source_architecture": SPEC_SMALL, "grown_architecture": SPEC_BIG,
        "old_unique_parameters": OLD_PARAMS, "grown_unique_parameters": NEW_PARAMS,
        "full_physical_file_bytes": len(trained_blob), "full_physical_bpp": bpp,
        "paid_dictionary_count": EXPECTED_MATRICES,
        "preserved_5bit_and_centroids_proof": proof,
        "TRAIN_only_96_answer_regression": {
            "compared": compared, "identical_full_sequences": identical,
            "new_errors_from_formerly_correct": regressed,
            "new_correct_from_formerly_wrong": gained,
            "source_correct": sum(x["strict_full_and_EOS"] for x in before),
            "grown_correct": sum(x["strict_full_and_EOS"] for x in after),
            "before_answers": before, "after_answers": after,
            "strict_preservation": (identical == compared and regressed == 0),
        },
        "native_C_full_model_check": native,
        "no_original_FP32_weights_teacher_or_QA": True,
        "no_LIVE_changes": True,
        "no_automatic_promotion_to_bigger_model": True,
        "no_native_speed_ratio_claim": True,
    }
    save(args.out/"ACTUAL_CALIBRI2_S7_FFN_GROWTH.json", result)
    print(json.dumps({
        "stage": result["stage"], "source_sha": args.source_sha, "grown_sha": sha,
        "n_old": OLD_PARAMS, "n_grown": NEW_PARAMS,
        "physical_bpp": bpp, "total_bytes": len(trained_blob),
        "paid_operators": EXPECTED_MATRICES,
        "TRAIN_answers_preserved": identical, "TRAIN_answers_total": compared,
        "new_errors": regressed, "TRAIN_correct_before": result["TRAIN_only_96_answer_regression"]["source_correct"],
        "TRAIN_correct_after": result["TRAIN_only_96_answer_regression"]["grown_correct"],
        "native_C_full_forward": native["real_native_C_complete_transformer_forward"],
    }), flush=True)
    if not result["TRAIN_only_96_answer_regression"]["strict_preservation"]:
        raise AssertionError("S7 FAIL: growth did not preserve 96 old TRAIN responses")

if __name__ == "__main__":
    main()
