"""S87 read-only, complete 456 TRAIN identities on truly serialized CB59.

Negative results are reported; this guard cannot promote S87. Greedy tokens,
including EOS, are compared to immutable S82 for every previously correct
TRAIN answer. CPU here reconstructs CB59 weights with PyTorch, NOT native C.
All previously inspected QA remains embargoed.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import statistics
import struct
import time
import zlib
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS, SOURCE_BYTES, MODEL_SPEC
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s54 import diverse_train_preserve_S52_full_sequences as s54
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s69 import train_new_contexts_without_old_skills_loss as s69
from calibri2.s73 import first_token_neural_recovery_protected_train as s73
from calibri2.s78 import mps_36m_feature_cached_late_rank16_restoration as s78
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from v186.pairs import build_pairs


def sha(b):
    return hashlib.sha256(b).hexdigest()


def open_envelope(packed):
    header = struct.Struct("<8sQI32s")
    if len(packed) < header.size + 1:
        raise AssertionError("CB61 truncated")
    magic, n, crc, digest = header.unpack_from(packed)
    if magic != b"CB61ZL01" or n != SOURCE_BYTES:
        raise AssertionError("Wrong compressed CB61 format/size")
    stream = zlib.decompressobj()
    raw = stream.decompress(packed[header.size:], n+1)
    if not stream.eof or stream.unused_data or stream.unconsumed_tail:
        raise AssertionError("CB61 truncated, extra data, or overflow")
    if len(raw) != n or zlib.crc32(raw) != crc or sha(raw) != digest.hex():
        raise AssertionError("Compressed core source checksum mismatch")
    CB59(raw)
    return raw


def build_cpu(raw, full_side, use_internal):
    model, sources = create_small_paid_model(
        CB59(raw), build_config(MODEL_SPEC))
    model.eval().requires_grad_(False)
    first, later = s78.controllers(full_side)
    first.eval().requires_grad_(False)
    later.eval().requires_grad_(False)
    pointer = s82.GenericContextPointer()
    pointer.load_state_dict({
        k.removeprefix("generic_pointer."): v for k,v in full_side.items()
        if k.startswith("generic_pointer.")
    }, strict=True)
    pointer.eval().requires_grad_(False)
    internal = None
    if use_internal:
        internal = IntegratedGreenTunnel(layers=(3,9), rank=8)
        internal.load_state_dict({
            k.removeprefix("internal_tunnel."):v for k,v in full_side.items()
            if k.startswith("internal_tunnel.")
        }, strict=True)
        internal.eval().requires_grad_(False)
        internal.install(model)
    head = model.lm_head.source().detach().cpu().contiguous()
    return model, sources, first, later, pointer, head, internal


def families(tok):
    new, _ = s82.newly_authored_train(tok)
    groups = {
        "S30": s30.make_training_pairs(tok),
        "S54": s54.build_new_train(tok),
        "history": build_pairs(tok, "TRAIN")[:4],
        "S69": s69.build_new_train(tok),
        "S73": s73.new_s73_train(tok),
        "S82_new": new,
    }
    if [2*len(v) for v in groups.values()] != [64,64,8,32,32,256]:
        raise AssertionError("Historical authorized 456 TRAIN cohort changed")
    return groups


def main():
    p=argparse.ArgumentParser()
    p.add_argument("original_cb59",type=Path)
    p.add_argument("original_S82_controller",type=Path)
    p.add_argument("new_S87_cb61",type=Path)
    p.add_argument("new_S87_controller",type=Path)
    p.add_argument("new_S87_manifest",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    original=a.original_cb59.read_bytes()
    original_side_bytes=a.original_S82_controller.read_bytes()
    new_cb61=a.new_S87_cb61.read_bytes()
    new_side_bytes=a.new_S87_controller.read_bytes()
    raw_new=open_envelope(new_cb61)
    metadata=json.loads(a.new_S87_manifest.read_text())
    if (len(original) != SOURCE_BYTES or sha(original)!=s65.S45_SHA or
        sha(original_side_bytes)!=metadata["old_S82_sidecar_sha256"] or
        sha(new_cb61)!=metadata["source_CB61_physical_sha256"] or
        sha(new_side_bytes)!=metadata["prototype_sidecar_sha256"] or
        sha(raw_new)!=metadata["source_sha256"]):
        raise AssertionError("Immutable source/sidecar SHA provenance mismatch")
    all_bytes=len(new_cb61)+len(new_side_bytes)+a.new_S87_manifest.stat().st_size
    if (all_bytes!=metadata["all_inclusive_physical_bytes"] or
        all_bytes*8/SOURCE_PARAMS>.7):
        raise AssertionError("All-inclusive actual physical 0.7 bpp audit FAIL")

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    original_side=load_file(str(a.original_S82_controller))
    joint_side=load_file(str(a.new_S87_controller))
    if any(k not in joint_side or not torch.equal(v,joint_side[k])
           for k,v in original_side.items()):
        raise AssertionError("Original S82 controller tensor was changed")
    old=build_cpu(original,original_side,False)
    new=build_cpu(raw_new,joint_side,True)
    grouped=families(tok)
    lost_full=[];lost_first=[];gained_full=[];rows=[]
    by_family={}
    timings={"S82":[],"S87":[]}
    for family, pairs in grouped.items():
        stats={"count":0,"old_full":0,"new_full":0,
               "old_first":0,"new_first":0,"lost_full":0,
               "lost_first":0,"gained_full":0}
        for pair in pairs:
            for side in ("left","right"):
                item=pair[side]
                expected=tok.encode(item["answer"],add_special_tokens=False)+[
                    tok.eos_token_id]
                values=[]
                for label, item_state in (("S82",old),("S87",new)):
                    model,sources,first,later,pointer,head,_=item_state
                    t=time.perf_counter()
                    generated=s82.full_cpu_generate(
                        model,sources,first,later,head,tok,item["question"],pointer)
                    if len(timings[label])<48:
                        timings[label].append(time.perf_counter()-t)
                    values.append(generated)
                before,after=values
                old_full=before==expected
                new_full=after==expected
                old_first=bool(before) and before[0]==expected[0]
                new_first=bool(after) and after[0]==expected[0]
                key=f'{family}/{pair["pair_id"]}/{side}'
                row={
                    "id":key,
                    "question_sha256":sha(item["question"].encode()),
                    "target_ids":expected,"S82_ids":before,"S87_ids":after,
                    "old_full":old_full,"new_full":new_full,
                    "old_first":old_first,"new_first":new_first}
                rows.append(row)
                stats["count"]+=1
                for name,value in (("old_full",old_full),("new_full",new_full),
                                   ("old_first",old_first),("new_first",new_first)):
                    stats[name]+=int(value)
                if old_full and not new_full:
                    lost_full.append(key);stats["lost_full"]+=1
                if old_first and not new_first:
                    lost_first.append(key);stats["lost_first"]+=1
                if new_full and not old_full:
                    gained_full.append(key);stats["gained_full"]+=1
        by_family[family]=stats
        print(json.dumps({"event":"S87_456_TRAIN_FAMILY", "family":family,
                          **stats}),flush=True)
    if len(rows)!=456 or len({r["id"] for r in rows})!=456:
        raise AssertionError("Not exactly 456 unique TRAIN identities")
    old_med=statistics.median(timings["S82"])
    new_med=statistics.median(timings["S87"])
    speed_ratio=old_med/new_med
    results={
        "stage":"S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT",
        "source_old_sha256":sha(original),
        "source_new_cb59_sha256":sha(raw_new),
        "physical_CB61_bytes":len(new_cb61),
        "full_sidecar_bytes":len(new_side_bytes),
        "manifest_bytes":a.new_S87_manifest.stat().st_size,
        "all_inclusive_physical_bytes":all_bytes,
        "all_inclusive_bpp":8*all_bytes/SOURCE_PARAMS,
        "all_456_per_identity_results":rows,"per_family":by_family,
        "lost_full_ids":lost_full,"lost_first_ids":lost_first,
        "gained_full_ids":gained_full,
        "full_safety_guard":not lost_full and not lost_first,
        "cpu_speed_ratio_48_prompt_median_S82_over_S87":speed_ratio,
        "cpu_speed_90pct_guard":speed_ratio>=.9,
        "cpu_backend":"PyTorch from actual CB59 compact source; NOT direct C",
        "GPU_CPU_exact_parity_checked":False,
        "never_promote_without_native_CPU_C_and_MPS_parity":True,
        "candidate_accepted":False,
        "all_inspected_QA_embargoed":True,
    }
    (a.out/"S87_ALL_456_TRAIN_GUARD_REPORT.json").write_text(
        json.dumps(results,indent=2))
    print(json.dumps({"event":"S87_ALL_456_TRAIN_GUARD_END",
                      "lost_full":len(lost_full),
                      "lost_first":len(lost_first),
                      "new_full":len(gained_full),
                      "CPU_ratio_S82_over_S87":speed_ratio,
                      "physical_bpp":8*all_bytes/SOURCE_PARAMS,
                      "safety_pass":results["full_safety_guard"],
                      "speed_pass":results["cpu_speed_90pct_guard"],
                      "candidate_accepted":False}),flush=True)
    if not results["full_safety_guard"] or not results["cpu_speed_90pct_guard"]:
        raise SystemExit("REJECTED: S87 456-TRAIN or CPU speed safety guard")


if __name__=="__main__":
    main()
