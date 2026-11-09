"""CALIBRI S94: lossless alternate *weights*, not an answer lookup. RESEARCH ONLY.

A single S93 36M CB59 core plus its fixed S82 controller can possibly
retain S87/S91/S93 trained skills by storing only the two changed core
dictionaries and Green Tunnel blocks for the two alternative profiles.
This stage tests that physical affordability and representative exact CPU
parity, NOT a learned router, not an accepted model, not new QA.

No answer tokens, response IDs, question hashes or routing rules enter the
serialized bank. Independent frozen 456-TRAIN reports establish the maximum
achievable union; a future prompt-only router must earn it without lookup.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from calibri2.s8.broad_replay import SOURCE_PARAMS
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s88.zero_gate_fastpath import install_verified_fastpath
from v176.operators import clear

MAX_BPP=0.7
CHANGED=("model.layers.3.mlp.down_proj.weight",
         "model.layers.9.mlp.down_proj.weight")
PROFILES=("S87","S91","S93")
BANK_MAGIC=b"S94BANK1"


def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def freeze_report(path:Path):
    data=json.loads(path.read_text())
    if data.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT":
        raise AssertionError("Frozen independent CPU full/EOS report missing")
    rows=data.get("all_456_per_identity_results",[])
    if len(rows)!=456 or len({r["id"] for r in rows})!=456:
        raise AssertionError("Not the complete frozen 456 TRAIN cohort")
    return {r["id"]:r for r in rows}


def check_sources(sources):
    """Only two existing FP16 core codebooks may vary; 5bit codes stay fixed."""
    base=sources["S93"]
    a=base.meta["tensor_descriptors"]
    if CHANGED[0] not in a or CHANGED[1] not in a:
        raise AssertionError("Missing expected two paid MLP codebooks")
    differences={}
    for name,other in sources.items():
        if base.meta!=other.meta:
            raise AssertionError("Different pretrained source or source aliases: "+name)
        moved=[]
        for key,desc in a.items():
            if desc["dtype"]=="f4":
                if not np.array_equal(base.raw[key],other.raw[key]):
                    raise AssertionError("Unpaid raw core vector changed: "+key)
                continue
            b=base.handles[key]
            q=other.handles[key]
            if (not np.array_equal(b.packed,q.packed) or
                not np.array_equal(b.row_scales,q.row_scales)):
                raise AssertionError("5bit codes or row scales changed: "+key)
            if not np.array_equal(b.centroids,q.centroids):
                if key not in CHANGED:
                    raise AssertionError("Unaccounted core codebook changed: "+key)
                moved.append(key)
        differences[name]=moved
    if any(set(differences[p])!=set(CHANGED) for p in ("S87","S91")):
        raise AssertionError("Cannot represent source alternatives with two dictionaries")
    return differences


def check_controllers(sides):
    base=sides["S93"]
    blocks={k for k in base if k.startswith("internal_tunnel.")}
    if not blocks:
        raise AssertionError("Internal Green Tunnel not physically present")
    for name,other in sides.items():
        if set(other)!=set(base):
            raise AssertionError("Full S82 controller/Green Tunnel ABI changed: "+name)
        for key in base:
            if not key.startswith("internal_tunnel.") and not torch.equal(base[key],other[key]):
                raise AssertionError("Frozen S82 controller changed: "+key)
    return sorted(blocks)


def lossless_bank(archives,sides,block_keys,out):
    """Store exact FP16 core dictionaries and exact original tunnel tensors.

    zlib compression is applied only to physically serialized *weights*.
    Neither output labels nor TRAIN question identifiers are stored.
    """
    index=[]
    raw=bytearray()
    for profile in ("S87","S91"):
        for name in CHANGED:
            arr=np.asarray(archives[profile].handles[name].centroids,
                           dtype="<f2")
            if tuple(arr.shape)!=(32,8):
                raise AssertionError("Physical CB59 codebook geometry changed")
            blob=arr.tobytes()
            index.append({"profile":profile,"kind":"centroid","name":name,
                "dtype":"float16","shape":[32,8],
                "offset":len(raw),"size":len(blob),"sha256":sha(blob)})
            raw.extend(blob)
        for key in block_keys:
            t=sides[profile][key].detach().cpu().contiguous()
            if t.dtype not in (torch.float32,torch.float16):
                raise AssertionError("Unexpected Green Tunnel physical dtype")
            blob=t.numpy().tobytes()
            index.append({"profile":profile,"kind":"internal_tunnel",
                "name":key,"dtype":str(t.dtype).removeprefix("torch."),
                "shape":list(t.shape),"offset":len(raw),
                "size":len(blob),"sha256":sha(blob)})
            raw.extend(blob)
    packed=zlib.compress(bytes(raw),9)
    header={"stage":"S94_LOSSLESS_WEIGHT_ONLY_THREE_PROFILE_BANK",
            "base_profile":"S93","alternative_profiles":["S87","S91"],
            "source_profiles_are_frozen":True,
            "has_no_questions_answers_or_routing_table":True,
            "uncompressed_sha256":sha(raw),
            "uncompressed_bytes":len(raw),
            "deflate_bytes":len(packed),"tensors":index}
    h=json.dumps(header,sort_keys=True,separators=(",",":")).encode()
    result=BANK_MAGIC+struct.pack("<I",len(h))+h+packed
    out.write_bytes(result)
    if out.read_bytes()!=result or zlib.decompress(result[len(BANK_MAGIC)+4+len(h):])!=raw:
        raise AssertionError("Lossless alternate trainable weights failed exact roundtrip")
    return result,header


def per_identity_union(reports):
    keys=set(reports["S87"])
    if any(set(r)!=keys for r in reports.values()):
        raise AssertionError("S87/S91/S93 identity set changed")
    by_profile={p:set() for p in PROFILES}
    first_profile={p:set() for p in PROFILES}
    for ident in sorted(keys):
        base=reports["S87"][ident]
        for profile in PROFILES:
            item=reports[profile][ident]
            for field in ("question_sha256","target_ids","S82_ids"):
                if item[field]!=base[field]:
                    raise AssertionError("TRAIN frozen identity mismatch: "+ident)
            gold=item["target_ids"]
            response=item["S87_ids"]
            if response==gold:
                by_profile[profile].add(ident)
            if response and response[0]==gold[0]:
                first_profile[profile].add(ident)
    union=set.union(*by_profile.values())
    protected=by_profile["S87"]|by_profile["S91"]
    protected_first=first_profile["S87"]|first_profile["S91"]
    new_from_s93=by_profile["S93"]-protected
    if tuple(len(by_profile[p]) for p in PROFILES)!=(196,257,268) or len(protected)!=282:
        raise AssertionError("S94 frozen physical TRAIN baseline changed")
    if len(new_from_s93)!=5 or len(union)!=287:
        raise AssertionError("Unexpected S93 beyond-union physical TRAIN outputs")
    return {
        "fully_correct_by_frozen_profile":{p:len(by_profile[p]) for p in PROFILES},
        "full_EOS_union_S87_S91":len(protected),
        "first_token_union_S87_S91":len(protected_first),
        "S93_new_full_beyond_S87_S91":len(new_from_s93),
        "full_EOS_oracle_upper_bound_three_profiles":len(union),
        "three_profile_oracle_upper_bound_rate":len(union)/456,
        "S93_base_full_forgetting_vs_S87_S91":len(protected-by_profile["S93"]),
        "S87_only_repair_examples":len(by_profile["S87"]-by_profile["S91"]),
        "S91_only_protected_examples":len(by_profile["S91"]-by_profile["S87"]),
        "S93_exclusive_new_examples":len(new_from_s93),
        "routing_labels_not_stored_in_physical_bank":True,
    },by_profile,first_profile


def representative_cpu_parity(base_raw,side,archives,all_sides,reports,by_profile):
    """Switch only paid dictionaries and tunnel weights in ONE core.

    Exact CPU inference on a representative subset verifies that dynamically
    selecting profiles reconstructs frozen physical reports without storing
    3 complete pretrained cores. It is NOT full-456 routed certification.
    """
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    samples={}
    grouped=s87.families(tok)
    rows={}
    for family,pairs in grouped.items():
        for pair in pairs:
            for side_name in ("left","right"):
                key=f'{family}/{pair["pair_id"]}/{side_name}'
                rows[key]=pair[side_name]["question"]
    if set(rows)!=set(reports["S87"]):
        raise AssertionError("Frozen TRAIN prompts changed")
    for k,text in rows.items():
        if sha(text.encode())!=reports["S87"][k]["question_sha256"]:
            raise AssertionError("Prompt text differs from CPU baseline")
    model,sources,first,later,pointer,head,inner=s87.build_cpu(base_raw,side,True)
    assert install_verified_fastpath(inner)==("3","9")
    try:
        for profile in PROFILES:
            with torch.no_grad():
                for key in CHANGED:
                    source=sources[key]
                    vector=torch.as_tensor(np.asarray(
                        archives[profile].handles[key].centroids,
                        dtype=np.float32),device=source.centroids.device)
                    source.centroids.copy_(vector)
                inner.load_state_dict({k.removeprefix("internal_tunnel."):v
                   for k,v in all_sides[profile].items()
                   if k.startswith("internal_tunnel.")},strict=True)
            # Exercise unique strengths plus a deterministic shared sample.
            unique=sorted(by_profile[profile]-
                   set.union(*(by_profile[k] for k in PROFILES if k!=profile)))
            regular=sorted(reports[profile])[::57]
            ids=list(dict.fromkeys(unique[:8]+regular[:8]))
            observed=[]
            for ident in ids:
                clear(sources)
                start=time.perf_counter()
                actual=s82.full_cpu_generate(model,sources,first,later,head,
                                             tok,rows[ident],pointer)
                observed.append(time.perf_counter()-start)
                if actual!=reports[profile][ident]["S87_ids"]:
                    raise AssertionError("Exact switched-physical CPU token parity loss: "+
                                         profile+" "+ident)
            samples[profile]={"cases_checked":len(ids),
                         "full_plus_EOS_token_match_all":True,
                         "median_elapsed_seconds":statistics.median(observed)}
    finally:
        inner.uninstall()
        del model
    return samples


def main():
    p=argparse.ArgumentParser()
    for label in PROFILES:
        p.add_argument(label+"_CB61",type=Path)
        p.add_argument(label+"_sidecar",type=Path)
        p.add_argument(label+"_manifest",type=Path)
        p.add_argument(label+"_456_report",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    raws={name:getattr(a,name+"_CB61").read_bytes() for name in PROFILES}
    sides_raw={name:getattr(a,name+"_sidecar").read_bytes() for name in PROFILES}
    manifests={name:json.loads(getattr(a,name+"_manifest").read_text()) for name in PROFILES}
    for profile in PROFILES:
        meta=manifests[profile]
        if (meta.get("source_CB61_physical_sha256")!=sha(raws[profile]) or
            meta.get("prototype_sidecar_sha256")!=sha(sides_raw[profile])):
            raise AssertionError("Immutable physical provenance mismatch: "+profile)
    if (manifests["S93"].get("base_source_S91_sha256")!=sha(raws["S91"]) or
        manifests["S91"].get("base_source_S87_sha256")!=sha(raws["S87"])):
        raise AssertionError("S94 chain of pretrained physical source provenance broken")
    bare={name:s87.open_envelope(raw) for name,raw in raws.items()}
    archives={name:CB59(raw) for name,raw in bare.items()}
    sides={name:load_file(str(getattr(a,name+"_sidecar"))) for name in PROFILES}
    differences=check_sources(archives)
    blocks=check_controllers(sides)
    reports={name:freeze_report(getattr(a,name+"_456_report")) for name in PROFILES}
    union,by_profile,first_profile=per_identity_union(reports)
    bank,bank_index=lossless_bank(archives,sides,blocks,a.out/"S94_ALTERNATE_EXACT_CORE_TUNNEL_BANK.bin")
    meta={"stage":"S94_COMPACT_EXACT_THREE_PROFILE_FEASIBILITY_TRAIN_ONLY",
        "model_core":"S93 physical 36M CB59, unchanged except paid dual dictionaries at runtime",
        "source_SHA256":{p:sha(raws[p]) for p in PROFILES},
        "sidecar_SHA256":{p:sha(sides_raw[p]) for p in PROFILES},
        "frozen_456_TRAIN_only":True,
        "QA_TEST_never_accessed":True,"LIVE_untouched":True,
        "candidate_accepted":False,"actual_trained_router_exists":False,
        "physical_bank_sha256":sha(bank),"physical_bank_bytes":len(bank),
        "changed_core_codebooks":differences,
        "alt_tunnel_tensor_count_per_profile":len(blocks),
        "three_profile_frozen_full_EOS_evidence":union,
        "intended_future_router":"prompt-only learned classifier, no prompt/hash-to-answer lookup"}
    path=a.out/"S94_PROFILE_BANK_FEASIBILITY_REPORT.json"
    for _ in range(20):
        data=json.dumps(meta,indent=2,sort_keys=True).encode()
        total=len(raws["S93"])+len(sides_raw["S93"])+len(bank)+len(data)
        fields={"manifest_bytes":len(data),
                "base_S93_physical_bytes":len(raws["S93"])+len(sides_raw["S93"]),
                "physical_all_inclusive_bytes":total,
                "physical_all_inclusive_bpp":round(8*total/SOURCE_PARAMS,12),
                "all_inclusive_bpp_at_most_0_7":8*total/SOURCE_PARAMS<=MAX_BPP}
        if all(meta.get(k)==v for k,v in fields.items()):
            break
        meta.update(fields)
    if not meta["all_inclusive_bpp_at_most_0_7"]:
        path.write_text(json.dumps(meta,indent=2,sort_keys=True))
        print(json.dumps({"event":"S94_BANK_PHYSICAL_BUDGET_FAILED",
            "bpp":meta["physical_all_inclusive_bpp"],
            "bank_bytes":len(bank),"accepted":False}),flush=True)
        raise SystemExit("S94 rejected: three exact physical profiles exceed 0.7 bpp")
    parity=representative_cpu_parity(bare["S93"],sides["S93"],
                                    archives,sides,reports,by_profile)
    meta["switched_exact_CPU_subset_parity"]=parity
    data=json.dumps(meta,indent=2,sort_keys=True).encode()
    for _ in range(20):
        total=len(raws["S93"])+len(sides_raw["S93"])+len(bank)+len(data)
        fields={"manifest_bytes":len(data),"physical_all_inclusive_bytes":total,
                "physical_all_inclusive_bpp":round(8*total/SOURCE_PARAMS,12),
                "all_inclusive_bpp_at_most_0_7":8*total/SOURCE_PARAMS<=MAX_BPP}
        meta.update(fields)
        updated=json.dumps(meta,indent=2,sort_keys=True).encode()
        if updated==data: break
        data=updated
    path.write_bytes(data)
    print(json.dumps({"event":"S94_LOSSLESS_ALTERNATIVE_MODEL_PROFILE_BANK_PHYSICAL_FEASIBILITY",
        "bpp":meta["physical_all_inclusive_bpp"],
        "exact_alternative_bank_bytes":len(bank),
        "full_EOS_frozen_three_profile_oracle_upper_bound":union["full_EOS_oracle_upper_bound_three_profiles"],
        "full_EOS_oracle_fraction":union["three_profile_oracle_upper_bound_rate"],
        "sample_CPU_token_parity":parity,
        "learned_router_tested":False,"candidate_accepted":False,
        "LIVE_untouched":True}),flush=True)


if __name__=="__main__":
    main()
