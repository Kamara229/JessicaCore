"""CALIBRI v179: exact source-index entropy and byte-level coding feasibility.

Reads ONLY pinned original physically-packed source. No teacher FP32, QA,
model generation, private TEST or production state. Statistical audit,
NOT a deployed entropy-coded model.
"""
import argparse
import hashlib
import json
import lzma
import time
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from direct_codebook_v59 import unpack5
from v174.intervene import extract_original
from v175.protocol import SOURCE_SHA, SOURCE_BYTES, TOTAL_UNIQUE
from v174.protocol import family


def entropy(counts):
    counts=np.asarray(counts,dtype=np.float64)
    if counts.ndim!=1 or np.any(counts<0):
        raise ValueError("Invalid histogram")
    total=float(counts.sum())
    if total<=0:
        return 0.0
    positive=counts[counts>0]/total
    return float(-(positive*np.log2(positive)).sum())


def conditioned_entropy(rows):
    """H(code at group t | previous group t-1), within each packed row."""
    codes=np.asarray(rows)
    if codes.ndim!=2 or codes.shape[1]<2 or np.any(codes>=32):
        raise ValueError("Invalid physical code rows")
    pair=np.bincount((codes[:,:-1].astype(np.uint16)*32+
                       codes[:,1:].astype(np.uint16)).reshape(-1),minlength=1024).reshape(32,32)
    return entropy(pair.reshape(-1))-entropy(pair.sum(axis=1))


def family_name(name):
    group=family(name)
    if group=="embedding_head":
        return "embedding_head"
    if group=="attention":
        return "attention"
    return "mlp_down" if name.endswith(".down_proj.weight") else "mlp_gate_up"


def analysis(source,out):
    original=extract_original(source)
    if len(original.blob)!=SOURCE_BYTES or original.hash()!=SOURCE_SHA:
        raise AssertionError("Not original immutable CB59 checkpoint")
    if original.meta["original_unique_parameter_count"]!=TOTAL_UNIQUE:
        raise AssertionError("Unexpected original model dimensionality")
    matrix_desc=original.meta["tensor_descriptors"]
    info={}
    raw_packed=[]
    weighted_marginal=0.0
    weighted_conditioned=0.0
    weighted_conditioned_pairs=0
    matrices=0
    total_groups=0
    family_sum=defaultdict(lambda:{"groups":0,"bits_marginal":0.0,"matrix_count":0})
    global_counts=np.zeros(32,dtype=np.int64)
    for name,d in sorted(matrix_desc.items()):
        if d["dtype"]!="CB59_K32V8":
            continue
        h=original.handles[name]
        m,n=h.shape
        ngroups=n//8
        count=m*ngroups
        raw=h.packed.tobytes()
        expected=(count*5+7)//8
        if len(raw)!=expected:
            raise AssertionError("Noncanonical original packed logical matrix")
        code=unpack5(raw,count).reshape(m,ngroups)
        counts=np.bincount(code.reshape(-1),minlength=32)
        marginal=entropy(counts)
        conditional=conditioned_entropy(code)
        fam=family_name(name)
        info[name]={"family":fam,"shape":list(h.shape),
                    "logical_groups":count,"bytes":len(raw),
                    "marginal_bits_per_5bit_index":marginal,
                    "adjacent_group_conditional_bits":conditional,
                    "used_centroids":int(np.count_nonzero(counts)),
                    "index_sha256":hashlib.sha256(raw).hexdigest()}
        weighted_marginal+=count*marginal
        weighted_conditioned+=m*(ngroups-1)*conditional
        weighted_conditioned_pairs+=m*(ngroups-1)
        total_groups+=count
        global_counts+=counts
        family_sum[fam]["groups"]+=count
        family_sum[fam]["bits_marginal"]+=count*marginal
        family_sum[fam]["matrix_count"]+=1
        raw_packed.append(raw)
        matrices+=1
    if matrices!=211 or total_groups!=16809984:
        raise AssertionError("Incomplete original indexed source tensors")
    complete=b"".join(raw_packed)
    if len(complete)!=10506240:
        raise AssertionError("Non-canonical 5-bit packed index count")
    start=time.perf_counter()
    zlib_bytes=zlib.compress(complete,level=9)
    zlib_elapsed=time.perf_counter()-start
    start=time.perf_counter()
    lzma_bytes=lzma.compress(complete,preset=0)
    lzma_elapsed=time.perf_counter()-start
    if zlib.decompress(zlib_bytes)!=complete or lzma.decompress(lzma_bytes)!=complete:
        raise AssertionError("Byte-level entropy-coder roundtrip failure")
    overheads={"zlib9":len(zlib_bytes),"lzma0":len(lzma_bytes)}
    families={k:{"matrices":v["matrix_count"],"groups":v["groups"],
                 "average_marginal_bits":v["bits_marginal"]/v["groups"]}
              for k,v in sorted(family_sum.items())}
    result={
      "experiment":"v179_exact_CB59_physical_index_entropy",
      "original_source_sha256":SOURCE_SHA,
      "original_source_bytes":SOURCE_BYTES,
      "unique_original_parameters":TOTAL_UNIQUE,
      "full_source_is_same_immutable_original":True,
      "all_unique_source_matrices":matrices,
      "original_packed_index_bytes":len(complete),
      "original_packed_index_bits_per_group":5,
      "index_groups":total_groups,
      "global_centroid_id_entropy":entropy(global_counts),
      "weighted_average_per_matrix_marginal_bits":weighted_marginal/total_groups,
      "weighted_adjacent_group_conditional_bits":weighted_conditioned/weighted_conditioned_pairs,
      "ideal_per_matrix_marginal_coding_savings_bytes":(5*total_groups-weighted_marginal)/8,
      "measured_lossless_packed_stream": {
          name:{"compressed_bytes":n,
                "difference_bytes":len(complete)-n,
                "compression_ratio":n/len(complete)}
          for name,n in overheads.items()},
      "measured_compress_wall_seconds": {"zlib9":zlib_elapsed,"lzma0":lzma_elapsed},
      "hypothetical_total_bytes_before_new_coder_metadata":{
          name:SOURCE_BYTES-(len(complete)-n) for name,n in overheads.items()},
      "families":families,
      "all_source_tensor_statistics":info,
      "quality_not_measured":True,
      "no_new_compressed_decoder":True,
      "no_claim_of_inference_speed_or_QA_improvement":True,
      "private_TEST_not_accessed":True,
      "no_production_changes":True,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"v179_original_source_index_entropy.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"entropy_complete","source_SHA256":SOURCE_SHA,
           "groups":total_groups,"packed_bytes":len(complete),
           "global_H":result["global_centroid_id_entropy"],
           "marginal_bits":result["weighted_average_per_matrix_marginal_bits"],
           "conditional_bits":result["weighted_adjacent_group_conditional_bits"],
           "actual_byte_savings":{k:v["difference_bytes"] for k,v in result["measured_lossless_packed_stream"].items()},
           "per_family":families}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    analysis(args.source,args.out)
