"""Physically complete CB59-layout source, but operator coordinates Hadamard-rotated.

Reader is the proven CB59 binary parser. A distinct mandatory v180 format
metadata field makes accidental use in the original unrotated native runtime
detectable; inferencing MUST use v180.native with activation transforms.
"""
import hashlib
import json
import struct
import zlib

import numpy as np
from direct_codebook_v59 import CB59, HEADER, compress_matrix
from v174.protocol import family
from .protocol import ARMS, BLOCKS, SOURCE_SHA, TOTAL_UNIQUE, MAX_BYTES
from .rotation import apply

def encode(original,teacher_state,arm):
    if arm not in ARMS or not isinstance(original,CB59) or original.hash()!=SOURCE_SHA:
        raise ValueError("Wrong pinned CB59 source or unknown rotation arm")
    if original.meta["original_unique_parameter_count"]!=TOTAL_UNIQUE:
        raise ValueError("Expected complete original source")
    if set(teacher_state)!=set(original.meta["full_original_state_keys"]):
        raise ValueError("Missing teacher original matrix")
    block=BLOCKS[arm]
    _,_,old_mlen,_=HEADER.unpack_from(original.blob)
    oldbody=memoryview(original.blob)[HEADER.size+old_mlen:-36]
    body=bytearray()
    desc={}
    evidence={}
    group_energy={name:{"weight_sq":0.,"error_sq":0.,"matrices":0} for name in ("embedding_head","attention","mlp")}
    for name,d in sorted(original.meta["tensor_descriptors"].items()):
        shape=tuple(d["shape"])
        if d["dtype"]=="CB59_K32V8":
            weight=teacher_state[name].detach().cpu().numpy().astype(np.float32,copy=False)
            if weight.shape!=shape:
                raise ValueError("Teacher/source shape mismatch: "+name)
            rotated=apply(weight,block)
            compressed=compress_matrix(rotated)
            fields={k:compressed[k] for k in ("packed","row_scales","centroids")}
            dtype="CB59_K32V8"
            energy=compressed["source_squared_energy"]
            error=compressed["squared_error"]
            evidence[name]={"original_shape":list(shape),"group_bits":5,
                "transformed_relative_weight_RMSE":compressed["relative_RMSE"],
                "original_energy":energy,"squared_error":error,
                "original_weight_hash":hashlib.sha256(weight.tobytes()).hexdigest()}
            f=family(name)
            group_energy[f]["weight_sq"]+=energy
            group_energy[f]["error_sq"]+=error
            group_energy[f]["matrices"]+=1
        elif d["dtype"]=="f4":
            off,size=d["parts"]["raw"]
            fields={"raw":bytes(oldbody[off:off+size])}
            dtype="f4"
        else:
            raise ValueError("Unrecognized state tensor: "+name)
        desc[name]={"shape":list(shape),"dtype":dtype,"parts":{}}
        for part in fields:
            block_data=fields[part]
            desc[name]["parts"][part]=[len(body),len(block_data)]
            body.extend(block_data)
    if sum(f["matrices"] for f in group_energy.values()) != 211:
        raise AssertionError("Not all 211 trained 2D source matrices encoded")
    meta=dict(original.meta)
    meta["tensor_descriptors"]=desc
    meta["calibri_rotation"]={"kind":"signed_block_fwht","size":block,
                               "constant_sign_seed":2654435761,
                               "untransformed_native_decoder_forbidden":True}
    meta["quality_not_guaranteed"]=True
    manifest=json.dumps(meta,sort_keys=True,separators=(",",":")).encode("utf8")
    if len(manifest)>5_000_000 or len(body)>300_000_000:
        raise ValueError("Oversized physical format")
    prefix=HEADER.pack(b"CB59",1,len(manifest),len(body))+manifest+bytes(body)
    marked=prefix+struct.pack("<I",zlib.crc32(prefix))
    blob=marked+hashlib.sha256(marked).digest()
    if len(blob)>MAX_BYTES:
        raise ValueError("Complete rotated source exceeds physical 0.7 bpp")
    candidate=CB59(blob)
    if candidate.meta["calibri_rotation"]["size"]!=block or len(candidate.handles)!=212:
        raise AssertionError("Native-source provenance/aliases missing")
    if candidate.handles["model.embed_tokens.weight"] is not candidate.handles["lm_head.weight"]:
        raise AssertionError("Tied source must be physically shared")
    family_rmse={name:{"matrices":v["matrices"],
       "weighted_relative_weight_RMSE":float(np.sqrt(v["error_sq"]/v["weight_sq"]))}
       for name,v in group_energy.items()}
    return blob,{"sha256":candidate.hash(),"bytes":len(blob),
        "bpp":8*len(blob)/TOTAL_UNIQUE,
        "source_sha256":SOURCE_SHA,"variant":arm,"fwht_block":block,
        "family_transformed_source_RMSE":family_rmse,
        "matrix_details":evidence,
        "all_physical_file_bytes_including_metadata_and_checksums":True,
        "FP32_teacher_discarded_for_inference":True}

def read(blob,arm):
    if arm not in ARMS:
        raise ValueError("Invalid orthogonal operator selection")
    model=CB59(blob)
    rotation=model.meta.get("calibri_rotation",{})
    if rotation!={"kind":"signed_block_fwht","size":BLOCKS[arm],
                 "constant_sign_seed":2654435761,
                 "untransformed_native_decoder_forbidden":True}:
        raise ValueError("Missing required signed source basis metadata")
    if model.meta["original_unique_parameter_count"]!=TOTAL_UNIQUE:
        raise ValueError("Original source parameter count drift")
    return model
