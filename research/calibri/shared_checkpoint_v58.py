#!/usr/bin/env python3
"""CALIBRI CB58 shared-weights aware complete signed INT6 checkpoint.

A model such as SmolLM2-135M-Instruct ties embedding and output weights.
Counting or storing one actual tensor twice would falsify bits/parameter.
Store the unique matrices in verified CB46/CF15 six-bit codec and add a
tiny CRC/SHA checked manifest of tensor alias keys, retaining the exact
original tying. This is CHECKPOINT compression, not compressed hot RAM.

All 2D floating tensors use 6-bit signed CF15 rows with FP16 row scales.
Raw 1D vectors remain exact source dtype. Each source tensor is
reconstructed into dense FP32 runtime on cold load for speed.
"""
from __future__ import annotations
import hashlib,json,struct,zlib
from pathlib import Path
import torch
from full_checkpoint_v46 import pack_state,unpack_state

MAGIC=b"CB58"
HEADER=struct.Struct("<4sBII")
FOOTER_BYTES=36


def split_aliases(source):
    """Find exact tied tensors, avoid duplicate physical storage/cap accounting.

    Storage identity is evaluated only WITHIN one source process. File
    alias references are stable explicit parameter NAMES, not pointers.
    """
    unique={}
    aliases={}
    seen={}
    for name,t in source.items():
        if not isinstance(t,torch.Tensor):
            raise ValueError("state contains non-tensor")
        if not t.is_contiguous():
            raise ValueError("CB58 needs contiguous pretrained source")
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),
             tuple(t.shape),tuple(t.stride()),t.dtype)
        if key in seen:
            canonical=seen[key]
            if not torch.equal(t,source[canonical]):
                raise ValueError("unverified distinct tensor storage alias")
            aliases[name]=canonical
        else:
            unique[name]=t.detach().cpu().contiguous()
            seen[key]=name
    if not unique:
        raise AssertionError("empty pretrained source")
    return unique,aliases


def pack_shared(source):
    unique,aliases=split_aliases(source)
    archive,manifest=pack_state(unique)
    record={
        "alias_derived_tensors":dict(sorted(aliases.items())),
        "unique_tensor_names":sorted(unique),
        "original_state_tensor_names":sorted(source),
        "unique_source_parameter_count":sum(v.numel() for v in unique.values()),
        "codec_of_unique_source":"CB46 signed INT6 physical 2D + exact vectors",
        "schema":"CB58 shared trained weight aliases version 1"}
    meta=json.dumps(record,sort_keys=True,separators=(",",":")).encode()
    if len(meta)>1_000_000 or len(archive)>1_000_000_000:
        raise ValueError("source archive too large for CB58 bounded parser")
    raw=HEADER.pack(MAGIC,1,len(meta),len(archive))+meta+archive
    with_crc=raw+struct.pack("<I",zlib.crc32(raw))
    blob=with_crc+hashlib.sha256(with_crc).digest()
    return blob,record,unique,manifest


def unpack_shared(blob,expected_keys=None):
    if len(blob)<HEADER.size+FOOTER_BYTES:
        raise ValueError("truncated physically stored CB58 checkpoint")
    magic,version,mlen,blen=HEADER.unpack_from(blob)
    if magic!=MAGIC or version!=1 or not 0<mlen<1_000_000 or not 0<blen<1_000_000_000:
        raise ValueError("invalid CB58 header/length")
    if HEADER.size+mlen+blen+FOOTER_BYTES!=len(blob):
        raise ValueError("invalid physical CB58 length")
    if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
        raise ValueError("wrong physical CB58 checksum SHA256")
    if zlib.crc32(blob[:-36])!=struct.unpack_from("<I",blob,len(blob)-36)[0]:
        raise ValueError("wrong physical CB58 CRC")
    meta=json.loads(blob[HEADER.size:HEADER.size+mlen].decode())
    source=unpack_state(blob[HEADER.size+mlen:-36])
    alias=meta["alias_derived_tensors"]
    unique_keys=meta["unique_tensor_names"]
    all_keys=meta["original_state_tensor_names"]
    if sorted(source)!=unique_keys or sorted(set(source)|set(alias))!=all_keys:
        raise ValueError("CB58 source tensor list inconsistent with physical manifest")
    if set(alias)&set(source):
        raise ValueError("alias duplicates a stored unique source tensor")
    if meta["unique_source_parameter_count"]!=sum(v.numel() for v in source.values()):
        raise ValueError("CB58 unique parameter count disagrees with decoded source")
    for name,target in sorted(alias.items()):
        if target not in source or name==target:
            raise ValueError("alias reference not a verified unique trained tensor")
        source[name]=source[target]
    if expected_keys is not None and set(source)!=set(expected_keys):
        raise ValueError("source model is not a complete trained state checkpoint")
    return source,meta


def cold_load_shared(model,path):
    data=Path(path).read_bytes()
    restored,meta=unpack_shared(data,expected_keys=model.state_dict().keys())
    before_in=model.get_input_embeddings().weight.data_ptr()
    before_out=model.get_output_embeddings().weight.data_ptr()
    outcome=model.load_state_dict(restored,strict=True)
    if outcome.missing_keys or outcome.unexpected_keys:
        raise AssertionError("not a complete pretrained INT6 runtime")
    if (before_in==before_out)!=(model.get_input_embeddings().weight.data_ptr()
                                   ==model.get_output_embeddings().weight.data_ptr()):
        raise AssertionError("tied runtime source weight identity changed")
    return {
        "actual_entire_checkpoint_bytes":len(data),
        "actual_entire_checkpoint_SHA256":hashlib.sha256(data).hexdigest(),
        "unique_trained_source_parameters":meta["unique_source_parameter_count"],
        "cold_loaded_dense_hot_bytes":sum(v.numel()*v.element_size()
             for v in model.parameters()),
        "alias_names":meta["alias_derived_tensors"],
    }


def smoke():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed=torch.nn.Embedding(8,12)
            self.lm=torch.nn.Linear(12,8,bias=False)
            self.lm.weight=self.embed.weight
            self.linear=torch.nn.Linear(12,12)
        def get_input_embeddings(self):return self.embed
        def get_output_embeddings(self):return self.lm
    m=Toy()
    assert m.embed.weight.data_ptr()==m.lm.weight.data_ptr()
    blob,meta,unique,_=pack_shared(m.state_dict())
    if meta["alias_derived_tensors"]!={"lm.weight":"embed.weight"}:
        raise AssertionError("tied source aliases not recorded by name")
    loaded,record=unpack_shared(blob,expected_keys=m.state_dict().keys())
    if loaded["lm.weight"] is not loaded["embed.weight"]:
        raise AssertionError("tied reconstruction uses duplicate tensor memory")
    altered=bytearray(blob);altered[len(blob)//2]^=1
    try:unpack_shared(bytes(altered))
    except ValueError:pass
    else:raise AssertionError("truncated/corrupted source hash not rejected")
    print("CALIBRI v58 tied CB58/CB46 original all-parameter INT6 bytes, alias, CRC, SHA smoke PASS")


if __name__=="__main__":
    smoke()
