"""CALIBRI v218 — per-row-block PAID source dictionaries, without a new bit/index.

The native CB59/v206 format uses one shared 32x8 vector codebook for every
entire original FP32 matrix. Such a global dictionary strongly constrains
tied 49,152x576 vocabulary reconstruction, even after v217 all-weight
fixed-index refit (v217 HEAD 2/16, BODY/FULL 0/16).

This experiment does something genuinely different: partition ONLY the
original tied vocabulary matrix into 128- or 256-row blocks; rebuild
source-only locally fitted K32V8 FP16 dictionaries and rewrite its existing
same-length 5-bit code stream. Both input embedding and output projection
are executed by a genuine native packed C kernel with local dictionaries;
all remaining 210 original native packed matrices are unmodified. There
are no sidecar neural parameters beyond the physically stored dictionaries.

Physical CB59 source + WHOLE sidecar file must be <=0.700 bpw. Both
candidate geometries are fixed here before any QA; do not pick models
based on repeatedly viewed public DEV16. No QA/TRAIN/VALID/TEST content
is consulted for codebook fitting. Authentic original FP32 *weights* only.
No live Jessica/Render/Supabase/APK changes, no Jessica tests.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import struct
import zlib

import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59,HEADER,pack5,unpack5
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v174.intervene import extract_original
from v174.protocol import freeze_cases
from v174.run import save
from v169.sources import all_cases
from v186.run import speed_pairs
from v196 import protocol as p
from v196.run import newly_frozen_public_dev_16
from v212.run import SOURCE_V206_SHA256

VOCAB_NAME="model.embed_tokens.weight"
VOCAB=49152
EMBED=576
DIMS=8
K=32
GROUPS=EMBED//DIMS
BLOCK_SIZES=(256,128)
SOURCE_COST_LIMIT=0.700
INIT_SAMPLE=1536
INIT_ITERATIONS=4
FULL_ITERATIONS=2


def nearest_group_indices(vectors,centers):
    if (vectors.ndim!=2 or centers.shape!=(K,DIMS) or
        vectors.shape[1]!=DIMS):
        raise AssertionError("Wrong vocabulary word-vector dictionary geometry")
    scores=np.square(centers).sum(axis=1)[None,:]-2.0*(vectors@centers.T)
    return np.argmin(scores,axis=1).astype(np.uint8)


def local_source_fit(original,original_handle,block_rows):
    if block_rows not in BLOCK_SIZES or VOCAB%block_rows:
        raise AssertionError("Unapproved source dictionary block size")
    if tuple(original_handle.shape)!=(VOCAB,EMBED) or original.shape!=(VOCAB,EMBED):
        raise AssertionError("Wrong original tied trained vocabulary geometry")
    scales=original_handle.row_scales.astype(np.float32)
    if not np.isfinite(scales).all() or np.any(scales<=0):
        raise AssertionError("Invalid physical v206 original row scales")
    old_ids=unpack5(original_handle.packed.tobytes(),VOCAB*GROUPS)
    old_matrix=original_handle.centroids[old_ids].reshape(VOCAB,EMBED)*scales[:,None]
    baseline_SSE=float(np.sum((original.astype(np.float64)-
                               old_matrix.astype(np.float64))**2))
    nblocks=VOCAB//block_rows
    target_codes=np.empty((VOCAB,GROUPS),dtype=np.uint8)
    block_centroids=np.empty((nblocks,K,DIMS),dtype=np.float16)
    new_SSE=0.0
    for block in range(nblocks):
        start=block*block_rows
        stop=start+block_rows
        part=original[start:stop]
        scal=scales[start:stop]
        groups=part.reshape(-1,DIMS).astype(np.float32)
        scale_vec=np.repeat(scal,GROUPS)
        vectors=groups/scale_vec[:,None]
        if not np.isfinite(vectors).all():
            raise AssertionError("Source-only normalized FP32 weight invalid")
        index=np.linspace(0,len(vectors)-1,min(INIT_SAMPLE,len(vectors)),
                          dtype=np.int64)
        sample=vectors[index]
        initial=np.linspace(0,len(sample)-1,K,dtype=np.int64)
        cent=sample[initial].astype(np.float16).astype(np.float32)
        for _ in range(INIT_ITERATIONS):
            assigned=nearest_group_indices(sample,cent)
            counts=np.bincount(assigned,minlength=K)
            for c in range(K):
                if counts[c]:
                    cent[c]=sample[assigned==c].mean(axis=0)
            cent=cent.astype(np.float16).astype(np.float32)
        # One complete assignment+weighted-refit iteration over ALL genuine
        # FP32 source vectors in each local block, not a sampled QA subset.
        weight_vec=np.square(scale_vec,dtype=np.float32)
        for _ in range(FULL_ITERATIONS):
            assigned=nearest_group_indices(vectors,cent)
            denominator=np.bincount(assigned,weights=weight_vec,
                                    minlength=K).astype(np.float64)
            num=np.stack([np.bincount(
                assigned,weights=weight_vec*vectors[:,dim],
                minlength=K) for dim in range(DIMS)],axis=1)
            occupied=denominator>1e-20
            cent[occupied]=(num[occupied]/denominator[occupied,None]).astype(np.float32)
            cent=cent.astype(np.float16).astype(np.float32)
        final_ids=nearest_group_indices(vectors,cent)
        target_codes[start:stop]=final_ids.reshape(block_rows,GROUPS)
        block_centroids[block]=cent.astype(np.float16)
        decoded=cent[final_ids]*scale_vec[:,None]
        new_SSE+=float(np.sum(
            (groups.astype(np.float64)-decoded.astype(np.float64))**2))
        if (block+1)%64==0 or block+1==nblocks:
            print(json.dumps({"stage":"v218_FP32_SOURCE_ONLY_partitioned_codebook",
                "block_rows":block_rows,
                "completed_blocks":block+1,"total_blocks":nblocks,
                "no_QA_or_hidden_state_training":True}),flush=True)
    if not new_SSE<baseline_SSE:
        raise AssertionError("Local source-only representation did not improve original weight MSE")
    return target_codes.reshape(-1),block_centroids,{
        "block_rows":block_rows,"codebook_count":nblocks,
        "actual_original_FP32_source_vectors":VOCAB*GROUPS,
        "original_v206_head_weight_SSE":baseline_SSE,
        "localized_head_weight_SSE":new_SSE,
        "localized_head_SSE_ratio":new_SSE/max(baseline_SSE,1e-30),
        "packed_group_code_bitwidth":5,
        "K32V8_centroids_per_block":True,
        "all_saved_dictionaries_FP16":True,
        "fit_uses_source_weights_only_NO_QA_TRAIN_VALID_DEV_or_TEST":True
    }


def repack_head_indices(original_bytes,original_archive,new_codes):
    if len(original_bytes)!=p.SOURCE_BYTES:
        raise AssertionError("Unsigned source size changed")
    info=original_archive.meta["tensor_descriptors"][VOCAB_NAME]
    head_offset,head_length=info["parts"]["packed"]
    encoded=pack5(new_codes)
    if len(encoded)!=head_length:
        raise AssertionError("New head indices must use exact original bit count")
    _,_,meta_length,_=HEADER.unpack_from(original_bytes)
    origin=HEADER.size+meta_length+head_offset
    raw=bytearray(original_bytes)
    raw[origin:origin+head_length]=encoded
    raw[-36:-32]=struct.pack("<I",zlib.crc32(raw[:-36]))
    raw[-32:]=hashlib.sha256(raw[:-32]).digest()
    rebuilt=bytes(raw)
    archive=CB59(rebuilt)
    if (archive.meta!=original_archive.meta or len(rebuilt)!=len(original_bytes) or
        archive.handles["lm_head.weight"] is not archive.handles[VOCAB_NAME]):
        raise AssertionError("Broken original source ABI or head aliases")
    assert rebuilt[:origin]==original_bytes[:origin]
    assert rebuilt[origin+head_length:-36]==original_bytes[origin+head_length:-36]
    return rebuilt


class PartitionedKernel:
    def __init__(self,path):
        self.lib=ctypes.CDLL(str(path))
        ptr=ctypes.c_void_p
        i64=ctypes.c_int64
        args=[ptr,ctypes.c_size_t,ptr,ptr,ptr,i64,i64,i64,i64,ptr]
        self.line=self.lib.cb218_part_linear
        self.line.argtypes=args; self.line.restype=ctypes.c_int
        self.emb=self.lib.cb218_part_embedding
        self.emb.argtypes=args; self.emb.restype=ctypes.c_int
    @staticmethod
    def ptr(x):
        return ctypes.c_void_p(int(x.ctypes.data))
    def linear(self,handle,centers,rows_per_block,x):
        arr=np.ascontiguousarray(x,dtype=np.float32)
        if arr.ndim!=2 or arr.shape[1]!=handle.shape[1]:
            raise AssertionError("Wrong native partitioned linear activation")
        y=np.empty((len(arr),handle.shape[0]),dtype=np.float32)
        rc=self.line(self.ptr(handle.packed),len(handle.packed),
                     self.ptr(handle.row_scales),self.ptr(centers),
                     self.ptr(arr),len(arr),handle.shape[0],handle.shape[1],
                     rows_per_block,self.ptr(y))
        if rc or not np.isfinite(y).all():
            raise AssertionError("New native partitioned 5bit linear failure "+str(rc))
        return y
    def embedding(self,handle,centers,rows_per_block,ids):
        tokens=np.ascontiguousarray(ids,dtype=np.int64).reshape(-1)
        result=np.empty((len(tokens),handle.shape[1]),dtype=np.float32)
        rc=self.emb(self.ptr(handle.packed),len(handle.packed),
                    self.ptr(handle.row_scales),self.ptr(centers),
                    self.ptr(tokens),len(tokens),handle.shape[0],handle.shape[1],
                    rows_per_block,self.ptr(result))
        if rc or not np.isfinite(result).all():
            raise AssertionError("New native partitioned 5bit embedding failure "+str(rc))
        return result


class PartitionedEmbedding(nn.Module):
    def __init__(self,handle,kernel,centers,blockrows):
        super().__init__()
        self.handle=handle;self.kernel=kernel
        self.centers=centers;self.blockrows=blockrows
        self.num_embeddings=handle.shape[0]
        self.embedding_dim=handle.shape[1]
    def forward(self,ids):
        x=self.kernel.embedding(self.handle,self.centers,self.blockrows,
                                ids.detach().cpu().numpy())
        return torch.from_numpy(x).reshape(tuple(ids.shape)+(self.embedding_dim,))


class PartitionedLMHead(nn.Module):
    def __init__(self,handle,kernel,centers,blockrows):
        super().__init__()
        self.handle=handle;self.kernel=kernel
        self.centers=centers;self.blockrows=blockrows
        self.in_features=handle.shape[1]
        self.out_features=handle.shape[0]
    def forward(self,x):
        if x.device.type!="cpu" or x.dtype!=torch.float32:
            raise AssertionError("Native source packed C requires CPU FP32")
        arr=x.detach().contiguous().reshape(-1,self.in_features).numpy()
        result=self.kernel.linear(self.handle,self.centers,self.blockrows,arr)
        return torch.from_numpy(result).reshape(tuple(x.shape[:-1])+
                                               (self.out_features,))


def install_blockwise_tied_vocabulary(model,codebooks_path,compiled_C):
    source=model.get_input_embeddings().handle
    if model.get_output_embeddings().handle is not source:
        raise AssertionError("Original native 5bit head and embedding not tied")
    blob=load_file(str(codebooks_path))
    if set(blob)!={"dictionary_fp16"}:
        raise AssertionError("Unexpected unpaid FP32 trained weights in sidecar")
    saved=blob["dictionary_fp16"]
    if (saved.dtype!=torch.float16 or saved.ndim!=3 or
        tuple(saved.shape[1:])!=(K,DIMS) or
        VOCAB%saved.shape[0]):
        raise AssertionError("Unpaid or invalid block-specific vocabulary dictionary")
    blockrows=VOCAB//saved.shape[0]
    if blockrows not in BLOCK_SIZES:
        raise AssertionError("Invalid precommitted vocabulary geometry")
    cent=np.ascontiguousarray(saved.numpy().astype(np.float32))
    kernel=PartitionedKernel(compiled_C)
    model.set_input_embeddings(PartitionedEmbedding(source,kernel,cent,blockrows))
    model.set_output_embeddings(PartitionedLMHead(source,kernel,cent,blockrows))
    if any(p.ndim>=2 for p in model.parameters()):
        raise AssertionError("Original dense trained source persisted in RAM")
    return kernel,source,cent,blockrows


def validate_source_kernel(kernel,handle,cent,blockrows):
    tokens=np.asarray([0,1,127,128,1024,VOCAB-1],dtype=np.int64)
    sparse=kernel.embedding(handle,cent,blockrows,tokens)
    codes=unpack5(handle.packed.tobytes(),VOCAB*GROUPS).reshape(VOCAB,GROUPS)
    expected=cent[(tokens//blockrows)[:,None],codes[tokens]].reshape(-1,EMBED)
    expected=expected*handle.row_scales[tokens,None]
    if not np.allclose(sparse,expected,rtol=2e-5,atol=2e-6):
        raise AssertionError("5-bit row demand embedding kernel failed original weight oracle")
    inp=np.linspace(-0.2,0.3,EMBED,dtype=np.float32)[None,:]
    actual=kernel.linear(handle,cent,blockrows,inp)[0,tokens]
    target=np.sum(expected*inp,axis=1)
    if not np.allclose(actual,target,rtol=1e-4,atol=3e-5):
        raise AssertionError("Native partitioned head C kernel differs from on-demand source")
    return {"actual_5bit_native_embedding_matches_partial_dense_oracle":True,
            "actual_5bit_native_head_matches_partial_dense_oracle":True,
            "dense_reconstruction_ONLY_for_six_unit_test_rows":True}


def run(v206_path,v161_path,v193_path,cshared,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(218)
    original_bytes=Path(v206_path).read_bytes()
    if (hashlib.sha256(original_bytes).hexdigest()!=SOURCE_V206_SHA256 or
        len(original_bytes)!=p.SOURCE_BYTES):
        raise AssertionError("Requires immutable original v206 real native CB59 source")
    archive=CB59(original_bytes)
    teacher=original_teacher().cpu().eval()
    if (teacher.get_input_embeddings().weight.data_ptr()!=
        teacher.get_output_embeddings().weight.data_ptr()):
        raise AssertionError("Real FP32 tied input and output weights changed")
    original_weights=teacher.get_input_embeddings().weight.detach().cpu().numpy()
    signed={}
    for blockrows in BLOCK_SIZES:
        code,centroids,source_diagnostic=local_source_fit(
            original_weights,archive.handles[VOCAB_NAME],blockrows)
        packed=repack_head_indices(original_bytes,archive,code)
        physical_path=out/f"CALIBRI_v218_head_local_{blockrows}rows_original_0p7.cb59"
        physical_path.write_bytes(packed)
        dictionary_path=out/f"CALIBRI_v218_{blockrows}rows_paid_codebooks.safetensors"
        save_file({"dictionary_fp16":torch.from_numpy(
            np.ascontiguousarray(centroids))},str(dictionary_path))
        bpp=8*(len(packed)+dictionary_path.stat().st_size)/p.FROZEN_SOURCE_135M_UNIQUE
        if bpp>SOURCE_COST_LIMIT:
            raise AssertionError("Inclusive source bit budget exceeded")
        signed[str(blockrows)]={
            "complete_native_CB59_source_file":physical_path.name,
            "paid_local_FP16_codebooks_file":dictionary_path.name,
            "original_group_5bit_index_stream_reallocated_same_length":True,
            "same_original_tied_embedding_and_head":True,
            "source_SHA256":hashlib.sha256(packed).hexdigest(),
            "dictionary_SHA256":hashlib.sha256(dictionary_path.read_bytes()).hexdigest(),
            "whole_135m_CB59_bytes":len(packed),
            "whole_dictionary_sidecar_bytes":dictionary_path.stat().st_size,
            "physical_all_inclusive_bpp":bpp,
            "source_weight_diagnostic":source_diagnostic
        }
    del teacher,original_weights
    seal={
        "experiment":"v218_actual_native_partitioned_K32V8_quantization",
        "v206_immutable_source_SHA256":SOURCE_V206_SHA256,
        "original_unique_135m_parameters":p.FROZEN_SOURCE_135M_UNIQUE,
        "precommitted_head_block_sizes":list(BLOCK_SIZES),
        "whole_0p700_count_includes_all_sidecars":True,
        "source_only_NO_TRAIN_QA_VALID_DEV_used_to_select":True,
        "original_other_210_matrices_kept_exactly_as_v206":True,
        "all_sources_sha_sealed_BEFORE_any_public_DEV_QA":True,
        "no_live_or_Jessica_changes":True,
        "candidate_sources":signed
    }
    save(out/"SEALED_v218_source_ONLY_128_256_BEFORE_QA.json",seal)
    print(json.dumps({"stage":"v218_source_only_all_variants_physically_sealed",
       "variants":{key:{"bpp":val["physical_all_inclusive_bpp"],
                        "head_source_SSE_ratio":
                             val["source_weight_diagnostic"]["localized_head_SSE_ratio"]}
                   for key,val in signed.items()}}),flush=True)

    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    initial=extract_original(Path(v161_path))
    if len(initial.blob)!=p.SOURCE_BYTES:
        raise AssertionError("Original native CB59 benchmark baseline missing")
    with TemporaryDirectory(dir=out) as td:
        folder=Path(td)
        for name in ("CB59","CB206",*[str(x) for x in BLOCK_SIZES]):
            (folder/name).mkdir()
        (folder/"CB59"/"source.cb59").write_bytes(initial.blob)
        cb59,_,_=make_model("baseline",SimpleNamespace(
            original=folder/"CB59"/"source.cb59",folder=folder/"CB59"))
        cb206,_,_=make_model("baseline",SimpleNamespace(
            original=Path(v206_path),folder=folder/"CB206"))
        native={"CB206":cb206}
        integrity={}
        for rows in BLOCK_SIZES:
            item=signed[str(rows)]
            new,_,_=make_model("baseline",SimpleNamespace(
                original=out/item["complete_native_CB59_source_file"],
                folder=folder/str(rows)))
            kernel,handle,centroids,blockrows=install_blockwise_tied_vocabulary(
                new,out/item["paid_local_FP16_codebooks_file"],cshared)
            integrity[str(rows)]=validate_source_kernel(
                kernel,handle,centroids,blockrows)
            native[f"CB218_{rows}"]=new
        save(out/"v218_native_C_before_QA_numerical_oracle.json",integrity)
        QA=newly_frozen_public_dev_16(native,tokenizer,v193_path,out)
        if QA["scores"]["CB206"]["machine_confirmed"]!=3:
            raise AssertionError("v206 fixed reference grade changed")
        speed={}
        for rows in BLOCK_SIZES:
            benchmark=speed_pairs({"CB59":cb59,"CB186":native[f"CB218_{rows}"]},
                                   tokenizer,freeze_cases(all_cases()))
            speed[str(rows)]=benchmark["median_speed_ratio_vs_CB59"]
    result={
        "completed":True,
        "version":"CALIBRI_v218_true_native_blockwise_vocabulary_5bit",
        "fp32_publicDEV16_correct":QA["scores"]["FP32_existing_135M"]["machine_confirmed"],
        "baseline_v206_publicDEV16_correct":QA["scores"]["CB206"]["machine_confirmed"],
        "v218_publicDEV16_results":{
            str(rows):{
                "machine_correct":QA["scores"][f"CB218_{rows}"]["machine_confirmed"],
                "machine_wrong":QA["scores"][f"CB218_{rows}"]["machine_recognized_wrong"],
                "ungraded":QA["scores"][f"CB218_{rows}"]["ungraded_not_known_wrong"],
                "physical_all_inclusive_bpp":signed[str(rows)]["physical_all_inclusive_bpp"],
                "native_CPU_speed_ratio_CB59":speed[str(rows)],
                "source_head_SSE_ratio":signed[str(rows)]["source_weight_diagnostic"]["localized_head_SSE_ratio"]
            } for rows in BLOCK_SIZES
        },
        "real_packed_C_tested_before_QA":integrity,
        "public_DEV16_repeated_exploration_NOT_final_independent_TEST":True,
        "no_DEV_sample_based_model_or_checkpoint_selection":True,
        "MAIN_LIVE_Jessica_Render_Supabase_APK_unchanged":True
    }
    save(out/"v218_REAL_NATIVE_135m_partitioned_5bit_QA_CPU.json",result)
    print(json.dumps({"stage":"v218_real_native_C_partitioned_source_QA",**result}),flush=True)
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("signed_v206",type=Path)
    ap.add_argument("original_v161_CB59",type=Path)
    ap.add_argument("original_v193_FP32_QA",type=Path)
    ap.add_argument("compiled_local_dictionary_C_kernel",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    run(args.signed_v206,args.original_v161_CB59,
        args.original_v193_FP32_QA,args.compiled_local_dictionary_C_kernel,
        args.output)
