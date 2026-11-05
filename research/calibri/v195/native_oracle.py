"""Actual original 360M C packed decoder vs exact dense decoded *same* source.

Read-only codebook source diagnostic, NOT a compact model QA score.
No pretrained FP32 HF teacher loaded or used for the oracle, so a
match genuinely excludes numerical C native decoder mistakes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from direct_codebook_v59 import CB59,unpack5
from direct_runtime_v59 import NativeKernel,compile_kernel

from . import protocol as P


def exact_weights(handle):
    m,n=handle.shape
    ids=unpack5(handle.packed.tobytes(),m*(n//8)).reshape(m,n//8)
    a=handle.centroids[ids].reshape(m,n)*handle.row_scales[:,None]
    return np.ascontiguousarray(a,dtype=np.float32)


def run(source_file,output):
    source_file=Path(source_file)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=source_file.read_bytes()
    if len(source)!=P.SOURCE_BYTES or hashlib.sha256(source).hexdigest()!=P.SOURCE_SHA256:
        raise AssertionError("Not the frozen entire original 360M v194 physical source")
    archive=CB59(source)
    if archive.meta["original_unique_parameter_count"]!=P.ORIGINAL_UNIQUE_PARAMETERS:
        raise AssertionError("Source unique original weights incomplete")
    if archive.handles["lm_head.weight"] is not archive.handles["model.embed_tokens.weight"]:
        raise AssertionError("Real native 360M tied head alias missing")
    so=compile_kernel(Path(__file__).resolve().parents[1]/"direct_kernel_v59.c",output/"native_v195_exact.so")
    native=NativeKernel(so)
    rng=np.random.default_rng(195)
    measured=[]
    for name in P.NATIVE_LINEAR_PATHS:
        if name not in archive.handles:
            raise AssertionError("Expected original transformer operator missing: "+name)
        handle=archive.handles[name]
        dense=exact_weights(handle)
        x=np.ascontiguousarray(rng.normal(0,0.25,(2,dense.shape[1])).astype(np.float32))
        result=native.linear(handle,x)
        # BLAS and LUT-C can differ in accumulation order at float32;
        # require small, defensible absolute AND relative numerical error.
        oracle=(x@dense.T).astype(np.float32)
        diff=result-oracle
        l2rel=float(np.linalg.norm(diff.astype(np.float64))/
                    max(np.linalg.norm(oracle.astype(np.float64)),1e-20))
        max_abs=float(np.max(np.abs(diff)))
        bound=bool(np.allclose(result,oracle,rtol=3e-4,atol=3e-4))
        measured.append({"tensor":name,"shape":list(handle.shape),
           "native_vs_exact_decoded_dense_relative_L2":l2rel,
           "native_vs_exact_decoded_dense_max_absolute":max_abs,
           "allclose_0p0003":bound,"source_only_not_FP32_teacher_oracle":True})
        if not bound or l2rel>1.e-4:
            raise AssertionError("Native C vs exact-source oracle numerical mismatch: "+name)
    embed=archive.handles["model.embed_tokens.weight"]
    m,n=embed.shape
    ids=np.array(P.NATIVE_EMBED_ROWS,dtype=np.int64)
    if np.any(ids>=m):raise AssertionError("Original embedding row id invalid")
    via_C=native.embedding(embed,ids)
    codes=unpack5(embed.packed.tobytes(),m*(n//8)).reshape(m,n//8)
    selected=embed.centroids[codes[ids]].reshape(len(ids),n)*embed.row_scales[ids,None]
    result=np.ascontiguousarray(selected,dtype=np.float32)
    if not np.allclose(via_C,result,rtol=1e-6,atol=1e-6):
        raise AssertionError("Original 360M C on-demand embedding diverges from packed oracle")
    max_abs=float(np.max(np.abs(via_C-result)))
    data={"complete":True,
      "no_uncompressed_FP32_source_loaded":True,
      "actual_complete_original_360M_source_sha256":P.SOURCE_SHA256,
      "original_unique_parameters":P.ORIGINAL_UNIQUE_PARAMETERS,
      "original_complete_binary_bytes":len(source),
      "actual_physical_bpp":8*len(source)/P.ORIGINAL_UNIQUE_PARAMETERS,
      "native_C_linear_vs_identical_codebook_dense_oracle":measured,
      "source_input_head_sample_rows_verified":list(P.NATIVE_EMBED_ROWS),
      "C_native_embedding_vs_exact_CB59_dense_max_absolute":max_abs,
      "C_native_numerically_matches_CB59_source":True,
      "C_native_matches_UNCOMPRESSED_original_FP32_unproven":True,
      "no_codebook_updates_no_training":True,
      "dense_reference_tensors_used_DIAGNOSTIC_only":True,
      "no_original_source_QA_test":True,"private_v89_TEST_unread":True}
    (output/"v195_real360M_native_C_vs_identical_source_exact_oracle.json").write_text(
       json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"stage":"v195_NATIVE_NUMERICAL_ORACLE_PASS",
       "source_sha":P.SOURCE_SHA256,
       "tested_full_real_matrices":len(measured),
       "embedding_rows":len(ids),
       "max_linear_rel_L2":max(x["native_vs_exact_decoded_dense_relative_L2"] for x in measured),
       "max_embedding_absolute":max_abs}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("physical_source_file",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    run(args.physical_source_file,args.output)
