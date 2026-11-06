"""CALIBRI v216 — physically paid 1 extra bit per tied embedding 8-vector.

CB59 v206 is an unchanged, authentic native C 5-bit/8-vector 135M model.
The root problem is poor vocabulary selectivity: v206 repeats 'No', 'red',
or '5' and all three earlier output-calibration experiments lost quality.

New REPRESENTATION (not another logit-bias distillation): the tied
vocabulary embedding/head gains one 1-bit residual index PER existing
8-weight vector. The 16 adjacent vocabulary rows in each block share
one 8D signed residual direction stored as 8 FP16 numbers. Original
FP32-minus-actual-v206 embedding residual is fitted by an 8x8
eigendecomposition within each 16-row block, without any QA/DEV data.

Physical overhead: 49,152 * (576/8) / 8 = 442,368 index bytes, plus
(49,152/16)*8*2 = 49,152 FP16 dictionary bytes and safetensors header.
Total CB59 + residual sidecar <=0.700 bpp **including everything**.
No full FP32 matrix, hidden neural sidecar, or unpaid weight at inference.
A compact C inner kernel computes the output residual, with packed
1-bit signs and 16-row-shared direction. Embedding rows are decoded
on demand, not whole 28M dense source.
Sealed BEFORE opening the public DEV16 score. No current LIVE changes.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file,load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59,unpack5
from paid_lowrank_v76 import make_model
from v174.intervene import extract_original
from v174.protocol import freeze_cases
from v174.run import save
from v169.sources import all_cases
from v84.sources import original_teacher
from v186.run import speed_pairs
from v196 import protocol as p
from v196.run import newly_frozen_public_dev_16
from v212.run import SOURCE_V206_SHA256

BLOCK_ROWS = 16
GROUP_WIDTH = 8
VOCAB = 49152
HIDDEN = 576
GROUPS = HIDDEN // GROUP_WIDTH
PHYSICAL_BUDGET_BPP = .700
MAX_SEALED_WEIGHT_BYTES = int((PHYSICAL_BUDGET_BPP *
                                p.FROZEN_SOURCE_135M_UNIQUE)//8)


def build_residual(weight,archive):
    """Source-only 1-bit residual code with fixed non-QA geometry."""
    handle=archive.handles["model.embed_tokens.weight"]
    if archive.handles["lm_head.weight"] is not handle:
        raise AssertionError("Tied embedding and LM head source was duplicated")
    if tuple(handle.shape)!=(VOCAB,HIDDEN) or tuple(weight.shape)!=(VOCAB,HIDDEN):
        raise AssertionError("Unexpected real original 135M tied vocabulary matrix")
    codes=unpack5(handle.packed.tobytes(),VOCAB*GROUPS)
    approximate=handle.centroids[codes].reshape(VOCAB,HIDDEN)*handle.row_scales[:,None]
    residual=(np.asarray(weight,dtype=np.float32)-approximate.astype(np.float32)).reshape(
        VOCAB,GROUPS,GROUP_WIDTH)
    if not np.isfinite(residual).all():
        raise AssertionError("Nonfinite quantization residual")
    prior_error=0.0
    remaining_error=0.0
    signs=np.empty((VOCAB,GROUPS),dtype=np.uint8)
    directions=np.empty((VOCAB//BLOCK_ROWS,GROUP_WIDTH),dtype=np.float16)
    block_status=[]
    for block in range(VOCAB//BLOCK_ROWS):
        window=residual[block*BLOCK_ROWS:(block+1)*BLOCK_ROWS].reshape(-1,GROUP_WIDTH)
        # Exactly one normalized principal direction per 16 vocabulary rows.
        covariance=window.T@window
        values, eigenvectors=np.linalg.eigh(covariance)
        direction=eigenvectors[:,-1].astype(np.float32)
        if not np.isfinite(direction).all():
            raise AssertionError("Bad residual eigenspace")
        projections=window@direction
        gain=float(np.mean(np.abs(projections)))
        # Fixed FP16 encoding, TRAIN/QA never used to choose shrinkage.
        signed_vector=np.asarray(direction*gain,dtype=np.float16)
        if not np.isfinite(signed_vector).all():
            raise AssertionError("Unrepresentable signed residual direction")
        directions[block]=signed_vector
        signs[block*BLOCK_ROWS:(block+1)*BLOCK_ROWS]=(
            projections.reshape(BLOCK_ROWS,GROUPS)>=0
        ).astype(np.uint8)
        reconstructed=(np.where(projections[:,None]>=0,
                               signed_vector.astype(np.float32)[None,:],
                               -signed_vector.astype(np.float32)[None,:]))
        prior_error+=float(np.sum(window.astype(np.float64)**2))
        remaining_error+=float(np.sum((window-reconstructed).astype(np.float64)**2))
        if block%512==0:
            print(json.dumps({"stage":"v216_fit_source_only",
                 "blocks":block+1,"of":VOCAB//BLOCK_ROWS}),flush=True)
    packed=np.packbits(signs.reshape(-1),bitorder="little")
    expected=VOCAB*GROUPS//8
    if len(packed)!=expected or len(directions)!=VOCAB//BLOCK_ROWS:
        raise AssertionError("Physical bit accounting for tied vocabulary violated")
    ratio=remaining_error/max(prior_error,1.e-30)
    if not 0<=ratio<1:
        raise AssertionError("One-bit residual source reconstruction failed")
    return {
        "bits":torch.from_numpy(packed.copy()).contiguous(),
        "direction_fp16":torch.from_numpy(directions.copy()).contiguous()
    },{"embedding_source_residual_SSE_before":prior_error,
       "embedding_source_residual_SSE_after":remaining_error,
       "embedding_source_SSE_ratio_after_to_before":ratio,
       "paid_one_bit_indices":int(signs.size),
       "paid_shared_FP16_residual_dictionary_entries":int(directions.size),
       "original_embedding_shape":[VOCAB,HIDDEN],
       "source_only_NO_PUBLIC_TRAIN_QA_VALID_or_DEV":True}


class NativeOneBit:
    def __init__(self,safetensor_path,shared_object_path):
        frozen=load_file(str(safetensor_path))
        if set(frozen)!={"bits","direction_fp16"}:
            raise AssertionError("Unexpected hidden Green Tunnel tensors")
        self.bits=frozen["bits"].detach().cpu().numpy().astype(np.uint8,copy=True)
        self.direction=frozen["direction_fp16"].detach().cpu().numpy().astype(
            np.float32,copy=True)
        if (self.bits.shape!=(VOCAB*GROUPS//8,) or
            self.direction.shape!=(VOCAB//BLOCK_ROWS,GROUP_WIDTH) or
            not np.isfinite(self.direction).all()):
            raise AssertionError("Physically saved one-bit codebook has wrong shape")
        self.signs=np.unpackbits(self.bits,bitorder="little").reshape(
            VOCAB,GROUPS).astype(np.float32)*2.0-1.0
        self.lib=ctypes.CDLL(str(shared_object_path))
        fn=self.lib.calibri_onebit_head_add
        ptr=ctypes.c_void_p
        fn.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int]
        fn.restype=ctypes.c_int
        self.apply=fn

    def add_head(self,hidden, logits):
        hidden_shape=hidden.shape
        if (hidden.dtype!=torch.float32 or logits.dtype!=torch.float32 or
            hidden_shape[-1]!=HIDDEN or logits.shape[:-1]!=hidden_shape[:-1] or
            logits.shape[-1]!=VOCAB):
            raise AssertionError("Wrong native vocabulary head geometry")
        a=hidden.detach().cpu().contiguous().reshape(-1,HIDDEN).numpy()
        y=logits.detach().cpu().contiguous().reshape(-1,VOCAB).numpy()
        ptr=ctypes.c_void_p
        for i in range(len(a)):
            rc=self.apply(ptr(int(a[i].ctypes.data)),
                          ptr(int(y[i].ctypes.data)),ptr(int(self.bits.ctypes.data)),
                          ptr(int(self.direction.ctypes.data)),
                          VOCAB,HIDDEN,BLOCK_ROWS)
            if rc:
                raise AssertionError("Native one-bit residual C kernel rejected input")
        if not np.isfinite(y).all():
            raise AssertionError("Nonfinite corrected real native CB59 head")
        return torch.from_numpy(y).reshape(logits.shape)

    def add_embedding(self,ids,embedding):
        if embedding.dtype!=torch.float32 or embedding.shape[-1]!=HIDDEN:
            raise AssertionError("Wrong native embedding activation")
        array=ids.detach().cpu().numpy().reshape(-1).astype(np.int64)
        if np.any((array<0)|(array>=VOCAB)):
            raise AssertionError("Out-of-vocab embedding ids")
        delta=(self.signs[array,:,None] *
               self.direction[array//BLOCK_ROWS,None,:]).reshape(-1,HIDDEN)
        return embedding+torch.from_numpy(delta).reshape(embedding.shape)

    def install(self,model):
        if (model.get_input_embeddings().handle is not
                model.get_output_embeddings().handle):
            raise AssertionError("Broken native shared 5bit embedding/head")
        def embedding_hook(_module, args, output):
            return self.add_embedding(args[0],output)
        def head_hook(_module,args,output):
            return self.add_head(args[0],output)
        handles=[
            model.get_input_embeddings().register_forward_hook(embedding_hook),
            model.get_output_embeddings().register_forward_hook(head_hook)
        ]
        return handles


def numerical_test(kernel):
    # True C vs algebraic 1bit reconstruction, before any QA score.
    x=np.linspace(-0.8,0.8,HIDDEN,dtype=np.float32)
    # Choose a block with nonzero residual direction.
    block=0
    v=kernel.direction[block]
    chosen=kernel.signs[:BLOCK_ROWS]
    expected=np.sum((chosen[:,:,None]*v[None,None,:]).reshape(
        BLOCK_ROWS,HIDDEN)*x[None,:],axis=1)
    # Test the C wrapper using full output rows with the true on-disk bytes.
    base=np.zeros((1,1,VOCAB),dtype=np.float32)
    real=kernel.add_head(torch.from_numpy(x[None,None,:]),torch.from_numpy(base))
    observed=real.numpy().reshape(-1)[:BLOCK_ROWS]
    if not np.allclose(expected,observed,rtol=1e-4,atol=2e-5):
        raise AssertionError("1-bit residual native C differs from dense diagnostic")
    idx=torch.tensor([[0,1,15,16]],dtype=torch.long)
    w=torch.zeros((1,4,HIDDEN),dtype=torch.float32)
    y=kernel.add_embedding(idx,w).numpy().reshape(4,GROUPS,GROUP_WIDTH)
    exp=kernel.signs[[0,1,15,16],:,None]*kernel.direction[
        np.array([0,1,15,16])//BLOCK_ROWS,None,:]
    if not np.allclose(y,exp,rtol=0,atol=2e-6):
        raise AssertionError("Embedded one-bit row decode incorrect")
    return {"C_residual_head_matches_explicit_dense_reference":True,
            "packed_embedding_row_decode_matches_stored_direction":True,
            "dense_reference_TEST_ONLY_not_runtime_weights":True}


def run(v206_source,v161_source,v193_fp32qa,shared_library,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(216)
    raw=Path(v206_source).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_V206_SHA256 or len(raw)!=p.SOURCE_BYTES:
        raise AssertionError("Wrong original 135M v206 compressed source")
    archive=CB59(raw)
    original=original_teacher().cpu().eval()
    original_weight=original.get_input_embeddings().weight.detach().cpu().numpy()
    if original.get_input_embeddings().weight.data_ptr()!=original.get_output_embeddings().weight.data_ptr():
        raise AssertionError("Original pretrained head not tied")
    weights,source_stats=build_residual(original_weight,archive)
    del original
    destination=out/"GREEN_TUNNEL_v216_onebit_16row_vocab_tied_fp16.safetensors"
    save_file(weights,str(destination))
    actually_paid=destination.stat().st_size
    complete_size=len(raw)+actually_paid
    if not complete_size<=MAX_SEALED_WEIGHT_BYTES:
        raise AssertionError("One-bit Green Tunnel exceeds complete physical <=0.700 source")
    physical_bpp=8*complete_size/p.FROZEN_SOURCE_135M_UNIQUE
    seal={"version":"v216_135M_pretrained_source_only_onebit_tied_embedding_residual",
          "source_SHA256":SOURCE_V206_SHA256,
          "original_unique_parameters":p.FROZEN_SOURCE_135M_UNIQUE,
          "signed_CB59_source_bytes":len(raw),
          "signed_onebit_sidecar_bytes":actually_paid,
          "signed_onebit_sidecar_SHA256":hashlib.sha256(destination.read_bytes()).hexdigest(),
          "whole_source_plus_onebit_bpp":physical_bpp,
          "strict_complete_0p700_budget_pass":True,
          "original_tied_embedding_head_corrected_on_both_sides":True,
          "no_extra_dense_trained_weight_matrix_stored":True,
          "all_211_internal_original_quantized_matrices_unchanged":True,
          "source_only_fit_all_weights_no_QA_training":True,
          "source_fit":source_stats,
          "all_CORRECTIONS_sha_sealed_before_ANY_public_DEV":True,
          "no_Jessica_tests_or_live_changes":True}
    save(out/"SEALED_v216_paid_onebit_BEFORE_QA.json",seal)
    print(json.dumps({"stage":"v216_paid_onebit_sealed",
                      "bpp":physical_bpp,
                      "extra_bytes":actually_paid,
                      "embedding_SSE_after_over_before":
                      source_stats["embedding_source_SSE_ratio_after_to_before"]}),flush=True)
    onebit=NativeOneBit(destination,shared_library)
    integrity=numerical_test(onebit)
    save(out/"v216_native_C_correctness_vs_explicit_onebit.json",integrity)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    control=extract_original(Path(v161_source))
    if len(control.blob)!=p.SOURCE_BYTES:
        raise AssertionError("Missing physical CB59 CPU-speed original")
    with TemporaryDirectory(dir=out) as td:
        tmp=Path(td)
        for name in ("CB59","CB206","GT216"):(tmp/name).mkdir()
        (tmp/"CB59"/"source.cb59").write_bytes(control.blob)
        cb59,_,_=make_model("baseline",SimpleNamespace(
            original=tmp/"CB59"/"source.cb59",folder=tmp/"CB59"))
        cb206,_,_=make_model("baseline",SimpleNamespace(
            original=Path(v206_source),folder=tmp/"CB206"))
        cb216,_,_=make_model("baseline",SimpleNamespace(
            original=Path(v206_source),folder=tmp/"GT216"))
        if any(param.ndim>=2 for param in cb216.parameters()):
            raise AssertionError("Unpaid runtime source 2D weights")
        hooks=onebit.install(cb216)
        try:
            dev=newly_frozen_public_dev_16({"CB206":cb206,"GT216":cb216},
                                            tok,v193_fp32qa,out)
            if dev["scores"]["CB206"]["machine_confirmed"]!=3:
                raise AssertionError("Baseline source or original QA changed")
            speed=speed_pairs({"CB59":cb59,"CB186":cb216},tok,
                              freeze_cases(all_cases()))
        finally:
            for hook in hooks:hook.remove()
    result={"version":"v216_native_C_packed_1bit_residual_135M_0p7bpp",
            "completed":True,"physical_total_bpp":physical_bpp,
            "FP32_correct_public_DEV16":dev["scores"]["FP32_existing_135M"]["machine_confirmed"],
            "v206_correct_public_DEV16":dev["scores"]["CB206"]["machine_confirmed"],
            "v216_correct_public_DEV16":dev["scores"]["GT216"]["machine_confirmed"],
            "v216_machine_wrong":dev["scores"]["GT216"]["machine_recognized_wrong"],
            "v216_ungraded":dev["scores"]["GT216"]["ungraded_not_known_wrong"],
            "native_32token_CPU_speed_ratio_CB59":speed["median_speed_ratio_vs_CB59"],
            "native_C_residual_kernel_self_verified":integrity,
            "source_weight_reconstruction_SSE_ratio":source_stats[
                "embedding_source_SSE_ratio_after_to_before"],
            "no_PUBLIC_DEV_hyperparameter_selection":True,
            "PUBLIC_DEV16_previously_opened_not_independent_final_TEST":True,
            "NO_LIVE_MAIN_RENDER_SUPABASE_APK":True}
    save(out/"v216_REAL_native_1bit_allin_0p700_fullQA_CPU.json",result)
    print(json.dumps({"stage":"v216_native_onebit_QA",**result}),flush=True)
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("real_v206_source",type=Path)
    ap.add_argument("original_CB59_source",type=Path)
    ap.add_argument("original_v193_FP32_QA",type=Path)
    ap.add_argument("compiled_1bit_C_kernel",type=Path)
    ap.add_argument("output",type=Path)
    a=ap.parse_args()
    run(a.real_v206_source,a.original_CB59_source,a.original_v193_FP32_QA,
        a.compiled_1bit_C_kernel,a.output)
