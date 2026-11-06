"""CALIBRI v213: universal tied-embedding Green Tunnel residual, rank 5 / 96.

The *original* 135M v206 packed source remains immutable. Fit a universal
low-rank residual of its tied input embedding/output head against FP32
original weights. The rank-5 residual + packed CB59 archive is hard-capped
at <= 0.700 physical bits/original parameter, while rank-96 is an explicitly
separate <=10 MB experimental Green Tunnel sidecar (does NOT fit the
inclusive 0.700 bpp budget). Both are frozen BEFORE any VALID/DEV QA.

Use exact actual native C-backed CB59 model plus hook-implemented residual
at runtime. Compare real native 16 full answers, CPU throughput, and physical
bytes, NEVER training text or QA answers to choose ranks, corrections,
codebooks, or checkpoints. No LIVE/Render/Supabase/Jessica interaction.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v174.intervene import extract_original
from v174.protocol import freeze_cases
from v174.run import save
from v169.sources import all_cases
from v175.evaluate import generate
from v176.operators import clear
from v186.run import speed_pairs
from v196 import protocol as p
from v196.run import newly_frozen_public_dev_16
from v196.recover import build_trainable_from_paid_source
from v212.run import SOURCE_V206_SHA256

# Ranks are fixed by the exact physical budget, NOT by QA/DEV outcomes.
RANKS = (5, 96)
MAX_EXTRA_EXPERIMENT_BYTES = 10_000_000
MAX_ALL_INCLUSIVE_BPP = 0.700


def make_factors(v206_bytes, fp32_teacher):
    """Unsupervised FP32 weight-residual reconstruction; no prompt/answer data."""
    student, sources=build_trainable_from_paid_source(CB59(v206_bytes))
    try:
        with torch.no_grad():
            decoded = student.model.embed_tokens.source().detach().cpu().float()
            original = fp32_teacher.get_input_embeddings().weight.detach().cpu().float()
            if decoded.shape != original.shape or original.ndim != 2:
                raise AssertionError("Head and embedding not exactly tied and matching")
            residual = (original - decoded).contiguous()
            if not torch.isfinite(residual).all():
                raise AssertionError("Nonfinite original FP32 compressed residual")
            torch.manual_seed(213)
            # A single fixed rank-96 randomized SVD, then rank-5 truncation.
            # No QA-based choice and never a trained Vocab/DEV lookup table.
            U,S,V = torch.svd_lowrank(residual,q=104,niter=2)
            A=(U[:,:96] * S[:96][None,:]).contiguous()
            B=(V[:,:96].T).contiguous()
            A=A.to(torch.float16).cpu().contiguous()
            B=B.to(torch.float16).cpu().contiguous()
            if not (torch.isfinite(A).all() and torch.isfinite(B).all()):
                raise AssertionError("FP16 Green Tunnel overflow")
            return {rank:{"A":A[:,:rank].contiguous(),"B":B[:rank,:].contiguous()}
                    for rank in RANKS}
    finally:
        clear(sources)


def install_head_and_embedding_tunnel(native, factors):
    """Native packed CB59 inference + small lowrank matrix multiplies.

    Return hooks. Green tunnel is NOT hidden in an uncounted dense matrix:
    the only added parameters are rank matrices stored in the sealed file.
    """
    a=factors["A"].to(torch.float32)
    b=factors["B"].to(torch.float32)
    if not (a.ndim==2 and b.ndim==2 and a.shape[1]==b.shape[0] and
            a.shape[0]==native.config.vocab_size and
            b.shape[1]==native.config.hidden_size):
        raise AssertionError("Unmatched Green Tunnel original model geometry")
    def embedding_correction(module,args,out):
        token_ids=args[0]
        added=F.embedding(token_ids,a) @ b
        if added.shape!=out.shape:
            raise AssertionError("Incorrect tied embedding restoration")
        return out+added.to(out.dtype)
    def head_correction(module,args,out):
        hidden=args[0]
        added=(hidden.float()@b.T)@a.T
        if added.shape!=out.shape:
            raise AssertionError("Incorrect LM head restoration")
        return out+added.to(out.dtype)
    hooks=[native.get_input_embeddings().register_forward_hook(embedding_correction),
           native.get_output_embeddings().register_forward_hook(head_correction)]
    return hooks


def experiment(v206_path,v161_path,v193_path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(213)
    compressed=Path(v206_path).read_bytes()
    if hashlib.sha256(compressed).hexdigest()!=SOURCE_V206_SHA256:
        raise AssertionError("Not the immutable v206 3/16 signed source")
    if len(compressed)!=p.SOURCE_BYTES:
        raise AssertionError("Physical v206 full source bytes mismatch")
    teacher=original_teacher().cpu().eval()
    if teacher.get_input_embeddings().weight.data_ptr()!=teacher.get_output_embeddings().weight.data_ptr():
        raise AssertionError("Teacher original embedding/head not actually tied")
    factors=make_factors(compressed,teacher)
    del teacher
    manifest={
        "experiment":"v213_original_FP32_weight_residual_not_QA_training",
        "physically_unchanged_135M_v206_SHA256":SOURCE_V206_SHA256,
        "source_original_unique_parameters":p.FROZEN_SOURCE_135M_UNIQUE,
        "physical_compressed_CB59_bytes":len(compressed),
        "physical_compressed_CB59_bpp":8*len(compressed)/p.FROZEN_SOURCE_135M_UNIQUE,
        "rank_precommitted_BEFORE_DEV_OR_VALID":list(RANKS),
        "how_built":"fp32_original_minus_signed_v206_tied_embedding_head_randomized_lowrank_SVD",
        "random_seed":213,"svd_oversampling":8,"power_iterations":2,
        "NO_TRAIN_QA_VALID_DEV_used_to_fit_or_select_sidecar":True,
        "NO_LIVE_Jessica_Render_Supabase_APK":True,
        "all_factors_physically_stored_FP16":True,
        "no_unpaid_dense_residual_matrix_exported":True,
        "ranked_experiments":{}
    }
    for rank in RANKS:
        path=out/f"GREEN_TUNNEL_v213_tied_head_rank{rank}.safetensors"
        save_file(factors[rank],str(path))
        expected=2*rank*sum(factors[rank][k].shape[i] for k,i in (("A",0),("B",1)))
        # Sanity: the tensor bytes must be accounted for in actual disk file.
        actual_bytes=path.stat().st_size
        if not (expected<=actual_bytes<=expected+4096):
            raise AssertionError("Green Tunnel sidecar file has hidden overhead")
        if rank==96 and actual_bytes>MAX_EXTRA_EXPERIMENT_BYTES:
            raise AssertionError("Rank-96 experimental sidecar >10MB")
        inclusive=8*(len(compressed)+actual_bytes)/p.FROZEN_SOURCE_135M_UNIQUE
        if rank==5 and inclusive>MAX_ALL_INCLUSIVE_BPP:
            raise AssertionError("Rank-5 source+sidecar violates inclusive 0.700 bpp")
        manifest["ranked_experiments"][str(rank)]={
            "sidecar_file":path.name,
            "sidecar_SHA256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "sidecar_bytes":actual_bytes,
            "physical_paid_CB59_source_bytes":len(compressed),
            "all_inclusive_bpp":inclusive,
            "sidecar_is_experimental_unpaid_if_rank96":rank==96,
            "fits_all_inclusive_0p700_budget":inclusive<=MAX_ALL_INCLUSIVE_BPP
        }
    save(out/"SEALED_v213_sidecars_BEFORE_ALL_DEV_QA.json",manifest)
    print(json.dumps({"stage":"v213_sidecars_sealed_before_QA",
        "ranks":manifest["ranked_experiments"]}),flush=True)

    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    original=extract_original(Path(v161_path))
    if len(original.blob)!=p.SOURCE_BYTES:
        raise AssertionError("Missing native CB59 baseline")
    with TemporaryDirectory(dir=out) as td:
        root=Path(td)
        for name in ("CB59","CB206","GT213_r5","GT213_r96"):
            (root/name).mkdir()
        (root/"CB59"/"source.cb59").write_bytes(original.blob)
        (root/"CB206"/"source.cb59").write_bytes(compressed)
        cb59,_,_=make_model("baseline",SimpleNamespace(original=root/"CB59"/"source.cb59",folder=root/"CB59"))
        cb206,_,_=make_model("baseline",SimpleNamespace(original=root/"CB206"/"source.cb59",folder=root/"CB206"))
        native={"CB206":cb206}
        handles={}
        for rank in RANKS:
            name=f"GT213_r{rank}"
            model,_,_=make_model("baseline",SimpleNamespace(
                original=Path(v206_path),folder=root/name))
            if any(x.ndim>=2 for x in model.parameters()):
                raise AssertionError("Not actual packed native CB59 runtime")
            loaded=load_file(str(out/f"GREEN_TUNNEL_v213_tied_head_rank{rank}.safetensors"))
            handles[name]=install_head_and_embedding_tunnel(model,loaded)
            native[name]=model
        try:
            qa=newly_frozen_public_dev_16(native,tokenizer,v193_path,out)
            if qa["scores"]["CB206"]["machine_confirmed"]!=3:
                raise AssertionError("v206 baseline native QA changed")
            frozen_dev=freeze_cases(all_cases())
            speed={}
            for rank in RANKS:
                name=f"GT213_r{rank}"
                bench=speed_pairs({"CB59":cb59,"CB186":native[name]},tokenizer,frozen_dev)
                bench["originally_expected_fixed_key"]="CB186"
                bench["actual_source"]=name
                speed[name]=bench
            save(out/"real_native_32_token_CPU_vs_CB59_by_Green_Tunnel_rank.json",speed)
        finally:
            for hlist in handles.values():
                for h in hlist:h.remove()
    result={
        "experiment":"v213_lowrank_embedding_head_green_tunnel",
        "completed":True,
        "sealing_manifest":manifest,
        "scores":{k:{
            "machine_confirmed":v["machine_confirmed"],
            "machine_wrong":v["machine_recognized_wrong"],
            "ungraded":v["ungraded_not_known_wrong"],
            "per_domain":v["domain_totals"]
            } for k,v in qa["scores"].items()},
        "speed_native_vs_CB59":{name:bench["median_speed_ratio_vs_CB59"] for name,bench in speed.items()},
        "public_DEV_only_not_checkpoint_selection":True,
        "no_unpaid_FP32_full_weights_in_inference":True,
        "NO_LIVE_CHANGES":True
    }
    save(out/"v213_REAL_NATIVE_GREEN_TUNNEL_rank5_rank96_QA_and_CPU.json",result)
    print(json.dumps({"stage":"v213_real_native_results",
       "scores":result["scores"],
       "speed":result["speed_native_vs_CB59"],
       "inclusive_bpp":{k:v["all_inclusive_bpp"] for k,v in manifest["ranked_experiments"].items()}
       }),flush=True)
    return result


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("v206",type=Path)
    ap.add_argument("original_v161",type=Path)
    ap.add_argument("original_v193_fp32_96qa",type=Path)
    ap.add_argument("output",type=Path)
    x=ap.parse_args()
    experiment(x.v206,x.original_v161,x.original_v193_fp32_96qa,x.output)
