"""CALIBRI 2.0 v001: FROM SCRATCH, genuinely complete 135M CB59 language model.

Initialization: AutoConfig + from_config (RANDOM), never from_pretrained model.
Pretrained tokenizer supplies tokenization ONLY; no source model weights,
source hidden activations, teacher logits, v228/v231 source archive.
Training is only WikiText TRAIN plus original synthetic TRAIN yes/no pairs.
All weight edits are already-paid CB59 5-bit group codes + FP16 K32V8
codebooks, with direct native C test. No QA-based checkpoint selection.
This is a CPU construction+short learning SMOKE, not pretrained-FP32 parity.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from direct_codebook_v59 import CB59,encode,unpack5
from direct_runtime_v59 import compile_kernel,NativeKernel
from quality import DATASET,CONFIG
from v174.run import save
from v176.operators import clear
from v185.codec import replace_discrete
from v185.optim import propose_codes,apply_codes
from v186.fit import paired_loss
from v186.pairs import build_pairs
from v196.recover import build_trainable_from_paid_source
from v200.rollout import probe_greedy_train,summarize_sequences

SEED=2020
MAX_BPP=0.700
STEPS_MAX=8
TRAIN_CONTEXT=32
LEARNING_RATE=0.012
CODE_UPDATE_CAP=512
BASELINE_PRETRAINED_SOURCE_SHA="0db01518534fb61e57a903f1c5155233993664ca34c4e7f9010fd9a1c8901483"
# Fully specified before seeing any predictions. All 211 codebooks get
# end-to-end gradients; 5-bit code proposals target four representative
# ORIGINAL architectural operators in alternating transformer blocks.
CODE_TARGETS=(
  "model.layers.0.self_attn.q_proj.weight",
  "model.layers.0.mlp.down_proj.weight",
  "model.layers.12.self_attn.o_proj.weight",
  "model.layers.12.mlp.up_proj.weight",
  "model.layers.20.self_attn.q_proj.weight",
  "model.layers.20.mlp.down_proj.weight",
  "model.layers.29.self_attn.o_proj.weight",
  "model.layers.29.mlp.up_proj.weight",
)

def digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def log(**fields):
    print(json.dumps(fields,ensure_ascii=False,default=str),flush=True)

def no_pretrained_teacher_contract():
    # Construction accesses model config and tokenizer by pinned revision,
    # but NEVER pretrained full-precision model weights.
    cfg=AutoConfig.from_pretrained(MODEL,revision=MODEL_SHA)
    cfg._attn_implementation="eager"
    return cfg

def check_physical(archive,expect_equal_size=None):
    if archive.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise AssertionError("Full 135M random model unique parameter count drift")
    if archive.hash()==BASELINE_PRETRAINED_SOURCE_SHA:
        raise AssertionError("Untrained model is identical to original FP32-derived weights")
    if len(archive.blob) > int(EXPECTED_UNIQUE*MAX_BPP/8):
        raise AssertionError("Complete random model exceeds paid 0.700 bits")
    if expect_equal_size is not None and archive.physical_bytes()!=expect_equal_size:
        raise AssertionError("Physical 135M source file changed its size")
    descriptors=archive.meta["tensor_descriptors"]
    paid=[name for name,d in descriptors.items() if d["dtype"]=="CB59_K32V8"]
    if len(paid)!=211:
        raise AssertionError(f"Not 211 original matrices: {len(paid)}")
    if archive.handles["lm_head.weight"] is not archive.handles["model.embed_tokens.weight"]:
        raise AssertionError("Tied embedding and head not physically shared")
    if len(archive.meta["full_original_state_keys"])<=len(paid):
        raise AssertionError("Norm / 1D source tensors omitted")
    return {"real_complete_paid_bytes":archive.physical_bytes(),
        "full_model_unique_params":EXPECTED_UNIQUE,
        "complete_model_bpp":8*archive.physical_bytes()/EXPECTED_UNIQUE,
        "SHA256":archive.hash(),
        "physical_5bit_8vector_matrices":len(paid),
        "tied_lm_head_and_embedding_shared":True,
        "all_original_raw_1D_and_source_metadata_counted":True,
        "no_extra_runtime_trainable_sidecar":True}

def native_direct_check(cb:CB59,out:Path):
    compiled=compile_kernel(
        Path(__file__).resolve().parents[2]/"direct_kernel_v59.c",
        out/"libcalibri2_v001_packed.so")
    kernel=NativeKernel(compiled)
    candidates=["model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.mlp.down_proj.weight"]
    checks=[]
    for name in candidates:
        h=cb.handles[name]
        m,n=h.shape
        ids=unpack5(h.packed.tobytes(),m*n//8)
        reconstruction=h.centroids[ids].reshape(m,n)*h.row_scales[:,None]
        v=np.linspace(-.75,.75,n,dtype=np.float32)[None,:]
        actual=kernel.linear(h,v)
        reference=v@reconstruction.T
        maximum=float(np.max(np.abs(actual-reference)))
        if not np.isfinite(maximum) or maximum>2e-3:
            raise AssertionError("Real native C output differs from CPU packed decoder")
        checks.append({"operator":name,"geometry":[m,n],
            "max_absolute_direct_kernel_error":maximum})
    h=cb.handles["model.embed_tokens.weight"]
    ids=np.array([1,2,13,119,299],dtype=np.int64)
    v=kernel.embedding(h,ids)
    all_codes=unpack5(h.packed.tobytes(),h.shape[0]*h.shape[1]//8)
    decoded=(h.centroids[all_codes.reshape(h.shape[0],-1)[ids]].reshape(len(ids),h.shape[1])*
             h.row_scales[ids,None])
    maximum=float(np.max(np.abs(v-decoded)))
    if maximum>2e-3:
        raise AssertionError("Native embedding does not honor direct 5bit source")
    return {"native_C_kernel_numeric_checks_passed":True,
            "packed_operators":checks,
            "embedding_max_absolute_error":maximum,
            "NOT_a_matched_v231_CPU_throughput_benchmark":True}

def initialize(out):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    cfg=no_pretrained_teacher_contract()
    started=time.perf_counter()
    fresh=AutoModelForCausalLM.from_config(
        cfg,torch_dtype=torch.float32).cpu().eval()
    unique=sum(int(p.numel()) for p in fresh.parameters())
    if unique!=EXPECTED_UNIQUE:
        raise AssertionError(f"Wrong identical pretrained architecture: {unique}")
    weights=fresh.state_dict()
    if not (weights["lm_head.weight"].data_ptr()==
            weights["model.embed_tokens.weight"].data_ptr()):
        raise AssertionError("Original tokenizer architecture weight tying drift")
    # True from-scratch random initialization. FP32 tensors exist ONLY here
    # for random initialization of original full state, not as pretrained
    # weights, not as inference weights, and are discarded before training.
    encoded,meta,distortion=encode(weights)
    del fresh,weights
    gc.collect()
    archive=CB59(encoded)
    physical=check_physical(archive)
    binary_path=out/"CALIBRI2_v001_RANDOM_INITIAL_135M.cb59"
    binary_path.write_bytes(encoded)
    if digest(binary_path.read_bytes())!=archive.hash():
        raise AssertionError("Failed physical random model serialization")
    physical["original_architecture_config_reference"]=MODEL
    physical["architecture_pinned_revision"]=MODEL_SHA
    physical["true_model_weights_pretrained"]=False
    physical["used_pretrained_model_config"]=True
    physical["used_pretrained_tokenizer_only"]=True
    physical["fresh_random_seed"]=SEED
    physical["initial_compression_RMSE_diagnostic"]=distortion["full_source_weighted_relative_RMSE"]
    physical["random_source_initialization_elapsed_seconds"]=round(time.perf_counter()-started,2)
    save(out/"CALIBRI2_v001_RANDOM_INITIAL_135M_manifest.json",physical)
    native=native_direct_check(archive,out)
    save(out/"CALIBRI2_v001_NATIVE_DIRECT_NUMERIC_SMOKE.json",native)
    log(stage="calibri2_v001_INITIAL_FROM_SCRATCH",**physical)
    log(stage="calibri2_v001_NATIVE_C_VALID",**native)
    return binary_path,physical

def freeze_train_text(tokenizer):
    # Freeze the training text and exact balanced instructional example
    # hashes BEFORE reading the random model. NEVER read QA or v231 labels.
    revision=HfApi().dataset_info(DATASET).sha
    ds=load_dataset(DATASET,CONFIG,split="train[:120]",
                    revision=revision,trust_remote_code=False)
    raw="\n\n".join(x["text"].strip() for x in ds if x["text"].strip())
    ids=tokenizer.encode(raw,add_special_tokens=False)
    if len(ids)<TRAIN_CONTEXT*8:
        raise AssertionError("Public TRAIN text too short")
    sample=torch.tensor(ids[:TRAIN_CONTEXT*8],dtype=torch.long)
    train_pairs=build_pairs(tokenizer,"TRAIN")
    if len(train_pairs)!=24:
        raise AssertionError("Old synthetic TRAIN pair generation was modified")
    return sample,train_pairs,{
        "dataset":DATASET,"config":CONFIG,"split":"train[:120]",
        "dataset_revision":revision,
        "first_256_train_token_ids_sha256":digest(sample.numpy().tobytes()),
        "TRAIN_balanced_opposing_pair_ids":[0,6],
        "train_example_question_hashes":[
            digest(p[s]["question"].encode("utf8"))
            for p in (train_pairs[0],train_pairs[6])
            for s in ("left","right")],
        "data_frozen_before_random_model_decoding":True,
        "no_teacher_pretrained_model_weights_or_labels_used":True,
        "no_QA_DEV_VALID_TEST_used":True}

def sample_for_step(step,train_tokens,train_pairs):
    if step%2==0:
        offset=(step//2)*TRAIN_CONTEXT
        seq=train_tokens[offset:offset+TRAIN_CONTEXT]
        if len(seq)!=TRAIN_CONTEXT:
            raise AssertionError("TRAIN token window drift")
        return ("wiki_train",seq,None)
    pair=train_pairs[0 if step%4==1 else 6]
    return ("synthetic_opposing_TRAIN",None,pair)

def task_loss(model,kind,token_ids,pair):
    if kind=="wiki_train":
        output=model(input_ids=token_ids[None],use_cache=False).logits
        loss=F.cross_entropy(output[0,:-1],token_ids[1:])
        return loss,{"teacher_forcing":"public wikitext TRAIN tokens",
            "sample_length":len(token_ids)}
    if kind=="synthetic_opposing_TRAIN":
        loss,details=paired_loss(model,None,pair)
        return loss,{"teacher_forcing":"both contradictory supervised TRAIN texts",
            "pair_id":pair["pair_id"],"domain":pair["domain"],
            "paired_margin":details["opposite_answer_margin_loss"]}
    raise AssertionError("Unknown purely-TRAIN task")

def trained_manifest(before,after,probes_before,probes_after,curve):
    x=summarize_sequences(probes_before)
    y=summarize_sequences(probes_after)
    if len(x["exact"])!=len(y["exact"]):
        pass  # Changes may be both positive and negative. Honest diagnostic.
    return {"from_scratch_training_completed":True,
        "all_211_paid_codebooks_received_gradient_each_step":True,
        "true_new_full_model_random_weights":True,
        "pretrained_model_weights_or_teacher_logits_seen":False,
        "stored_full_packed_archive_bpp":after["complete_model_bpp"],
        "stored_source_hash_changed":after["SHA256"]!=before["SHA256"],
        "QA_held_out_not_used_for_training_or_model_choice":True,
        "TRAIN_only_paired_answer_diagnostics":{
            "before_exact":len(x["exact"]),"after_exact":len(y["exact"]),
            "before_correct_prefix_tokens":x["prefix_tokens"],
            "after_correct_prefix_tokens":y["prefix_tokens"],
            "before_adjacent_repetition_excess":x["repetition_excess"],
            "after_adjacent_repetition_excess":y["repetition_excess"]},
        "train_steps":curve,
        "no_claim_to_FP32_answer_quality":True,
        "native_CPU_at_least_90pct_v231_NOT_YET_BENCHMARKED":True}

def training(out,original,steps):
    if steps<1 or steps>STEPS_MAX:raise ValueError("Bounded infrastructure pilot only")
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    train_tokens,train_pairs,train_data=freeze_train_text(tok)
    save(out/"CALIBRI2_v001_PUBLIC_TRAIN_precommitted_inputs.json",train_data)
    raw=Path(original).read_bytes()
    archive=CB59(raw)
    if archive.hash()!=digest(raw):
        raise AssertionError("Corrupt independently constructed initial weights")
    before=check_physical(archive)
    if before["SHA256"]!=json.loads(
        (out/"CALIBRI2_v001_RANDOM_INITIAL_135M_manifest.json").read_text())["SHA256"]:
        raise AssertionError("Full training source is not independently random CALIBRI 2.0 init")
    del raw
    model,sources=build_trainable_from_paid_source(archive)
    model.cpu().eval()
    if len(sources)!=211:
        raise AssertionError("Not all original 211 paid layers trainable")
    required={id(v.centroids) for v in sources.values()}
    actual={id(x) for x in model.parameters() if x.requires_grad}
    if required!=actual:
        raise AssertionError("Unaccounted runtime trainable original/dense weights")
    probe_pairs=[train_pairs[i] for i in (0,1,6,7)]
    before_answers=probe_greedy_train(model,sources,probe_pairs,max_new_tokens=6)
    save(out/"CALIBRI2_v001_8_synthetic_TRAIN_before.json",before_answers)
    optimizer=torch.optim.Adam([v.centroids for v in sources.values()],
                                lr=LEARNING_RATE)
    curve=[]
    for i in range(steps):
        target_name=CODE_TARGETS[i]
        target=sources[target_name]
        tag,ids,pair=sample_for_step(i,train_tokens,train_pairs)
        kept=[]
        def catch_grad(_mod,_inputs,reconstruction):
            if not any(t is reconstruction for t in kept):
                reconstruction.retain_grad()
                kept.append(reconstruction)
        handle=target.register_forward_hook(catch_grad)
        optimizer.zero_grad(set_to_none=True)
        clear(sources)
        try:
            loss,descriptor=task_loss(model,tag,ids,pair)
            if not torch.isfinite(loss):
                raise AssertionError("Nonfinite scratch model TRAIN objective")
            loss_value=float(loss.detach())
            loss.backward()
        finally:
            handle.remove()
        if not kept or any(x.grad is None for x in kept):
            raise AssertionError("No real end-to-end matrix gradient "+target_name)
        dense_grad=sum(x.grad.detach().cpu() for x in kept).numpy().copy()
        active=sum(s.centroids.grad is not None for s in sources.values())
        if active!=211:
            raise AssertionError(f"Only {active}/211 paid codebooks got gradient")
        grad_norm=float(torch.nn.utils.clip_grad_norm_(
            [v.centroids for v in sources.values()],1.0,
            error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for s in sources.values():s.centroids.clamp_(-16,16)
        clear(sources)
        with torch.inference_mode():
            after_centroid,details=task_loss(model,tag,ids,pair)
            baseline=float(after_centroid)
            current_weights=target().detach().cpu().numpy().copy()
        # v185 projected 5bit code optimizer; reference is the model's
        # OWN just-updated random-train weights, never the original FP32.
        candidate=propose_codes(target,dense_grad,current_weights)
        if candidate["proposed_groups"]>CODE_UPDATE_CAP:
            raise AssertionError("Out-of-budget indexed changes")
        accepted=False
        candidate_value=baseline
        if candidate["proposed_groups"]:
            with torch.no_grad():apply_codes(target,candidate,True)
            clear(sources)
            with torch.inference_mode():
                loss_next,_=task_loss(model,tag,ids,pair)
                candidate_value=float(loss_next)
            accepted=np.isfinite(candidate_value) and candidate_value<baseline-1e-7
            if not accepted:
                with torch.no_grad():apply_codes(target,candidate,False)
        clear(sources)
        item={"step":i+1,"domain":tag,
            "TRAIN_example":descriptor,"selected_paid_original_operator":target_name,
            "actual_all_paid_centroid_grads":active,"grad_norm_before_clip":grad_norm,
            "TRAIN_loss_before":loss_value,
            "TRAIN_loss_after_centroids":baseline,
            "TRAIN_loss_after_candidate_codes":candidate_value,
            "candidate_5bit_vector_reassignments":candidate["proposed_groups"],
            "accepted_5bit_vector_reassignments": (
              candidate["proposed_groups"] if accepted else 0),
            "accepted_only_for_TRAIN_improvement":accepted,
            "no_original_pretrained_FP32_weights_used":True}
        curve.append(item)
        log(stage="calibri2_v001_FROM_SCRATCH_learning_step",**item)
    after_answers=probe_greedy_train(model,sources,probe_pairs,max_new_tokens=6)
    save(out/"CALIBRI2_v001_8_synthetic_TRAIN_after.json",after_answers)
    packed,change=replace_discrete(archive,sources,strict=False)
    saved=out/"CALIBRI2_v001_AFTER_SCRATCH_TRAIN_135M.cb59"
    saved.write_bytes(packed)
    reread=CB59(saved.read_bytes())
    after=check_physical(reread,before["real_complete_paid_bytes"])
    if change["sha256"]!=after["SHA256"]:
        raise AssertionError("Sparse training physical export changed SHA")
    if not (change["actual_changed_5bit_group_indices"]>0 or
            change["actually_changed_FP16_codebook_cells"]>0):
        raise AssertionError("Fake training: no physical 5bit/FP16 values changed")
    report=trained_manifest(before,after,before_answers,after_answers,curve)
    report["initial"]=before
    report["trained"]=after
    report["actual_physical_edit_summary"]=change
    report["training_data"]=train_data
    save(out/"ACTUAL_CALIBRI2_v001_FROM_SCRATCH_REAL_TRAIN.json",report)
    log(stage="CALIBRI2_v001_FINAL_PILOT",
        physical_bpp=after["complete_model_bpp"],
        edited_5bit_vectors=change["actual_changed_5bit_group_indices"],
        edited_16bit_centroid_values=change["actually_changed_FP16_codebook_cells"],
        true_random_initial_source=before["SHA256"],
        trained_source=after["SHA256"],
        real_8_synthetic_TRAIN_correct_before=report["TRAIN_only_paired_answer_diagnostics"]["before_exact"],
        real_8_synthetic_TRAIN_correct_after=report["TRAIN_only_paired_answer_diagnostics"]["after_exact"],
        not_pretrained_FP32_parity=True)

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("out",type=Path)
    arg.add_argument("--steps",type=int,default=4)
    args=arg.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    original,physical=initialize(args.out)
    training(args.out,original,args.steps)

if __name__=="__main__":
    main()
