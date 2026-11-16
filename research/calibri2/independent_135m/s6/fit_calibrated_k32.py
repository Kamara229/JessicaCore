#!/usr/bin/env python3
"""S6: original trained 135M FP32 activation-aware CB59 211-matrix refit.

No NRA, no EVAL or sealed QA. Trained FP32 activation moments exist in
TRAIN memory only. *Inference* always uses unchanged original CB59
5-bit/8weight native codec; 1D tensors/scales/aliases never change.
This is a new quantization objective, not teacher-guided answer lookup.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from direct_codebook_v59 import CB59, pack5, unpack5
from v185.codec import replace_discrete
from v196.recover import build_trainable_from_paid_source
from v176.operators import clear
from calibri2.independent_135m.s2.rootcause import (
    PROMPTS as ALREADY_SEEN_DEBUG, measure, prompt_ids,
)

S0_SHA = "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
SOURCE_BYTES = 11229303
SAMPLE = 8192
REFINE_STEPS = 8
DAMP = 0.15
ACTIVATION_CLIP = 6.0
BLK_ROWS = 128
# Frozen TRAIN-only calibration corpus, drawn from already disclosed S0/S1
# and S3 TRAIN examples. Excludes architecture C0 12 benchmark EVAL prompts.
TRAIN = (
    "Answer with one word: What color is fresh snow?",
    "Finish in one word: Water turns into ice when it ...",
    "Answer with one digit: How much is 2 plus 2?",
    "Answer with one word: What is the opposite of hot?",
    "Answer with one word: What do bees make?",
    "Answer with one word: What is the opposite of up?",
    "Complete briefly: The sun rises in the ...",
    "Answer with one word: What color is a ripe banana?",
    "Answer with one word only. What is the opposite of yes?",
    "Answer with one word only. What is the opposite of black?",
    "Answer with one word only. What color is most grass?",
    "Answer with one word only. What sound does a cat make?",
    "Finish this phrase: A bird can ...",
    "State one word for the opposite of dark.",
    "Complete: The ocean is usually ...",
    "Write the number obtained by adding five and five.",
    "Answer with one word: A young dog is a ...",
    "Continue in one word: Grass grows in the ...",
    "Finish in one word: It is cold in ...",
    "Provide a one-word synonym for quick.",
)


def sha(x: bytes) -> str:
    return hashlib.sha256(x).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     sort_keys=True) + "\n", encoding="utf-8")


def emit(stage, **value):
    print(json.dumps({"stage":stage, **value}, ensure_ascii=False), flush=True)


@torch.inference_mode()
def collect_activation_moments(teacher, tokenizer, names):
    """Real original FP32 inputs to ALL original 210 decoder matmuls + LM head."""
    captured = {}
    seen = {}
    handles = []
    for module_name, module in teacher.named_modules():
        if not isinstance(module, nn.Linear) or module.bias is not None:
            continue
        original_name = module_name + ".weight"
        key = ("model.embed_tokens.weight"
               if original_name == "lm_head.weight" else original_name)
        if key not in names:
            continue
        if key in captured:
            raise AssertionError("Original linear quantizer calibration duplicated " + key)
        captured[key] = torch.zeros(module.in_features, dtype=torch.float64)
        seen[key] = 0

        def hook(_module, args, *, identity=key):
            activation = args[0].detach().float()
            flatten = activation.reshape(-1, activation.shape[-1])
            captured[identity] += flatten.square().sum(0).double().cpu()
            seen[identity] += flatten.shape[0]
        handles.append(module.register_forward_pre_hook(hook))
    if set(captured) != set(names):
        missing = set(names) - set(captured)
        raise AssertionError("Source-only decoder activation capture missed genuine " +
                             repr(sorted(missing)[:12]))
    contexts = [prompt_ids(tokenizer, prompt) for prompt in TRAIN]
    try:
        for idx, context in enumerate(contexts):
            result = teacher(input_ids=context, use_cache=False)
            del result
            if (idx+1) % 5 == 0:
                emit("S6_original_FP32_TRAIN_activation_capture",
                     prompts=idx+1, matrices=len(captured))
    finally:
        for h in handles:
            h.remove()
    moments = {}
    for key, value in captured.items():
        if seen[key] == 0:
            raise AssertionError("Original actual pretrained FP32 matrix not visited: " + key)
        activation = (value / seen[key]).numpy().astype(np.float32)
        if np.any(~np.isfinite(activation)) or np.max(activation) <= 0:
            raise AssertionError("Invalid original FP32 train activation second moment")
        median = max(float(np.median(activation)), 1e-12)
        # Balanced and deterministic regularization of diagonal covariance:
        # avoid a handful of outliers deciding the complete source codec.
        ratio = np.sqrt(activation / median).astype(np.float32)
        normalized = (DAMP + (1.0-DAMP)*np.clip(
            ratio, 1.0/ACTIVATION_CLIP, ACTIVATION_CLIP)).astype(np.float32)
        normalized /= float(np.mean(normalized))
        moments[key] = normalized
    return moments, {name:{"TRAIN_activation_tokens":seen[name],
                             "feature_count":len(moments[name]),
                             "min_weight":float(moments[name].min()),
                             "max_weight":float(moments[name].max())}
                     for name in sorted(names)}


def choose_weighted(vectors, weights, centers):
    """Diagonal-covariance weighted 8D squared distance for each source group.

    Omit ||v||² term since independent of code choice; use fast 8D x 32
    matrix multiply, no 3D allocation across all original 135M groups.
    """
    x = (vectors.astype(np.float32,copy=False) *
         weights.astype(np.float32,copy=False))
    c = centers.astype(np.float32,copy=False)
    distance = (weights @ (c*c).T) - (2.0 * (x @ c.T))
    if not np.all(np.isfinite(distance)):
        raise AssertionError("Non-finite genuine activated-weight K32 assignments")
    return np.argmin(distance, axis=1).astype(np.uint8)


def activation_refit_matrix(original, handle, activation, *, samples=SAMPLE,
                            passes=REFINE_STEPS, blockrows=BLK_ROWS):
    """Fit same 32x8 FP16 centers + same 5-bit group ids at original scales."""
    w = np.asarray(original,dtype=np.float32)
    rows, ncol = w.shape
    if handle.shape != (rows,ncol) or ncol % 8 or len(activation) != ncol:
        raise AssertionError("Original pretrained codebook dimensions mutated")
    group_count = ncol // 8
    normalized_weights = np.asarray(activation,dtype=np.float32).reshape(-1,8)
    row_scales = handle.row_scales.astype(np.float32)
    if row_scales.shape != (rows,):
        raise AssertionError("Changed source baseline row scales")
    # A universal 8192 representative source-code vector sample, positioned
    # around the entire original matrix, not EVAL-selected.
    total = rows*group_count
    positions = np.linspace(0,total-1,min(samples,total),dtype=np.int64)
    row_idx = positions // group_count
    group_idx = positions % group_count
    col = group_idx[:,None]*8+np.arange(8)[None,:]
    sampled = (w[row_idx[:,None],col] /
               row_scales[row_idx,None]).astype("<f2").astype(np.float32)
    sampled_weight = normalized_weights[group_idx]
    # Start from the physically existing CB59 source. Refit objective
    # (teacher TRAIN diagonal activation), NOT same original L2-only kmeans.
    centroids = np.asarray(handle.centroids,dtype=np.float32).copy()
    if centroids.shape != (32,8):
        raise AssertionError("K32 V8 original decoder dictionary drift")
    for _ in range(passes):
        codes = choose_weighted(sampled,sampled_weight,centroids)
        # Weighted 32x8 centroid update via true selected source vectors.
        for dim in range(8):
            wt = sampled_weight[:,dim]
            denominator = np.bincount(codes,weights=wt,minlength=32)
            numerator = np.bincount(
                codes,weights=wt*sampled[:,dim],minlength=32)
            occupied = denominator > 1e-12
            centroids[occupied,dim] = (
                numerator[occupied]/denominator[occupied])
        centroids = centroids.astype("<f2").astype(np.float32)
        if not np.isfinite(centroids).all():
            raise AssertionError("FP16 paid centroid overflow")
    packed = []
    old_sse=0.;new_sse=0.
    old_activated=0.;new_activated=0.;source_energy=0.
    old_idx_all=unpack5(handle.packed.tobytes(),total)
    changed_groups=0
    for start in range(0,rows,blockrows):
        stop=min(rows,start+blockrows)
        v=w[start:stop].reshape(-1,8)
        row_group=np.repeat(row_scales[start:stop],group_count)
        vec=(v/row_group[:,None]).astype("<f2").astype(np.float32)
        wg=np.broadcast_to(normalized_weights[None,:,:],
                            (stop-start,group_count,8)).reshape(-1,8)
        idx=choose_weighted(vec,wg,centroids)
        packed.append(pack5(idx))
        old_codes=old_idx_all[start*group_count:stop*group_count]
        changed_groups+=int(np.count_nonzero(old_codes!=idx))
        old_reconstruct=(handle.centroids[old_codes].astype(np.float32)*
                         row_group[:,None])
        new_reconstruct=(centroids[idx]*row_group[:,None])
        old_diff=(v-old_reconstruct).astype(np.float64)
        new_diff=(v-new_reconstruct).astype(np.float64)
        old_sse+=float(np.sum(old_diff*old_diff))
        new_sse+=float(np.sum(new_diff*new_diff))
        old_activated+=float(np.sum(old_diff*old_diff*wg))
        new_activated+=float(np.sum(new_diff*new_diff*wg))
        source_energy+=float(np.sum(v.astype(np.float64)**2))
    raw=b"".join(packed)
    if len(raw)!=len(handle.packed) or not source_energy>0:
        raise AssertionError("Different original physical 5-bit matrix source")
    if changed_groups==0 and np.array_equal(handle.centroids,centroids):
        raise AssertionError("New original-FP32 activation-aware quantizer did nothing")
    return {
        "centroids":torch.from_numpy(centroids.astype("<f2").copy()),
        "codes":torch.from_numpy(unpack5(raw,total).astype(np.int64)),
        "stats":{
           "row_count":rows,"input_width":ncol,
           "5bit_code_groups_reassigned":changed_groups,
           "before_FP32_original_weight_SSE":old_sse,
           "after_FP32_original_weight_SSE":new_sse,
           "before_FP32_TRAIN_activation_weighted_SSE":old_activated,
           "after_FP32_TRAIN_activation_weighted_SSE":new_activated,
           "FP32_source_energy":source_energy,
           "weighted_SSE_after_over_before":new_activated/max(old_activated,1e-20),
           "plain_SSE_after_over_before":new_sse/max(old_sse,1e-20),
        }
    }


@torch.inference_mode()
def layer_cosine(fp32,student,source,ctx):
    a=fp32(input_ids=ctx,use_cache=False,output_hidden_states=True)
    clear(source)
    b=student(input_ids=ctx,use_cache=False,output_hidden_states=True)
    out={}
    if len(a.hidden_states)!=31 or len(b.hidden_states)!=31:
        raise AssertionError("Actual pretrained 30-layer outputs changed")
    for i in (0,1,10,20,30):
        q=a.hidden_states[i][0,-1].float()
        r=b.hidden_states[i][0,-1].float()
        out[str(i)]={
            "cosine_FP32":float(torch.nn.functional.cosine_similarity(
                q[None],r[None]).item()),
            "relative_l2":float(torch.linalg.vector_norm(q-r)/
                                torch.linalg.vector_norm(q).clamp_min(1e-10))}
    clear(source)
    return out


def run(S0_source,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(13561117)
    torch.set_num_threads(2)
    original_blob=Path(S0_source).read_bytes()
    if sha(original_blob)!=S0_SHA or len(original_blob)!=SOURCE_BYTES:
        raise AssertionError("Wrong original independently coded FP32→135M CB59 provenance")
    archive=CB59(original_blob)
    if archive.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise AssertionError("Unique original FP32 135M parameter geometry drift")
    names=sorted(name for name,descriptor in
                 archive.meta["tensor_descriptors"].items()
                 if descriptor["dtype"]=="CB59_K32V8")
    if len(names)!=211:
        raise AssertionError("Source must be exactly 211 original compressed matrices")
    plan={
       "stage":"S6_original_pretrained_FP32_TRAIN_activations_aware_codebooks",
       "source_SHA_before":S0_SHA,
       "original_pretrained_model_revision":MODEL_SHA,
       "original_unique_parameters":EXPECTED_UNIQUE,
       "actual_original_matmul_weights":len(names),
       "source_codebook_K":32,
       "source_group_size":8,
       "source_bits_per_group":5,
       "source_vector_samples_for_refit":SAMPLE,
       "centroid_weighted_update_iterations":REFINE_STEPS,
       "regularization_floor":DAMP,
       "activation_clip":ACTIVATION_CLIP,
       "calibration_questions":list(TRAIN),
       "calibration_prompt_SHA256":[sha(q.encode()) for q in TRAIN],
       "debug_only_previously_exposed_prompts":list(ALREADY_SEEN_DEBUG),
       "exclude_all_prior_eval_benchmark":True,
       "NRA_absent":True,
       "original_FP32_not_shipped_in_inference":True,
       "no_Jessica_chat_or_live":True,
       "new_QA_EVAL_used":False,
    }
    save(output/"S6_PRECOMMIT_BEFORE_TEACHER_LOAD.json",plan)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    fp32=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32).cpu().eval()
    if sum(x.numel() for x in fp32.parameters())!=EXPECTED_UNIQUE:
        raise AssertionError("Not original authentic pretrained 135M checkpoint")
    state=fp32.state_dict()
    # Real source-only activations (not labels/QA) are recalculated from the
    # original genuine learned 30-layer FP32 model, no student rollouts.
    activations,counts=collect_activation_moments(fp32,tok,set(names))
    save(output/"S6_ALL_211_ACTUAL_FP32_TRAIN_ACTIVATION_MOMENTS_SUMMARY.json",counts)
    new={}
    per_matrix={}
    for i,name in enumerate(names):
        ref=state[name].cpu().numpy()
        fitted=activation_refit_matrix(ref,archive.handles[name],
                                       activations[name])
        new[name]=SimpleNamespace(
            codes=fitted["codes"],centroids=fitted["centroids"])
        per_matrix[name]=fitted["stats"]
        if (i+1)%25==0:
            emit("S6_weighted_source_K32V8_matrix_refitted",
                 completed=i+1,total=len(names),
                 matrix=name,
                 activation_ratio=round(fitted["stats"][
                    "weighted_SSE_after_over_before"],5))
    candidate,seal=replace_discrete(archive,new,strict=False)
    if (len(candidate)!=SOURCE_BYTES or
        seal["all_2d_original_matrices"]!=211 or
        seal["complete_bpp"]>0.7):
        raise AssertionError("Source-only physical model does not fit original 0.7 bpp")
    new_archive=CB59(candidate)
    if (archive.meta!=new_archive.meta or
        new_archive.handles["model.embed_tokens.weight"] is not
        new_archive.handles["lm_head.weight"]):
        raise AssertionError("Invalid real pretrained original CB59 ABI")
    path=output/"S6_ORIGINAL_FP32_135M_ACTIVATION_AWARE_K32V8.cb59"
    path.write_bytes(candidate)
    error={
        "global_FP32_weight_SSE_before":sum(r["before_FP32_original_weight_SSE"] for r in per_matrix.values()),
        "global_FP32_weight_SSE_after":sum(r["after_FP32_original_weight_SSE"] for r in per_matrix.values()),
        "global_TRAIN_activation_SSE_before":sum(r["before_FP32_TRAIN_activation_weighted_SSE"] for r in per_matrix.values()),
        "global_TRAIN_activation_SSE_after":sum(r["after_FP32_TRAIN_activation_weighted_SSE"] for r in per_matrix.values()),
        "all_original_211_matrices":per_matrix,
    }
    error["global_TRAIN_activation_SSE_ratio"]=(error[
        "global_TRAIN_activation_SSE_after"]/error[
        "global_TRAIN_activation_SSE_before"])
    error["global_FP32_weight_SSE_ratio"]=(error[
        "global_FP32_weight_SSE_after"]/error[
        "global_FP32_weight_SSE_before"])
    save(output/"S6_FULL_ORIGINAL_FP32_ALL211_WEIGHT_AND_ACTIVATION_DISTORTION.json",error)
    emit("S6_authentic_physically_paid_original_135M_source",
         sha256=sha(candidate),physical_bytes=len(candidate),
         physical_bpp=seal["complete_bpp"],original_full211_matrices=211,
         changed_5bit_groups=seal["actual_changed_5bit_group_indices"],
         activation_weighted_error_ratio=round(
             error["global_TRAIN_activation_SSE_ratio"],6),
         original_weight_error_ratio=round(
             error["global_FP32_weight_SSE_ratio"],6))
    original_student,original_src=build_trainable_from_paid_source(archive)
    new_student,new_src=build_trainable_from_paid_source(new_archive)
    original_student.cpu().eval().requires_grad_(False)
    new_student.cpu().eval().requires_grad_(False)
    records=[]
    for i,prompt in enumerate(ALREADY_SEEN_DEBUG):
        ctx=prompt_ids(tok,prompt)
        teacher_answer=measure(fp32,None,tok,ctx)
        old_answer=measure(original_student,original_src,tok,ctx)
        new_answer=measure(new_student,new_src,tok,ctx)
        before_hidden=layer_cosine(fp32,original_student,original_src,ctx)
        after_hidden=layer_cosine(fp32,new_student,new_src,ctx)
        records.append({
            "TRAIN_DEBUG_question_id":i,
            "original_previously_exposed_question":prompt,
            "FP32_original_greedy":teacher_answer,
            "S0_CB59_original_greedy":old_answer,
            "S6_activation_K32V8_greedy":new_answer,
            "S0_original_hidden_cosine":before_hidden,
            "S6_original_hidden_cosine":after_hidden,
            "S0_FP32_first_token_match":bool(
                 teacher_answer["tokens"] and old_answer["tokens"] and
                 teacher_answer["tokens"][0]==old_answer["tokens"][0]),
            "S6_FP32_first_token_match":bool(
                 teacher_answer["tokens"] and new_answer["tokens"] and
                 teacher_answer["tokens"][0]==new_answer["tokens"][0]),
            "not_untouched_EVAL":True,
        })
        save(output/"S6_PRIOR_TRAIN_DEBUG_ACTUAL_GENERATIONS.json",records)
        emit("S6_original135M_act_aware_REAL_DEBUG",
             idx=i,FP32=teacher_answer["text"][:72],
             baseline_CB59=old_answer["text"][:72],
             new_ACTAWARE_CB59=new_answer["text"][:72],
             old_final_cos=round(before_hidden["30"]["cosine_FP32"],5),
             new_final_cos=round(after_hidden["30"]["cosine_FP32"],5),
             matching_FP32_first=records[-1]["S6_FP32_first_token_match"])
    answer={
       **plan,
       "execution_complete":True,
       "new_full_source_sha256":sha(candidate),
       "new_source_bytes":len(candidate),
       "all_inclusive_source_bpp":seal["complete_bpp"],
       "same_original_tensor_descriptors_and_aliases":True,
       "physically_paid_all_211_matrices":True,
       "all_original_FP32_1D_and_row_scales_unchanged":True,
       "source_5bit_changes":seal["actual_changed_5bit_group_indices"],
       "source_FP16_centroid_changes":seal["actually_changed_FP16_codebook_cells"],
       "FP32_reference_TRAIN_activation_loss":error,
       "debug_actual_generated_outputs":records,
       "same_FP32_first_token_original":sum(r["S0_FP32_first_token_match"] for r in records),
       "same_FP32_first_token_new":sum(r["S6_FP32_first_token_match"] for r in records),
       "EOS_generated_new":sum(r["S6_activation_K32V8_greedy"]["EOS_seen"]
                               for r in records),
       "new_out_of_distribution_EVAL_completed":False,
       "native_C_CPU_90pct_speed_test_completed":False,
       "FP32_language_parity_proven":False,
       "candidate_accepted":False,
    }
    save(output/"S6_ACTUAL_ORIGINAL_135M_ACTIVATION_AWARE_SOURCE_RESULT.json",answer)
    emit("S6_FINAL_REAL_FP32_calibrated_135M_SOURCE",
         original_activation_weighted_ratio=round(
             error["global_TRAIN_activation_SSE_ratio"],6),
         original_weight_only_ratio=round(error[
             "global_FP32_weight_SSE_ratio"],6),
         first_FP32_matches_baseline=answer["same_FP32_first_token_original"],
         first_FP32_matches_new=answer["same_FP32_first_token_new"],
         max_possible_debug_first_matches=len(ALREADY_SEEN_DEBUG),
         EOS_seen=answer["EOS_generated_new"],
         physical_bpp=seal["complete_bpp"],FP32_parity_proven=False)
    del fp32
    gc.collect()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("original_S0_pretrained_CB59",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.original_S0_pretrained_CB59,a.out)
