"""Frozen-source embedding-only dictionary recoding pilot at EXACT CB59 bit rate.

No QA prompts, no model outputs, no selected hidden TEST or training labels.
No packed model is modified: preview codebooks are discarded after scoring.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from direct_codebook_v59 import CB59,unpack5,_nearest_integer_128
from v117.protocol import ORIGINAL_CB59_SHA
from v128.protocol import read_v119
from v84.sources import original_teacher
from .protocol import FIT_GROUPS,VALID_GROUPS,ITERATIONS,SEED,select,groups,physical_budget
from .fit import lloyd,product_centers,score


def run(host,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    packed=read_v119(host)
    if hashlib.sha256(packed.base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original codebook archive SHA mismatch")
    archive=CB59(packed.base)
    handle=archive.handles["model.embed_tokens.weight"]
    if handle.shape!=(49152,576):
        raise ValueError("unexpected tied embedding dictionary shape")
    teacher=original_teacher()
    try:
        weight=teacher.get_input_embeddings().weight.detach().float().cpu().numpy().copy()
    finally:
        del teacher
    if weight.shape!=handle.shape or not np.isfinite(weight).all():
        raise ValueError("invalid pretrained source")
    total=weight.shape[0]*(weight.shape[1]//8)
    fit_indices,valid_indices=select(total)
    fit_actual,fit_normalized,fit_scale=groups(weight,handle.row_scales,fit_indices)
    valid_actual,valid_normalized,valid_scale=groups(weight,handle.row_scales,valid_indices)
    original_centroids=handle.centroids.astype(np.float32)
    original_codes=unpack5(handle.packed.tobytes(),total)
    if len(original_codes)!=total:
        raise AssertionError("incomplete physical encoded input embedding")
    exact_codes=original_codes[valid_indices]

    variants={
        "CB59_actual_packed_codes":original_centroids,
        "same_rate_uniform_Lloyd8":lloyd(fit_normalized,original_centroids,
                                         passes=ITERATIONS),
        "same_rate_weighted_Lloyd8":lloyd(fit_normalized,original_centroids,
                                          energy=np.square(fit_scale.astype(np.float64)),
                                          passes=ITERATIONS),
        "same_rate_product_8x4_V4":product_centers(fit_normalized,original_centroids,
                                                    passes=ITERATIONS),
    }
    scores={}
    for name,centroids in variants.items():
        res=score(valid_actual,valid_normalized,valid_scale,centroids,
                  source_codes=exact_codes if name=="CB59_actual_packed_codes" else None)
        res["codebook_sha256_fp16"]=hashlib.sha256(
            centroids.astype("<f2").tobytes()).hexdigest()
        scores[name]=res
    rerun=_nearest_integer_128(valid_normalized,original_centroids)
    verification=float(np.mean(rerun==exact_codes))
    result={
        "experiment":"v137 same 5bit/8 vector embedding source coding",
        "frozen_original_CB59_sha":ORIGINAL_CB59_SHA,
        "frozen_parent_v119_sha256":hashlib.sha256(host.read_bytes()).hexdigest(),
        "source_pretrained_weights_only":True,
        "NO_LM_TEST_or_answers_or_logits":True,
        "NO_codebook_selected_for_deployment":True,
        "NO_new_physical_host_or_modified_weights":True,
        "compare_only_same_K32_V8_bits":True,
        "frozen_fit_seed":SEED,
        "separate_source_group_fit_count":FIT_GROUPS,
        "separate_source_group_valid_count":VALID_GROUPS,
        "iterations":ITERATIONS,
        "baseline_actual_codes_match_recomputed_nearest":verification,
        "rate":physical_budget(len(packed.base),handle.shape),
        "outcomes_on_disjoint_SOURCE_WEIGHT_groups_ONLY":scores,
        "meaning_of_results":"Weight RMSE improvements are NOT language quality, KL, SciQ, BoolQ or 0.7bpp full-model PASS",
    }
    file=out/"v137_embedding_source_only_dictionary.json"
    file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v137_SOURCE_ONLY_SAME_RATE_EMBED_DICTIONARIES",
                      "score":scores,"code_match":verification,
                      "rate":result["rate"],"report":str(file)},
                     ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("host",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args();run(a.host,a.out)
