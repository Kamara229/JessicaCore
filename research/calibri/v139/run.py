"""Original-source only 1-bit residual quantizer pilot, NOT an LLM QA test.

The tested 0.694bpp is a *budget projection*; no new valid physical compressed
runtime has been produced here. No v89 TEST, labels, teacher logits or Green.
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
from v137.protocol import select,groups,SEED,FIT_GROUPS,VALID_GROUPS
from v138.protocol import choose_source_centers
from .protocol import budget
from .refinement import train,score

KEY="model.embed_tokens.weight"


def run(host,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    v119=read_v119(host)
    if hashlib.sha256(v119.base).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("source archive not frozen original CB59")
    archive=CB59(v119.base)
    handle=archive.handles[KEY]
    if handle.shape!=(49_152,576):
        raise ValueError("unexpected 135M tied embedding shape")
    teacher=original_teacher()
    try:
        w=teacher.get_input_embeddings().weight.detach().float().cpu().numpy().copy()
    finally:
        del teacher
    total=w.shape[0]*(w.shape[1]//8)
    fit_index,valid_index=select(total)
    fit_x,fit_normalized,fit_scales=groups(w,handle.row_scales,fit_index)
    val_x,val_normalized,val_scales=groups(w,handle.row_scales,valid_index)

    original_codes=unpack5(handle.packed.tobytes(),total)
    pinned_original_fit=original_codes[fit_index]
    pinned_original_valid=original_codes[valid_index]
    weighted=choose_source_centers(w,handle)["weighted_recode_CB59"]
    weighted_fit=_nearest_integer_128(fit_normalized,weighted)
    weighted_valid=_nearest_integer_128(val_normalized,weighted)

    options=[
      ("original_CB59",handle.centroids,pinned_original_fit,pinned_original_valid),
      ("v137_weighted_K32",weighted,weighted_fit,weighted_valid),
    ]
    scores={}
    for name,base_centers,codes_fit,codes_valid in options:
        if name=="original_CB59" and float(np.mean(_nearest_integer_128(val_normalized,base_centers)==codes_valid))!=1.0:
            raise AssertionError("baseline physical 5bit codes and native nearest-code disagree")
        refinement=train(fit_normalized,codes_fit,base_centers,
                         energy=np.square(fit_scales.astype(np.float64)),steps=8)
        baseline=score(val_x,val_normalized,val_scales,codes_valid,base_centers)
        upgraded=score(val_x,val_normalized,val_scales,codes_valid,base_centers,refinement)
        passed=upgraded["relative_weight_RMSE"] < baseline["relative_weight_RMSE"]
        scores[name]={
            "heldout_RMSE_improved":bool(passed),
            "baseline_K32_5bit_source_validation":baseline,
            "candidate_K32_K2_6bit_source_validation":upgraded,
            "relative_RMSE_ratio_refinement_vs_base":
                upgraded["relative_weight_RMSE"]/baseline["relative_weight_RMSE"],
            "K32x2xV8_FP16_candidate_sha256":
                hashlib.sha256(refinement.astype("<f2").tobytes()).hexdigest(),
        }

    output={
      "stage":"v139_PREPAID_SIX_BIT_EMBED_ONEBIT_REFINEMENT_SOURCE_ONLY",
      "pinned_source_original_CB59_sha256":ORIGINAL_CB59_SHA,
      "source_host_v119_sha256":hashlib.sha256(host.read_bytes()).hexdigest(),
      "selected_matrix":KEY,
      "tied_output_head_same_refinement_bitstream_required":True,
      "source_only_training":True,
      "no_QA_TEST_or_language_model_inference":True,
      "no_physical_candidate_built":True,
      "no_active_weights_modified":True,
      "no_lm_accuracy_claim":True,
      "v89_official_independent_TEST_unopened":True,
      "neural_Green_and_Jessica_unmodified":True,
      "train_source_groups":FIT_GROUPS,"heldout_source_groups":VALID_GROUPS,
      "seed":SEED,"physical_budget_IF_archive_implemented":budget(len(v119.base)),
      "heldout_source_weight_results":scores,
      "next_decision":"If held-out weight RMSE improves, implement fully CRC/SHA-verified paid one-bit sidecar and direct-native tied embedding+head kernel, then test complete LM QA/KL/speed. 0.7 bpp is a physical budget forecast, not proof.",
    }
    target=out/"v139_source_weight_residual_bit.json"
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":output["stage"],"budget":output["physical_budget_IF_archive_implemented"],
      "results":scores,"report":str(target)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("host",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.host,args.out)
