"""Reject or confirm an outlier-driven paid original-layer v_proj gain.

v156 new source: WikiText v_proj -0.1133 NLL, -0.1164 KL, but only 3/8
windows had better NLL and only 2/8 had better KL; most gain came from
a single window (-2.2188 NLL). AG News gave 6/8 improvements. This is not
robust evidence, regardless of aggregate source mean PASS.

v157 freezes CB153 whole-source SHA, operator selection, and a strict
NEW source 32x64 window/domain gate *before* reading TRAIN text:
mean NLL & KL strictly better, each at least 20/32 improved windows, and
leave the single best negative delta OUT and remaining average still better.
No new source 2D bits, training, model selection or adjustments.
"""
import argparse
import hashlib
import json
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA
from v142.scoring import compare_logits,aggregate
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v154.arms import subset
from .protocol import (ARMS,DOMAINS,SELECTED,CONTROL,WINDOW,WINDOWS,
                       TOKENS,SCORED_TOKENS_PER_DOMAIN,check)
from .sources import fourth_untouched_sources
from .evidence import verify_prior_report
from .audit import summarize


@torch.inference_mode()
def evaluate_one_model(model,source):
    if len(source)!=TOKENS:
        raise ValueError("source four TRAIN windows length changed")
    collection=[]
    seconds=0.
    for index in range(WINDOWS):
        tokens=source[index*WINDOW:(index+1)*WINDOW][None,:]
        tick=perf_counter()
        logits=model(input_ids=tokens,use_cache=False).logits.detach().clone()
        seconds+=perf_counter()-tick
        if logits.shape!=(1,WINDOW,49152) or not torch.isfinite(logits).all():
            raise ValueError("source reference 135M vocabulary changed")
        collection.append(logits)
    return collection,seconds


def run(original_CB153,real_v156_result,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    original_bytes=original_CB153.read_bytes()
    physical=unpack(original_bytes)
    if hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original FP32 pretrained CB59 baseline changed")
    old_evidence=verify_prior_report(real_v156_result)
    if physical.sha256!=old_evidence["prior_source_SHA256"]:
        raise ValueError("not frozen fully paid CB153 source")
    modes={
        ARMS[0]:subset(physical,()),
        ARMS[1]:subset(physical,(SELECTED,)),
        ARMS[2]:subset(physical,(CONTROL,)),
        ARMS[3]:subset(physical,tuple(physical.patches)),
    }
    if tuple(modes)!=ARMS:
        raise AssertionError("posthoc operator arm introduced")

    # Serialize the actual selection AND strong preregistered robustness gate
    # to artifact BEFORE any new source text can enter the research process.
    sealed=out/"v157_predeclared_robustness_before_4th_holdout.json"
    sealed.write_text(json.dumps({
        "original_v156_evidence":old_evidence,
        "entire_physical_CB153_SHA256":physical.sha256,
        "all_existing_paid_source_bytes":len(original_bytes),
        "same_real_0_67158bpp_archive_charged_all_arms":True,
        "arms":list(ARMS),
        "source_windows_32_each_original_DOMAIN":True,
        "min_improved_NLL_and_KL_windows_per_domain":20,
        "leave_largest_favorable_window_out_mean_NLL_and_KL_must_still_improve":True,
        "new_fourth_source_not_loaded_at_seal_time":True
    },indent=2)+"\n")

    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("frozen SmolLM2 teacher architecture not matched")
    source,provenance=fourth_untouched_sources()
    references={domain:evaluate_one_model(teacher,source[domain])[0]
                for domain in DOMAINS}
    del teacher

    results={arm:{domain:[] for domain in DOMAINS} for arm in ARMS}
    timings={arm:{} for arm in ARMS}
    with TemporaryDirectory(dir=out) as temp:
        root=Path(temp)
        frozen_CB59=root/"complete_pinned.cb59"
        frozen_CB59.write_bytes(physical.original)
        student,meta,_=make_model("baseline",SimpleNamespace(
            original=frozen_CB59,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("unpaid original pretrained 2D matrix")
        emb=student.model.embed_tokens
        head=student.lm_head
        if emb.handle is not head.handle:
            raise ValueError("original physical tied embedding identity lost")
        for arm,paid_subset in modes.items():
            context=(install_paid_block(student,paid_subset,root)
                     if paid_subset.patches else nullcontext({"native_CB59":True}))
            with context:
                for domain in DOMAINS:
                    predictions,elapsed=evaluate_one_model(student,source[domain])
                    timings[arm][domain]=elapsed
                    for n,pred in enumerate(predictions):
                        tokens=source[domain][n*WINDOW:(n+1)*WINDOW][None,:]
                        measured=compare_logits(references[domain][n],pred,tokens)
                        measured["window_index"]=n
                        results[arm][domain].append(measured)
            if student.model.embed_tokens is not emb or student.lm_head is not head:
                raise AssertionError("paid source modifications survived trial")
            if any(p.ndim>=2 for p in student.parameters()):
                raise ValueError("original trained source loaded dense")

    scores={arm:{domain:aggregate(results[arm][domain]) for domain in DOMAINS}
            for arm in ARMS}
    check_results=summarize(scores,results)
    for arm in ARMS:
        for domain in DOMAINS:
            if scores[arm][domain]["tokens_scored"]!=SCORED_TOKENS_PER_DOMAIN:
                raise ValueError("missing locked target-source gold tokens")

    report_data={
        "stage":"v157_FOURTH_UNSEEN_SOURCE_32WINDOW_VPROJ_OUTLIER_ROBUSTNESS",
        "immutable_complete_CB153_SHA256":physical.sha256,
        "physically_paid_all_original_5plus1_source_bpp":report(original_bytes),
        "all_7_stored_modules_physically_charged_including_single_op_arms":True,
        "prior_v156_exploratory_result":old_evidence,
        "frozen_selection_and_gate_written_BEFORE_source_loaded":sealed.name,
        "fresh_fourth_original_TRAIN_split_source":provenance,
        "only_real_original_pretrained_source_and_no_NEW_fitting":True,
        "predeclared_32_windows_and_2016_gold_tokens_per_domain_arm":True,
        "all_four_native_operator_modes":list(ARMS),
        "all_4_arm_frozen_true_NLL_PPL_KL_argmax":scores,
        "paired_per_window_source_metrics":results,
        "window_level_outlier_and_strong_two_domain_gate":check_results,
        "full_model_CPU_seconds_by_arm_and_domain":timings,
        "no_frozen_independent_v89_TEST":True,
        "no_QA_answers_or_lookahead_training":True,
        "no_actual_language_or_semantic_restoration_claim":True,
        "no_Jessica_LIVE_Render_Supabase_APK_changes":True,
    }
    destination=out/"v157_fourth_source_operator_robustness.json"
    destination.write_text(json.dumps(report_data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "strong_gate_PASS":check_results["v_proj_STRONG_two_domain_robustness_pass"],
        "new_fourth_domain_results":check_results["honest_fourth_NEW_source_results"],
        "original_genuine_all_in_bpp":report_data["physically_paid_all_original_5plus1_source_bpp"]["all_in_bpp_unique_original"],
        "full_report":str(destination),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("pinned_complete_CB153",type=Path)
    parser.add_argument("completed_v156_JSON",type=Path)
    parser.add_argument("out",type=Path)
    x=parser.parse_args()
    run(x.pinned_complete_CB153,x.completed_v156_JSON,x.out)
