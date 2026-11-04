"""Identify contributions of actual first-block 5+1-bit original-weight operators.

Mechanistic diagnosis ONLY on 4x64 previously used original TRAIN FIT
WikiText and AG News; never call source VALID or official v89 TEST, SciQ
or BoolQ. Same immutable 11,292,174 byte 0.6715785bpp CB161 source,
16 RAM-only operator subsets: native CB59, all seven original packed
extra 1-bit modules, seven one-module-only, seven leave-one-out.
No optimizer, new source bits, dynamic QA head, or Green Tunnel.
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
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v158.run import pinned_fit
from v142.scoring import compare_logits

from .protocol import (check,ARMS,FIT_DOMAINS,WINDOW,WINDOWS,TOKENS_EACH)
from .evidence import verify_v162_fail
from .arms import all_predeclared_arms
from .audit import summarize


@torch.inference_mode()
def four_windows(model,source):
    if len(source)!=TOKENS_EACH:
        raise ValueError("original source only 4x64 prior TRAIN FIT tokens allowed")
    collected=[]
    seconds=0.
    for w in range(WINDOWS):
        ids=source[w*WINDOW:(w+1)*WINDOW][None,:]
        start=perf_counter()
        output=model(input_ids=ids,use_cache=False).logits.detach().clone()
        seconds+=perf_counter()-start
        if output.shape!=(1,WINDOW,49152) or not torch.isfinite(output).all():
            raise ValueError("original complete source model produced invalid logits")
        collected.append(output)
    return collected,seconds


def run(immutable_archive,real_v162_report,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    complete=immutable_archive.read_bytes()
    paid=unpack(complete)
    if hashlib.sha256(paid.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("complete original CB59 trained source modified")
    earlier=verify_v162_fail(paid,real_v162_report)
    source,origin=pinned_fit()
    if tuple(source)!=FIT_DOMAINS:
        raise ValueError("missing original prior source TRAIN FIT groups")

    interventions=all_predeclared_arms(paid)
    if tuple(interventions)!=ARMS:
        raise ValueError("16 original source operator diagnostic arms changed")
    frozen=out/"v164_locked_before_original_TRAIN_FIT_attribution.json"
    frozen.write_text(json.dumps({
        "existing_prior_v162_32window_failure":earlier,
        "unmodified_full_parent_SHA":paid.sha256,
        "all_physically_paid_original_source_bytes":len(complete),
        "full_original_pretrained_bpp":report(complete)["all_in_bpp_unique_original"],
        "all_16_RERUN_DIAGNOSTIC_only_arms":list(ARMS),
        "original_source_TRAIN_FIT_only":origin,
        "never_open_new_VALID_or_protected_TEST":True,
        "no_posthoc_operator_selection_as_validated_release":True,
    },indent=2,ensure_ascii=False)+"\n")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("original FP32 reference pretrained 30 decoder layers required")
    reference={d:four_windows(teacher,source[d])[0] for d in FIT_DOMAINS}
    del teacher

    records={arm:{d:[] for d in FIT_DOMAINS} for arm in ARMS}
    elapsed={arm:{} for arm in ARMS}
    with TemporaryDirectory(dir=out) as t:
        root=Path(t)
        original=root/"original_CB59_full_source.cb59"
        original.write_bytes(paid.original)
        model,meta,_=make_model(
            "baseline",SimpleNamespace(original=original,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("unpaid trained 2D original source matrix resident in runtime")
        input_embedding=model.model.embed_tokens
        tied_head=model.lm_head
        original_block0=model.model.layers[0]
        if input_embedding.handle is not tied_head.handle:
            raise ValueError("the original embedding and original head are not tied")

        for arm,partial in interventions.items():
            context=(install_paid_block(model,partial,root) if partial.patches
                     else nullcontext({"native_CB59":True}))
            with context:
                for domain in FIT_DOMAINS:
                    original_outputs,sec=four_windows(model,source[domain])
                    elapsed[arm][domain]=sec
                    for index,(target,actual) in enumerate(
                            zip(reference[domain],original_outputs)):
                        ids=source[domain][index*WINDOW:(index+1)*WINDOW][None,:]
                        measure=compare_logits(target,actual,ids)
                        measure["reused_TRAIN_FIT_window"]=index
                        records[arm][domain].append(measure)
            if (model.model.embed_tokens is not input_embedding or
                    model.lm_head is not tied_head or
                    model.model.layers[0] is not original_block0):
                raise ValueError("real operator patch not restored after diagnostic")
            if any(p.ndim>=2 for p in model.parameters()):
                raise ValueError("original trained 2D FP32 model leaked into packed student")

    measured=summarize(records,elapsed)
    final={
        "stage":"v164_ORIGINAL_PAID_BLOCK0_SEVEN_OPERATOR_OLD_TRAIN_FIT_ATTRIBUTION",
        "whole_physical_paid_archive_SHA256":paid.sha256,
        "whole_original_model_source_rate":report(complete),
        "full_source_7_paid_operators_charged_even_when_RAM_disabled":True,
        "previous_v162_robust_two_domain_gate_FAILED_not_suppressed":earlier,
        "old_TRAIN_FIT_original_source_only":origin,
        "source_model_weights_changed_or_refit":False,
        "all_predeclared_16_original_operator_arms":list(ARMS),
        "training_FIT_per_arm_natural_text_next_token_metrics":records,
        "paired_all_16_arm_full_model_diagnostic":measured,
        "all_native_full_model_timings":elapsed,
        "no_statistically_independent_VALID_TEST_or_frozen_v89_TEST":True,
        "diagnostic_operator_ranking_not_a_genuine_heldout_validation":True,
        "no_new_semantic_or_QA_success_claim":True,
        "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    path=out/"v164_real_original_paid_layer0_7ops_on_prior_TRAIN_FIT.json"
    path.write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":final["stage"],
        "full_original_physical_bpp":report(complete)["all_in_bpp_unique_original"],
        "prior_robust_v162":"FAIL",
        "old_source_all7_and_best_SINGLE_PER_DOMAIN":{
            d:{
                "all7":measured["per_previous_original_TRAIN_FIT_source"][d][
                    "all7_vs_original_CB59"],
                "best_single_NOT_VALID":measured["per_previous_original_TRAIN_FIT_source"][d][
                    "best_single_by_OLD_TRAIN_FIT_NLL_only_NOT_VALID"],
            } for d in FIT_DOMAINS},
        "report":str(path),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("frozen_v161_complete_paid_archive",type=Path)
    p.add_argument("actual_completed_v162_report",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    run(args.frozen_v161_complete_paid_archive,
        args.actual_completed_v162_report,args.out)
