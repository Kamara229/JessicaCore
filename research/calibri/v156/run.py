"""Third genuinely unseen original TRAIN-source crossdomain check.

v155's old-diagnostic WORST operator, model.layers.6.self_attn.v_proj,
surprisingly improved both true NLL and teacher KL on v155's two domains.
That exploratory heldout result selected this hypothesis, so CANNOT double
as final validation. Freeze the original complete CB153 SHA and v155
evidence BEFORE opening a different train-only source split in v156.

Four preregistered arms are original native CB59, v_proj only, o_proj
only and all7 original 1-bit module patches, with one immutable original
complete 0.6716-bpp CB153 physical archive charged for all conditions.
NO green sidecar, trained FP32 weights or dynamic adapter on deployment.
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
from v142.scoring import compare_logits,aggregate
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v154.arms import subset
from v117.protocol import ORIGINAL_CB59_SHA

from .protocol import (SELECTED_KEY,SECONDARY_KEY,ARMS,DOMAINS,WINDOW,
                       WINDOWS,TOKENS,check)
from .selection import freeze_exploratory_evidence
from .sources import load_third_holdout
from .audit import summarize


@torch.inference_mode()
def forward_eight(model,source):
    if len(source)!=TOKENS:
        raise ValueError("wrong registered source length")
    full=[]
    sec=0.
    for window in range(WINDOWS):
        ids=source[window*WINDOW:(window+1)*WINDOW][None,:]
        t0=perf_counter()
        output=model(input_ids=ids,use_cache=False).logits.detach().clone()
        sec+=perf_counter()-t0
        if output.shape!=(1,WINDOW,49152) or not torch.isfinite(output).all():
            raise ValueError("native LLM source logits invalid")
        full.append(output)
    return full,sec


def run(paid_source,earlier_report,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    blob=paid_source.read_bytes()
    paid=unpack(blob)
    if hashlib.sha256(paid.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("original trained CB59 compressed source altered")
    prior=freeze_exploratory_evidence(earlier_report)
    if prior["actual_v155_sha"]!=paid.sha256:
        raise ValueError("third source diagnostic uses another physically paid candidate")

    arms={
        ARMS[0]:subset(paid,()),
        ARMS[1]:subset(paid,(SELECTED_KEY,)),
        ARMS[2]:subset(paid,(SECONDARY_KEY,)),
        ARMS[3]:subset(paid,tuple(paid.patches)),
    }
    if tuple(arms)!=ARMS:
        raise ValueError("unregistered source conditional operator")
    # Immutable full archive and decision must be frozen to disk before
    # source data are loaded; no fitting/reselection is performed afterward.
    seal=out/"v156_vproj_SELECTED_before_new_SOURCE.json"
    seal.write_text(json.dumps({
        "selected_from":prior,
        "whole_source_SHA256":paid.sha256,
        "physical_whole_source_bpp":report(blob)["all_in_bpp_unique_original"],
        "all_four_predeclared_arms":list(ARMS),
        "no_source_VALID_accessed_yet":True,
    },indent=2)+"\n")

    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("not original 135M pretrained FP32 source")
    source,provenance=load_third_holdout()
    ref={domain:forward_eight(teacher,source[domain])[0]
         for domain in DOMAINS}
    del teacher
    raw={arm:{domain:[] for domain in DOMAINS} for arm in ARMS}
    timings={arm:{} for arm in ARMS}
    with TemporaryDirectory(dir=out) as scratch:
        root=Path(scratch)
        original=root/"original_trained.cb59"
        original.write_bytes(paid.original)
        student,meta,_=make_model(
            "baseline",SimpleNamespace(original=original,folder=root))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("candidate loaded original dense FP32 weight matrix")
        old_emb=student.model.embed_tokens
        old_head=student.lm_head
        if old_emb.handle is not old_head.handle:
            raise ValueError("original tied physical codewords changed")
        for arm,physical in arms.items():
            active=(install_paid_block(student,physical,root) if physical.patches
                    else nullcontext({"native_CB59_only":True}))
            with active:
                for domain in DOMAINS:
                    outputs,duration=forward_eight(student,source[domain])
                    timings[arm][domain]=duration
                    for w,(teacher_logits,student_logits) in enumerate(zip(ref[domain],outputs)):
                        ids=source[domain][w*WINDOW:(w+1)*WINDOW][None,:]
                        row=compare_logits(teacher_logits,student_logits,ids)
                        row["heldout_window"]=w
                        raw[arm][domain].append(row)
            if (student.model.embed_tokens is not old_emb or
                    student.lm_head is not old_head):
                raise AssertionError("paid transformer source unexpectedly mutated tied embedding")
            if any(p.ndim>=2 for p in student.parameters()):
                raise AssertionError("unpaid dense pretrained source survived")

    summary={arm:{domain:aggregate(raw[arm][domain]) for domain in DOMAINS}
             for arm in ARMS}
    audit=summarize(summary,raw)
    report_data={
        "stage":"v156_POST_SELECTION_THIRD_DISJOINT_SOURCE_VALID_VPROJ",
        "immutable_whole_CB153_sha":paid.sha256,
        "all_in_physical_source":report(blob),
        "original_stored_seven_paid_patches_NOT_free_in_any_arm":True,
        "frozen_prior_v155_exploratory_selection":prior,
        "prior_SELECTION_file_written_before_new_source_open":seal.name,
        "third_disjoint_source_TRAIN_only":provenance,
        "no_fit_or_hyperparameter_change_after_v155":True,
        "eight_64_token_windows_and_504_gold_next_tokens_per_domain":True,
        "all_four_locked_source_operator_arms":list(ARMS),
        "per_arm_per_domain_gold_NLL_PPL_FP32_teacher_KL_argmax":summary,
        "raw_paired_windows":raw,
        "paired_window_sign_and_holdout_audit":audit,
        "full_model_CPU_prefill_seconds":timings,
        "no_official_v89_locked_TEST":True,
        "no_test_or_QA_answer_labels_used":True,
        "no_semantic_generation_or_release_claim":True,
        "no_live_Jessica_Render_Supabase_APK_change":True,
    }
    result=out/"v156_vproj_third_new_source_holdout.json"
    result.write_text(json.dumps(report_data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "v_proj_third_holdout_PASS_both":audit[
            "v_proj_improves_NLL_and_FP32_teacher_KL_on_both_third_holdout_domains"],
        "two_new_domain_deltas":audit["new_domain_deltas_and_pairing"],
        "all_4_arm_metrics":summary,
        "real_physical_bpp":report_data["all_in_physical_source"]["all_in_bpp_unique_original"],
        "report":str(result),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("frozen_full_CB153",type=Path)
    p.add_argument("completed_v155_source_VALID_JSON",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.frozen_full_CB153,a.completed_v155_source_VALID_JSON,a.out)
