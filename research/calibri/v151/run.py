"""Successive fully-paid int3 rank2: once-fit then untouched source VALID.

No second-rank codebook tuned against original SciQ/BoolQ v150, v149 VALID,
or independent official v89 TEST. Original complete physical CB149 is
unchanged and nested. No LIVE Jessica/Render/Supabase/APK modifications.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v117.protocol import ORIGINAL_CB59_SHA,MAX_SOURCE_BYTES
from v142.scoring import compare_logits,aggregate
from v141.run import ORIGINAL_V140_SHA
from v143.runtime import PackedFusedHead
from v147.runtime import fused_half_head
from v149.codec import unpack as unpack_parent
from v149.runtime import ContextualHead
from v149.run import locked_fit

from .protocol import (PINNED_CB149_SHA,NEW_VALID_DOMAINS,
                       NEW_VALID_WINDOWS,NEW_VALID_TOKENS,WINDOW,
                       TRAIN_HIDDEN,FIT_DOMAINS,check_protocol)
from .codec import pack,unpack,physical_report
from .fit import fit_strict
from .runtime import install
from .sources import new_unseen_valid


@torch.inference_mode()
def verify_new_valid(teacher,student,cb59,cb149,cb151):
    # The final source archive MUST already have been serialized and hashed.
    tokens,provenance=new_unseen_valid()
    arms={"CB59":cb59,"CB149_paid_rank1":cb149,"CB151_paid_rank2_int3":cb151}
    raw={arm:{domain:[] for domain in NEW_VALID_DOMAINS} for arm in arms}
    for domain in NEW_VALID_DOMAINS:
        ids=tokens[domain]
        for index in range(NEW_VALID_WINDOWS):
            item=ids[index*WINDOW:(index+1)*WINDOW][None,:]
            with torch.inference_mode():
                target=teacher(input_ids=item,use_cache=False).logits.detach()
                base=student.model(input_ids=item,use_cache=False).last_hidden_state
                if base.shape!=(1,WINDOW,TRAIN_HIDDEN):
                    raise ValueError("original compressed decoder geometry altered")
                for arm,head in arms.items():
                    start=perf_counter()
                    student_logits=head(base).detach()
                    duration=perf_counter()-start
                    measurement=compare_logits(target,student_logits,item)
                    measurement["window_index"]=index
                    measurement["head_only_CPU_seconds"]=duration
                    raw[arm][domain].append(measurement)

    summary={arm:{domain:aggregate(raw[arm][domain])
                  for domain in NEW_VALID_DOMAINS} for arm in arms}
    shifts={}
    for domain in NEW_VALID_DOMAINS:
        baseline=summary["CB59"][domain]
        first=summary["CB149_paid_rank1"][domain]
        second=summary["CB151_paid_rank2_int3"][domain]
        shifts[domain]={
            "new_minus_prepaid_rank1_TRUE_NLL":
                second["mean_TRUE_next_token_NLL"]-first["mean_TRUE_next_token_NLL"],
            "new_minus_prepaid_rank1_teacher_KL":
                second["teacher_to_candidate_KL"]-first["teacher_to_candidate_KL"],
            "new_minus_CB59_TRUE_NLL":
                second["mean_TRUE_next_token_NLL"]-baseline["mean_TRUE_next_token_NLL"],
            "new_minus_CB59_teacher_KL":
                second["teacher_to_candidate_KL"]-baseline["teacher_to_candidate_KL"],
            "original_teacher_next_token_top1_agreement":
                second["teacher_next_token_argmax_agreement_fraction"],
            "new_minus_parent_teacher_next_token_agreement":
                second["teacher_next_token_argmax_agreement_fraction"]-
                first["teacher_next_token_argmax_agreement_fraction"],
        }
    passes=all(shifts[d]["new_minus_prepaid_rank1_TRUE_NLL"]<0 and
               shifts[d]["new_minus_prepaid_rank1_teacher_KL"]<=0
               for d in NEW_VALID_DOMAINS)
    return summary,shifts,raw,provenance,passes


def run(archive,out):
    torch.set_num_threads(2)
    check_protocol()
    out.mkdir(parents=True,exist_ok=True)
    source=archive.read_bytes()
    parent=unpack_parent(source)
    if parent.sha256!=PINNED_CB149_SHA or parent.parent.paid.sha256!=ORIGINAL_V140_SHA:
        raise ValueError("wrong frozen parent physical CB149 or CB140 source")
    if hashlib.sha256(parent.parent.paid.original).hexdigest()!=ORIGINAL_CB59_SHA:
        raise ValueError("wrong original pretrained codebook bytes")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("wrong FP32 pretrained teacher")
    with TemporaryDirectory(dir=out) as scratch:
        folder=Path(scratch)
        original=folder/"original_frozen.cb59"
        original.write_bytes(parent.parent.paid.original)
        model,metadata,_=make_model(
            "baseline",SimpleNamespace(original=original,folder=folder))
        if not metadata["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original native compressed source no longer packed")
        native_base=model.lm_head
        original_embedding=model.model.embed_tokens
        half_head=fused_half_head(model,parent.parent.paid,folder)
        static_head=PackedFusedHead(
            half_head.handle,half_head.kernel,parent.parent.bias)
        rank1=ContextualHead(static_head,parent)
        model.lm_head=rank1
        if not (original_embedding.handle is native_base.handle is rank1.handle):
            raise ValueError("original tied packed source was split")

        # FIT only on original designated source-fit data, not validation.
        train,fit_origins=locked_fit(teacher,model)
        u,bias,vector,step,fit_report=fit_strict(
            train["hidden"],train["teacher"],train["prior"])
        blob=pack(source,u,bias,vector,step)
        physical_file=out/"CALIBRI_v151_paid_successive_int3_rank2.cb151"
        physical_file.write_bytes(blob)
        paid=unpack(physical_file.read_bytes())
        if paid.base.sha256!=parent.sha256 or paid.size>MAX_SOURCE_BYTES:
            raise ValueError("archive changed or too many physical bytes")
        rank2=install(model,paid)
        if rank2.handle is not original_embedding.handle:
            raise ValueError("second rank broke original tie")

        # These held-out source TRAIN VALID windows are accessed strictly
        # AFTER complete paid CB151 has been saved and SHA verified.
        metrics,differences,raw_records,provenance,passed=verify_new_valid(
            teacher,model,native_base,rank1,rank2)

    del teacher
    result={
        "stage":"v151_SUCCESSIVE_FULLY_PAID_CONTEXTUAL_INT3_RANK2",
        "parent_CB149_SHA256":PINNED_CB149_SHA,
        "physically_serialized_CB151":physical_report(blob),
        "saved_physical_source_name":physical_file.name,
        "all_original_pretrained_layers_unmodified":True,
        "teacher_used_for_original_TRAIN_FIT_only":True,
        "original_source_FIT_provenance":fit_origins,
        "first_rank_immutable":True,
        "second_rank_predeclared_8_power_iterations_int3":True,
        "source_TRAIN_FIT_MSE_before_after":fit_report,
        "new_untouched_TRAIN_VALID_sources":provenance,
        "frozen_holdout_504_next_tokens_per_domain_per_arm":True,
        "all_new_real_valid_results":metrics,
        "real_holdout_2nd_rank_vs_rank1_and_CB59":differences,
        "second_rank_improves_both_NLL_and_KL_on_both_NEW_domains":passed,
        "actual_per_window_raw_results":raw_records,
        "NO_official_v89_independent_TEST":True,
        "NO_new_SciQ_BoolQ_QA_used_for_FIT":True,
        "NO_semantic_generation_or_usefulness_proof":True,
        "NO_LIVE_Jessica_Render_Supabase_APK_change":True,
    }
    outpath=out/"v151_physically_paid_rank2_validation.json"
    outpath.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":result["stage"],
        "complete_source":result["physically_serialized_CB151"],
        "fit":fit_report,
        "passed_two_new_holdout_domains":passed,
        "holdout_changes":differences,
        "scores":metrics,
        "report":str(outpath),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("original_frozen_CB149",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.original_frozen_CB149,args.out)
