"""Green Tunnel prerequisite: original teacher top1 within packed model topK?

The user-defined 'Green Tunnel' could be (1) a small OUTPUT reranker or
(2) a separate hidden-state/weight correction network. BEFORE training
anything, establish a strict SUPPORT CEILING for option (1): if the
original teacher top1 token is not in the frozen CB59/CB161 topK, even
an unpaid omniscient teacher-guided reranker restricted to those K
candidates CANNOT recover that teacher choice. This is NOT a practical
probability of success nor proof of correct human speech, but a
falsifiable diagnostic identifying whether output-only routing is even
plausible versus deeper correction.

Data: historical original TRAIN FIT, four 64-token WikiText + four 64-
token AG News windows used in original v76/v158/v164. No new VALID, QA
or v89 locked independent TEST. Immutable physically paid entire
CB161 (11,292,174B, 0.6715785 bpp) anchored against completed v163
real FOUR greedy generations with just ~0.086 distinct-token ratio.
Original FP32 teacher is evaluation ONLY and NOT candidate or sidecar.
No new Green Tunnel weights; none omitted from physical rate.
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace

import torch
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v153.codec import unpack,report
from v153.runtime import install_paid_block
from v158.run import pinned_fit
from v117.protocol import ORIGINAL_CB59_SHA
from .protocol import (check,ARMS,FIT_DOMAINS,WINDOW,WINDOWS,
                       TOKENS_PER_DOMAIN,FULL_PARENT_ARCHIVE_BYTES)
from .evidence import verify_bad_real_generation
from .scoring import one_window,aggregate


@torch.inference_mode()
def run_model_window(model,token):
    logits=model(input_ids=token,use_cache=False).logits.detach().clone()
    if logits.shape!=(1,WINDOW,49152) or not torch.isfinite(logits).all():
        raise ValueError("expected native original complete 135M model source")
    return logits


def run(archive,real_v163_report,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    physical_blob=archive.read_bytes()
    physical=unpack(physical_blob)
    if (physical.size!=FULL_PARENT_ARCHIVE_BYTES or
            hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA):
        raise ValueError("original whole model bitstream is not the pinned 5bit base")
    old=verify_bad_real_generation(physical,real_v163_report)
    if old["physical_archive_SHA256"]!=physical.sha256:
        raise ValueError("full physical source changed since real generation")
    # Seal proof, questions, every K value BEFORE reading even historical FIT.
    seal=out/"v165_GT_support_test_registered_without_NEW_VALID.json"
    seal.write_text(json.dumps({
        "prior_real_v163_greedy_generation":old,
        "actual_entire_source_byte_count":physical.size,
        "physical_rate":report(physical_blob),
        "corrective_7_first_block_weight_tables_ALREADY_paid":True,
        "GREEN_TUNNEL_has_NOT_been_fitted_built_or_applied":True,
        "topK_target":list((1,8,32,128,512,2048)),
        "candidate_model_arms":list(ARMS),
        "true_input_source":"only PREVIOUSLY USED TRAIN FIT WikiText/AGNews 4x64",
        "heldout_QA_and_official_v89_TEST_UNOPENED":True,
        "oracle_uses_UNPAID_FP32_teacher_must_not_be_reported_as_candidate":True,
    },ensure_ascii=False,indent=2)+"\n")

    data,provenance=pinned_fit()
    if tuple(data)!=FIT_DOMAINS or any(len(data[d])!=WINDOW*WINDOWS for d in FIT_DOMAINS):
        raise ValueError("not precisely old TRAIN FIT source")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("not frozen original pretrained 135M teacher")
    observations={arm:{domain:[] for domain in FIT_DOMAINS} for arm in ARMS}
    seconds={arm:{domain:0.0 for domain in FIT_DOMAINS} for arm in ARMS}
    with TemporaryDirectory(dir=out) as temp:
        scratch=Path(temp)
        file=scratch/"original_full_CB59_source.cb59"
        file.write_bytes(physical.original)
        candidate,meta,_=make_model("baseline",SimpleNamespace(
            original=file,folder=scratch))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("not original direct native packed source")
        embedding=candidate.model.embed_tokens
        head=candidate.lm_head
        if embedding.handle is not head.handle:
            raise ValueError("original source trained tied input/output is not shared")
        original_block=candidate.model.layers[0]
        for domain in FIT_DOMAINS:
            for i in range(WINDOWS):
                ids=data[domain][i*WINDOW:(i+1)*WINDOW][None,:]
                teacher_logits=run_model_window(teacher,ids)
                start=perf_counter()
                baseline=run_model_window(candidate,ids)
                seconds[ARMS[0]][domain]+=perf_counter()-start
                observations[ARMS[0]][domain].append(
                    one_window(teacher_logits,baseline,ids))
                del baseline
                start=perf_counter()
                with install_paid_block(candidate,physical,scratch):
                    modified=run_model_window(candidate,ids)
                seconds[ARMS[1]][domain]+=perf_counter()-start
                observations[ARMS[1]][domain].append(
                    one_window(teacher_logits,modified,ids))
                del modified,teacher_logits
                if (candidate.model.embed_tokens is not embedding or
                        candidate.lm_head is not head or
                        candidate.model.layers[0] is not original_block):
                    raise ValueError("original layer source operator changed after diagnostic")
                if any(param.ndim>=2 for param in candidate.parameters()):
                    raise ValueError("dense original pretrained 2D matrices loaded")
    del teacher

    summaries={arm:{domain:aggregate(observations[arm][domain])
                    for domain in FIT_DOMAINS} for arm in ARMS}
    report_data={
        "stage":"v165_NO_TRAINING_GREEN_TUNNEL_TOPK_RESCUE_UPPER_BOUND",
        "actual_frozen_CB161_sha256":physical.sha256,
        "exact_full_physical_pretrained_model_rate":report(physical_blob),
        "completed_v163_GREEDY_FOUR_OUTPUT_FAIL":old,
        "no_new_weights_or_external_tunnel_trained":True,
        "hypothetical_oracle_teacher_guidance_IS_NOT_DEPLOYABLE":True,
        "source_existing_original_TRAIN_FIT_only":provenance,
        "no_heldout_VALID_test_or_SciQ_BoolQ_questions_opened":True,
        "original_FP32_teacher_used_ONLY_as_reference_to_define_oracle":True,
        "real_pretrained_student_models":list(ARMS),
        "all_complete_original_model_logits_per_original_FIT_window":True,
        "raw_gold_and_teacher_TOP1_optimistic_tie_rank_per_position":observations,
        "GREEN_TUNNEL_TOPK_oracle_upper_bounds_not_measured_Green_Tunnel":summaries,
        "full_model_CPU_source_seconds_not_separate_GT_speed":seconds,
        "no_inference_modules_purchased_or_omitted_from_bpp":True,
        "official_v89_independent_TEST_FROZEN":True,
        "no_QA_coherent_text_or_Green_Tunnel_performance_success_claim":True,
        "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    result=out/"v165_original_paid_source_GREEN_TUNNEL_TOPK_feasibility.json"
    result.write_text(json.dumps(report_data,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "physical_model_bpp":report_data[
            "exact_full_physical_pretrained_model_rate"]["all_in_bpp_unique_original"],
        "prior_real_greedy_distinct_ratios":{
            "native_CB59":old["real_v163_four_greedy_CB59_distinct_token_ratio"],
            "paid_original_block0":old["real_v163_four_greedy_paid_block0_distinct_token_ratio"],
        },
        "oracle_teacher_top1_in_student_topK":{
            arm:{domain: {
                str(k):summaries[arm][domain][
                    "topK_support_information_theoretic_reranker_ceiling_not_performance"][str(k)][
                        "teacher_TOP1_recovery_upper_bound_fraction"]
                for k in (1,8,32,128,512,2048)}
                for domain in FIT_DOMAINS} for arm in ARMS},
        "not_an_actual_Green_Tunnel_training_or_quality_measure":True,
        "result":str(result),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("immutable_paid_CB161_source",type=Path)
    p.add_argument("completed_real_v163_QA_and_greedy_JSON",type=Path)
    p.add_argument("out",type=Path)
    x=p.parse_args()
    run(x.immutable_paid_CB161_source,x.completed_real_v163_QA_and_greedy_JSON,x.out)
