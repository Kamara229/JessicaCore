"""Causal Green Tunnel placement diagnostic; teacher activations are UNPAID.

Actual previous v165: even omniscient output top128 reranker on full
0.67158 bpp CB161 could select teacher top1 only 11.5% Wiki / 9.5%
AG News on same ORIGINAL old TRAIN FIT contexts. Hence an output-only
candidate bottleneck is almost certainly inadequate, and the next step
is to find which hidden-stream boundaries might causally restore full
teacher next-token NLL/KL if exact original teacher states are supplied.

Run FIVE independent oracle swaps of the teacher's ORIGINAL FP32 hidden
state at embedding, block0, block9, block28 and final norm, as opposed
to falsely claiming that our compressed model computed these activations.
These are non-deployable teacher-assisted counterfactuals. A real Green
Tunnel with its own trained constant-sized neural weights must be built
and physically charged separately and may NOT reach any oracle bound.

Inputs limited to 4x64 OLD TRAIN FIT original source WikiText and News,
identical across teacher, native CB59 and fully paid original CB161.
No other model or production backend changed. v89 independent TEST locked.
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

from .protocol import (check,ARM_NAMES,FIT_DOMAINS,TEACHER_ONLY_STAGES,
                       WINDOW,WINDOWS,PAID_V161_BYTES)
from .evidence import check_v165
from .hooks import record_reference,inject_illegal_teacher_oracle
from .audit import summarize


@torch.inference_mode()
def paired_logits(model,ids):
    result=model(input_ids=ids,use_cache=False).logits.detach().clone()
    if result.shape!=(1,WINDOW,49152) or not torch.isfinite(result).all():
        raise ValueError("actual pretrained original source logits invalid")
    return result


def run(archive,v165_report,out):
    torch.set_num_threads(2)
    check()
    out.mkdir(parents=True,exist_ok=True)
    data=archive.read_bytes()
    physical=unpack(data)
    if (len(data)!=PAID_V161_BYTES or
            hashlib.sha256(physical.original).hexdigest()!=ORIGINAL_CB59_SHA):
        raise ValueError("not pinned complete original physically paid CB161 source")
    measured_prior=check_v165(v165_report,physical)
    if measured_prior["full_paid_CB161_SHA256"]!=physical.sha256:
        raise ValueError("source changed after original topK ceiling")
    sealed=out/"v166_non_deployable_teacher_oracle_protocol_sealed.json"
    sealed.write_text(json.dumps({
        "source_CB161_SHA256":physical.sha256,
        "original_physical_model_BYTES":len(data),
        "original_physical_model_bpp":report(data)["all_in_bpp_unique_original"],
        "old_measured_v165_output_only_ceiling":measured_prior,
        "five_precommitted_teacher_hidden_swap_stages":TEACHER_ONLY_STAGES,
        "native_original_and_physically_paid_arms":ARM_NAMES,
        "original_previously_consumed_TRAIN_FIT_only":FIT_DOMAINS,
        "no_new_source_VALID_QA_independent_TEST_exposed":True,
        "FP32_original_teacher_hidden_info_NOT_available_to_a_real_Green_Tunnel":True,
        "FP32_teacher_weight_and_activation_cost_EXCLUDED_from_native_model_rate_but_ORACLE_NOT_ELIGIBLE":True,
        "not_a_trained_or_physically_purchased_Green_Tunnel":True,
    },indent=2,ensure_ascii=False)+"\n")

    old_text,origin=pinned_fit()
    if tuple(old_text)!=FIT_DOMAINS:
        raise ValueError("must use original two historic FIT corpora only")
    teacher=original_teacher()
    if len(teacher.model.layers)!=30:
        raise ValueError("not original full pretrained FP32 reference model")
    by_arm={arm:{domain:{"baseline":[],
                           **{name:[] for name in TEACHER_ONLY_STAGES}}
                 for domain in FIT_DOMAINS}
            for arm in ARM_NAMES}
    timings={arm:{domain:{"baseline":0.,
                           **{name:0. for name in TEACHER_ONLY_STAGES}}
                  for domain in FIT_DOMAINS}
             for arm in ARM_NAMES}

    with TemporaryDirectory(dir=out) as temp:
        scratch=Path(temp)
        source=scratch/"original_fixed_native_CB59.cb59"
        source.write_bytes(physical.original)
        student,meta,_=make_model(
            "baseline",SimpleNamespace(original=source,folder=scratch))
        if not meta["zero_dense_pretrained_2D_hot_weights"]:
            raise ValueError("original native 2D pretrained matrices unexpectedly dense")
        tied=student.model.embed_tokens
        head=student.lm_head
        layer0=student.model.layers[0]
        if tied.handle is not head.handle:
            raise ValueError("original trained tied input and output were decoupled")

        for domain in FIT_DOMAINS:
            tokens=old_text[domain]
            if len(tokens)!=WINDOW*WINDOWS:
                raise ValueError("original source tokens not exact historical four windows")
            for i in range(WINDOWS):
                ids=tokens[i*WINDOW:(i+1)*WINDOW][None,:]
                # Original FP32 reference only, measured on matched source.
                with record_reference(teacher) as hidden:
                    truth=paired_logits(teacher,ids)
                if tuple(hidden)!=TEACHER_ONLY_STAGES:
                    raise ValueError("reference FP32 output did not visit 5 preregistered stages")

                for arm in ARM_NAMES:
                    context=(nullcontext({"native_CB59":True})
                             if arm==ARM_NAMES[0]
                             else install_paid_block(student,physical,scratch))
                    with context:
                        for stage in ("baseline",)+TEACHER_ONLY_STAGES:
                            swap=(nullcontext()
                                  if stage=="baseline"
                                  else inject_illegal_teacher_oracle(
                                      student,stage,hidden[stage]))
                            start=perf_counter()
                            with swap:
                                output=paired_logits(student,ids)
                            timings[arm][domain][stage]+=perf_counter()-start
                            measured=compare_logits(truth,output,ids)
                            measured["source_old_FIT_window"]=i
                            measured["oracle_FP32_hidden_ASSISTANCE_NOT_DEPLOYABLE"]=(stage!="baseline")
                            measured["reference_intervention_STAGE"]=stage
                            by_arm[arm][domain][stage].append(measured)
                if (student.model.embed_tokens is not tied or student.lm_head is not head
                        or student.model.layers[0] is not layer0):
                    raise ValueError("original actual packed student was mutated by oracle hook")
                if any(p.ndim>=2 for p in student.parameters()):
                    raise ValueError("original native 2D trained matrix leaked into student")

    del teacher
    analysis=summarize(by_arm)
    report_data={
        "stage":"v166_GREEN_TUNNEL_UNPAID_TEACHER_HIDDEN_BOUNDARY_CAUSAL_ORACLES",
        "original_full_real_physical_CB161_source_SHA256":physical.sha256,
        "only_real_compressed_model_rate_including_all_paid_operator_weights":report(data),
        "actual_completed_v165_output_only_ceiling":measured_prior,
        "original_pretrained_FP32_reference_NOT_a_deployment_archive":True,
        "teacher_hidden_injected_externally_is_UNPAID_oracle_AND_not_a_GT_model":True,
        "same_original_TRAIN_FIT_only":origin,
        "five_exact_real_source_hidden_intervention_boundaries":list(TEACHER_ONLY_STAGES),
        "all_paired_original_gold_next_token_metrics_and_top1":by_arm,
        "full_original_two_model_X_five_teacher_oracle_causal_effects":analysis,
        "single_source_CPU_forward_seconds_without_stable_speed_benchmark":timings,
        "future_small_separate_GT_must_have_training_data_params_bytes_speed_and_new_VALID_recorded":True,
        "no_oracle_quality_claimed_as_trained_or_physically_paid_source":True,
        "frozen_independent_v89_TEST_unopened":True,
        "no_new_QA_or_SciQ_BoolQ_or_LANGUAGE_acceptance_claim":True,
        "no_LIVE_Jessica_Render_Supabase_APK_modifications":True,
    }
    target=out/"v166_original_hidden_GT_causal_boundary_unpaid_oracles.json"
    target.write_text(json.dumps(report_data,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "stage":report_data["stage"],
        "complete_original_candidate_bpp":report(data)["all_in_bpp_unique_original"],
        "prior_teacher_TOP1_in_paid_128_upper_bounds":{
            domain:measured_prior[
                "actual_unpaid_v165_output_only_GT_topK_limits"][domain][
                    "perfect_teacher_top128_recovery_ceiling"]
            for domain in FIT_DOMAINS},
        "five_ORACLE_boundary_teacher_NLL_KL_TOP1_effects":analysis[
            "actual_old_TRAIN_FIT_source_only"],
        "results_are_NONDEPLOYABLE_FP32_TEACHER_BOUND_ONLY":True,
        "report":str(target),
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("complete_original_paid_CB161",type=Path)
    parser.add_argument("completed_real_v165_topK_JSON",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.complete_original_paid_CB161,args.completed_real_v165_topK_JSON,args.out)
