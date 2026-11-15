"""S8: physically train calibrated selection among shared 3-way relation hypotheses on S5 TRAIN12.

Uses immutable S5 genuine Calibri 43M CPU layer8 post-GT features and the
physically trained S7 NRA3 CB59 archive. No repeated 43M prefill, no new
operator, no hidden future knowledge, no independent EVAL. The three
hypotheses are weighted in ONE existing GRU update per round.

Stage output is TRAIN FIT ONLY. Zero Calibri output influence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import torch

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s1.entropy_envelope import (
    store_lossless_cb59,restore_lossless_cb59,
)
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s6.physical_joint_grammar_qat import (
    S5_ORIGINAL_RUN,S5_SOURCE_SHA256,S194_FP16_SHA256,
    load_pinned_real43m_train12,PhysicallyPaidNRA3QAT,
    train_locked_12_real_hidden,
)
from research.nra3_s7.parallel_training import (
    attach_parallel_objective,parallel_train_objective,
)


S7_GENUINE_REAL_RUSSIAN_RUN=43710668076
S7_PAID_CB59_SHA="a51a2984b422d188c8b4d0f67ab9dd10c52760923a8dfd2187bd1f1d57325641"
S4_TRAIN12_SHA="6ee1f270d413f4cfc79727bd48386ee29881520fcf6834309869733fd00f7a61"


def _digest(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def _check(value,description):
    if not value:
        raise ValueError("NRA3 S8: "+description)


def _cpu_latency_ms(model,sample,repeat=3):
    """Exploratory *NRA-only* time, NOT full 43M end-to-end speed proof."""
    with torch.inference_mode():
        x=sample.hidden320.unsqueeze(0)
        model(x)
        times=[]
        for _ in range(repeat):
            t0=time.perf_counter()
            model(x)
            times.append((time.perf_counter()-t0)*1000)
    return statistics.median(times)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s5-hidden",type=Path,required=True)
    ap.add_argument("--s5-report",type=Path,required=True)
    ap.add_argument("--s7-cb59",type=Path,required=True)
    ap.add_argument("--s7-report",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("S8 does not overwrite a prior research candidate")
    torch.manual_seed(307117)
    torch.set_num_threads(2)
    items,s5=load_pinned_real43m_train12(args.s5_hidden,args.s5_report)
    s7_bytes=args.s7_cb59.read_bytes()
    source_sha=_digest(s7_bytes)
    _check(source_sha==S7_PAID_CB59_SHA and len(s7_bytes)==653798,
           "S7 physically paid 6M parent source SHA mismatch")
    s7=json.loads(args.s7_report.read_text(encoding="utf-8"))
    for key,expected in (
        ("CB59_trained_sha256",S7_PAID_CB59_SHA),
        ("original_unique_NRA3_parameters",6_106_154),
        ("source_S5_real_43M_layer8_sha256",S5_SOURCE_SHA256),

        ("source_S4_Russian_UD_TRAIN12_sha256",S4_TRAIN12_SHA),
        ("original_parent_43M_weights_updated",False),
        ("full_native_C_cpu_NRA3_parallel_inference",True),
        ("candidate_accepted",False),
        ("output_gate",0),
        ("independent_heldout_Russian_accuracy_measured",False),
        ("sixth_reserved_S3CI_EVAL128_untouched",True),
        ("old_five_consumed_EVAL128_untouched",True),
    ):
        _check(s7.get(key)==expected,"original S7 TRAIN-only source flag drift: "+key)
    _check(len(items)==12 and
           s5["S4_RU_UD_TRAIN12_CANONICAL_SHA256"]==S4_TRAIN12_SHA,
           "S5 physical TRAIN12 annotation SHA mismatch")
    cfg=NRA3Config(parallel_hypotheses=3,parallel_selector_version=1)
    qat=attach_parallel_objective(PhysicallyPaidNRA3QAT(s7_bytes,cfg))
    metrics=train_locked_12_real_hidden(qat,items,steps=24,lr=.006)
    paid=qat.save_full_paid_physical_cb59()
    counts=qat.physical_change_counts(paid)
    _check(len(paid)==len(s7_bytes) and _digest(paid)!=source_sha and
           CB59(paid).meta["original_unique_parameter_count"]==6_106_154,
           "S8 original 5bit paid source changed capacity or did not learn")
    args.out.mkdir(parents=True,exist_ok=True)
    native_lib=args.out/"lib_calibri_cb59_s8_native.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(native_lib)],
                   check=True)
    native_parallel=NRA3NativePackedCPU(paid,cfg,native_lib).eval()
    native_s7=NRA3NativePackedCPU(
        s7_bytes,NRA3Config(parallel_hypotheses=3),native_lib).eval()
    native_before=NRA3NativePackedCPU(s7_bytes,cfg,native_lib).eval()
    with torch.inference_mode():
        evidence=[parallel_train_objective(
            native_parallel,s.hidden320.unsqueeze(0),s.row,s.aligned)
                  for s in items]
        native_train_loss=sum(float(e.total) for e in evidence)/len(evidence)
        means=torch.stack([e.branch_means for e in evidence]).mean(0)
        heads=sum(float(e.relational_ud_head) for e in evidence)/len(evidence)
    from research.nra3_s8.selector_supervision import (
        locked_train12_weak_target_counts,BRANCHES,
    )
    labeled=locked_train12_weak_target_counts()
    with torch.inference_mode():
        before_evidence=[
            parallel_train_objective(
                native_before,s.hidden320.unsqueeze(0),s.row,s.aligned)
            for s in items
        ]
        after_choices=[]
        prior_choices=[]
        classes=[]
        for model,store in ((native_parallel,after_choices),
                            (native_before,prior_choices)):
            for sample in items:
                output=model(sample.hidden320.unsqueeze(0))
                indices=[w.last_token for w in sample.aligned.word_alignment]
                store.extend(output.parallel_hypothesis_gates[
                    0,indices,:].argmax(-1).tolist())
        for e in evidence:
            _check(e.selector_gold_indices is not None and
                   e.selector_train_ce is not None,
                   "S8 full TRAIN-only router target missing")
            classes.extend(e.selector_gold_indices.tolist())
    _check(len(classes)==len(after_choices)==len(prior_choices)>30 and
           len(set(classes))==3,"S8 no 3-class registered weak supervision")
    selector_ce_before=sum(float(e.selector_train_ce) for e in before_evidence)/12
    selector_ce_after=sum(float(e.selector_train_ce) for e in evidence)/12
    route_acc_before=sum(a==b for a,b in zip(prior_choices,classes))/len(classes)
    route_acc_after=sum(a==b for a,b in zip(after_choices,classes))/len(classes)
    label_hist={BRANCHES[k]:classes.count(k) for k in range(3)}
    used_hist={BRANCHES[k]:after_choices.count(k) for k in range(3)}
    _check(all(v>0 for v in label_hist.values()) and
           means.abs().max()>0 and selector_ce_after>0,
           "S8 calibrated three-route metric invalid")
    _check(abs(native_train_loss-metrics["after_train12_multitask_loss"])<0.025
           and native_train_loss<metrics["before_train12_multitask_loss"]
           and means.shape==(3,) and abs(float(means.sum())-1.)<1e-5
           and all(bool(torch.isfinite(e.relational_ud_head)) for e in evidence),
           "S8 physically trained parallel native C CPU disagrees with TRAIN loss")
    _check(len(list(native_parallel.parameters()))==0 and
           native_parallel.native_cpu_weights_dense_resident is False and
           native_parallel.output_gate==0 and
           native_parallel.candidate_accepted is False and
           len(native_parallel.source.handles)==43,
           "S7 native original CB59 or zero gate violated")
    base_ms=_cpu_latency_ms(native_s7,items[0])
    parallel_ms=_cpu_latency_ms(native_parallel,items[0])
    _check(base_ms>0 and parallel_ms>0,"S8 CPU microbenchmark invalid")
    raw=args.out/"NRA3_S8_CALIBRATED_PARALLEL_RELATION_RU_TRAIN12_PAID_CB59.cb59"
    raw.write_bytes(paid)
    _check(raw.read_bytes()==paid,"S8 raw CB59 physical file reload drift")
    entropy=store_lossless_cb59(paid)
    compressed=args.out/"NRA3_S8_CALIBRATED_PARALLEL_RELATION_RU_TRAIN12_PAID_S61.n3s61"
    compressed.write_bytes(entropy)
    _check(restore_lossless_cb59(compressed.read_bytes())==paid,
           "S8 lossless physical source reload drift")
    result={
        "stage":"NRA3_S8_CALIBRATED_THREE_BRANCH_RELATION_SELECTOR_TRAIN12_REAL43M_RESEARCH",
        "source_S7_run_id":S7_GENUINE_REAL_RUSSIAN_RUN,
        "source_S7_paid_CB59_sha256":source_sha,
        "source_S5_real_43M_layer8_sha256":S5_SOURCE_SHA256,
        "source_S4_Russian_UD_TRAIN12_sha256":S4_TRAIN12_SHA,
        "hypothesis_count":3,
        "selector_mode":"evidence_cosine_long_range_local_concentration_role_affinity",
        "selector_train_weak_target_counts":labeled,
        "selector_train_actual_word_target_counts":label_hist,
        "selector_train_prediction_counts":used_hist,
        "selector_train_ce_before_native":selector_ce_before,
        "selector_train_ce_after_native":selector_ce_after,
        "selector_train_accuracy_before_native":route_acc_before,
        "selector_train_accuracy_after_native":route_acc_after,
        "selection_target_heuristic_is_training_only":True,
        "selector_uses_no_inference_gold":True,
        "selector_real_independent_quality_gain_proven":False,
        "native_3way_vs_S7_3way_only_CPU_comparison":True,
        "hypothesis_types":["long_range_causal","local_causal_4_tokens",
                            "predicted_role_complementarity_causal"],
        "hypothesis_scoring_from_shared_original_paid_rel_q_k_v":True,
        "independent_43M_full_forward_count_for_each_hypothesis":0,
        "original_CB59_43_operator_count":43,
        "original_unique_NRA3_parameters":native_parallel.unique_parameters,
        "extra_paid_trainable_parameters_added_for_parallel_hypotheses":0,
        "original_5bit_code_indices_unchanged":True,
        "old_1d_physical_source_unchanged":True,
        "true_Russian_TRAIN_sentences":len(items),
        "new_real_relation_ud_head_objective":True,
        "train_only_relational_ud_head_loss_after_native":heads,
        "parallel_branch_mean_probs_native":means.tolist(),
        "full_native_C_cpu_NRA3_parallel_inference":True,
        "whole_NRA_only_S7_uncalibrated_3way_median_ms":base_ms,
        "whole_NRA_only_S8_calibrated_3way_median_ms":parallel_ms,
        "NRA_only_speed_ratio_exploratory_not_full_model":base_ms/parallel_ms,
        "prior_S7_3way_only_median_ms":base_ms,
        "new_S8_calibrated_3way_only_median_ms":parallel_ms,
        "real_43M_plus_GT_CPU_speed_ratio_verified":False,
        "full_616_J7_candidate_protection_measured":False,
        "full_answer_EOS_unseen_generalization_measured":False,
        "actual_Calibri_GT_NRA_inclusive_bpp_certified":False,
        "independent_heldout_Russian_accuracy_measured":False,
        "relational_core_trained_with_UD_syntax_links_only":True,
        "no_coreference_or_multihop_gold_trained":True,
        "old_five_consumed_EVAL128_untouched":True,
        "sixth_reserved_S3CI_EVAL128_untouched":True,
        "no_LIVE_Render_Supabase_APK_or_Jessica_chat":True,
        "CB59_filename":raw.name,
        "CB59_original_bytes":len(s7_bytes),
        "CB59_trained_physical_bytes":len(paid),
        "CB59_trained_sha256":_digest(paid),
        "S61_filename":compressed.name,
        "S61_trained_physical_bytes":len(entropy),
        "S61_trained_sha256":_digest(entropy),
        "physical_saved_only_after_cpu_model_corroboration":True,
        "output_gate_zero_Calibri_unchanged":True,
        "candidate_accepted":False,
        "output_gate":0,
        **metrics,
        **counts,
    }
    report=args.out/"NRA3_S8_CALIBRATED_ROUTER_RU12_TRAIN_PHYSICAL_REPORT.json"
    report.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,
                                 indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
