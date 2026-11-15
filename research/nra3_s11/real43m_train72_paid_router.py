"""S11: real 43M hidden TRAIN72 -> 39-param physically paid live-in-NRA router.

Source: immutable S10 72 original Calibri post-GT layer8 real CPU FP16;
frozen S8 43 physically packed CB59 matrices, no gradient to parent/GT/NRA.
Train only tiny selector on imperfect S9 UD weak TRAIN labels. Then
execute ENTIRE frozen NRA with loaded FP16 router in the real 6-cycle
3way native C+tiny-CPU path. No independent holdout/QA or LIVE influence.

This source explicitly separates train-only branch accuracy and actual
semantic/generalization truth (not measured). A model may train without
being eligible for activation. No spent EVAL128 or reserved S3CI used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.paid_router import (
    PaidThreeWayRouter,D,K,SOURCE_PARAMETERS,PHYSICAL_BYTES,train_balanced_router,
)
from research.nra3_s11.real_ru72_source import (
    checked_real_train72,S10_CAPTURE_RUN,S10_FEATURE_SHA,
)
from research.nra3_s11.weak_ru72_train_labels import (
    BRANCHES,weak_train_router_labels,
)

S8b_ORIGINAL_PAID_CB59_SHA="d710aed710a44719b6003434a2dd83150861928ce167364cb92329beabf7be97"
S8b_RUN=43731085988


def require(ok,msg:str):
    if not ok:
        raise ValueError("NRA3 S11 full REAL 72 physical router "+msg)


def _bench(model,hidden,*,reps=2):
    # Exploratory NRA-only, NOT actual parent 43M+GT+NRA speed.
    x=hidden.unsqueeze(0)
    with torch.inference_mode():
        model(x)
        vals=[]
        for i in range(reps):
            start=time.perf_counter()
            model(x)
            vals.append((time.perf_counter()-start)*1000)
    return statistics.median(vals)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s10-hidden",required=True,type=Path)
    ap.add_argument("--s10-report",required=True,type=Path)
    ap.add_argument("--s8-cb59",required=True,type=Path)
    ap.add_argument("--s8-report",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("S11 never overwrites an existing physical research run")
    torch.set_num_threads(2)
    samples,s10=checked_real_train72(args.s10_hidden,args.s10_report)
    require(len(samples)==72 and s10["real_source_train72_count"]==72,
            "missing true S10 original 43M+GT 72 physical source")
    raw=args.s8_cb59.read_bytes()
    before_sha=hashlib.sha256(raw).hexdigest()
    require(len(raw)==653798 and before_sha==S8b_ORIGINAL_PAID_CB59_SHA,
            "original frozen S8b rejected but physically trained CB59 wrong SHA")
    prior=json.loads(args.s8_report.read_text(encoding="utf-8"))
    for k,expected in (
        ("CB59_trained_sha256",S8b_ORIGINAL_PAID_CB59_SHA),
        ("original_unique_NRA3_parameters",6_106_154),
        ("selector_version",2),
        ("source_S5_real_43M_layer8_sha256",
         "ba04d54d399419ec857d01b8401af868213a7f8b2c219d384986b8cd08b1ba1f"),
        ("output_gate",0),
        ("candidate_accepted",False),
        ("old_five_consumed_EVAL128_untouched",True),
        ("sixth_reserved_S3CI_EVAL128_untouched",True),
    ):
        require(prior.get(k)==expected,"S8b original physical parent flag "+k)
    origin=CB59(raw)
    require(origin.meta["original_unique_parameter_count"]==6_106_154
            and len(origin.handles)==43,
            "not exactly the pre-existing full packed 43-matrix 6M model")
    args.out.mkdir(parents=True,exist_ok=True)
    lib=args.out/"lib_s11_direct_original_C_cb59.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                    "research/calibri/direct_kernel_v59.c","-o",str(lib)],
                   check=True)
    v2=NRA3NativePackedCPU(raw,NRA3Config(parallel_hypotheses=3,
                                         parallel_selector_version=2),lib).eval()
    require(v2.packed_matrix_count()==43 and
            v2.native_cpu_weights_dense_resident is False and
            not list(v2.parameters()),
            "S8b baseline loaded unaccounted dense FP32 area")
    x_list=[];y_list=[];pred_old=[];alignment_count=0
    with torch.inference_mode():
        for sample in samples:
            features=v2(sample.hidden320.unsqueeze(0))
            ix=torch.tensor([a.last_token for a in sample.aligned.word_alignment],
                            dtype=torch.long)
            evidence=features.parallel_router_features
            gate=features.parallel_hypothesis_gates
            require(evidence is not None and evidence.shape==
                    (1,len(sample.hidden320),D) and
                    gate is not None and gate.shape[-1]==K,
                    "native CB59 actual three-way feature/gate ABI")
            x_list.append(evidence[0,ix,:].detach().clone())
            y=weak_train_router_labels(sample.row,sample.aligned)
            require(len(y)==len(ix), "new 72 UD complete causal alignment drift")
            y_list.append(y)
            pred_old.extend(gate[0,ix,:].argmax(-1).tolist())
            alignment_count+=len(ix)
    x=torch.cat(x_list).float().contiguous()
    y=torch.cat(y_list).long().contiguous()
    require(x.shape==(alignment_count,D) and
            y.shape==(alignment_count,) and len(set(y.tolist()))==3 and
            bool(torch.isfinite(x).all()),"actual TRAIN72 3way evidence/teacher invalid")
    physical,fit=train_balanced_router(x,y,optimizer_steps=200,lr=.025)
    require(len(physical)==PHYSICAL_BYTES and
            fit["extra_physical_parameters"]==SOURCE_PARAMETERS==39 and
            hashlib.sha256(physical).hexdigest()==fit["extra_selector_SHA256"],
            "physically accounted SHA-locked FP16 tiny 3way weights invalid")
    saved=args.out/"NRA3_S11_TINY_PAID_RUSSIAN_TRAIN72_CAUSAL_3WAY_SELECTOR_FP16.n3r11"
    saved.write_bytes(physical)
    require(saved.read_bytes()==physical and
            PaidThreeWayRouter.from_physical(saved.read_bytes()).physical_bytes()==physical,
            "router source cannot survive physical file reload")
    with torch.inference_mode():
        v3=NRA3NativePackedCPU(
            raw,NRA3Config(parallel_hypotheses=3,parallel_selector_version=3),lib,
            paid_router_physical=saved.read_bytes()).eval()
        require(v3.packed_matrix_count()==43 and
                v3.source_sha256==before_sha and
                v3.native_cpu_weights_dense_resident is False and
                v3.output_gate==0 and v3.candidate_accepted is False and
                v3.paid_router is not None and
                v3.paid_router.physical_bytes()==physical,
                "genuine native C+paid tiny FP16 3way inference not physical")
        pred_new=[];all_gates=[];nan_checks=0
        for sample in samples:
            f=v3(sample.hidden320.unsqueeze(0))
            ix=torch.tensor([a.last_token for a in sample.aligned.word_alignment])
            gates=f.parallel_hypothesis_gates[0,ix,:]
            require(gates.shape==(len(ix),3) and
                    torch.allclose(gates.sum(-1),torch.ones(len(ix)),atol=2e-5) and
                    bool(torch.isfinite(f.relations).all()),
                    "loaded real physically paid 3way native relation corrupt")
            pred_new.extend(gates.argmax(-1).tolist())
            all_gates.append(gates.detach().clone())
            nan_checks+=1
        actual_gates=torch.cat(all_gates)
        true_train=y.tolist()
        actual_train_accuracy=sum(a==b for a,b in zip(pred_new,true_train))/len(y)
        old_train_accuracy=sum(a==b for a,b in zip(pred_old,true_train))/len(y)
        actual_train_ce=float(torch.nn.functional.nll_loss(
            actual_gates.clamp_min(1e-8).log(),y))
        v2_ms=_bench(v2,samples[0].hidden320)
        v3_ms=_bench(v3,samples[0].hidden320)
    require(nan_checks==72 and raw==args.s8_cb59.read_bytes(),
            "old 6M original physically trained 43 CB59 operators were changed")
    hard_old={BRANCHES[i]:pred_old.count(i) for i in range(3)}
    hard_new={BRANCHES[i]:pred_new.count(i) for i in range(3)}
    teacher={BRANCHES[i]:true_train.count(i) for i in range(3)}
    # The S8 collapsed hypothesis should not be mislabeled as an
    # improvement. A gate branch can be used but still be semantically
    # wrong. Any acceptance requires independent preregistered EVAL.
    all_branches_used=min(hard_new.values())>=max(3,int(len(y)*.05))
    enough_train_teacher=(actual_train_accuracy>old_train_accuracy+.03)
    result={
        "stage":"NRA3_S11_REAL_43M_RUSSIAN_UD_TRAIN72_PHYSICALLY_PAID_39_FP16_NATIVE_ROUTER_RESEARCH",
        "S10_original_frozen_43M_physical_run":S10_CAPTURE_RUN,
        "S10_real_43M_post_GT_hidden320_sha256":S10_FEATURE_SHA,
        "S8b_original_paid_6M_run":S8b_RUN,
        "S8b_original_paid_CB59_sha256":before_sha,
        "S8b_original_CB59_bytes_unchanged":len(raw),
        "S8b_original_43_CB59_5bit_packed_matrices_reused_unchanged":True,
        "S8b_original_NRA3_unique_parameters":6_106_154,
        "S11_aux_selector_unique_parameters":SOURCE_PARAMETERS,
        "S11_total_unique_area_plus_router_parameters":6_106_154+SOURCE_PARAMETERS,
        "S11_router_FP16_physical_file":saved.name,
        "S11_router_FP16_physical_sha256":hashlib.sha256(physical).hexdigest(),
        "S11_router_FP16_physical_bytes":len(physical),
        "S11_raw_CB59_plus_router_physically_stored_bytes":len(raw)+len(physical),
        "S11_raw_CB59_plus_router_physical_bpp":8*(len(raw)+len(physical))/(6_106_154+SOURCE_PARAMETERS),
        "S11_uses_small_3x12_FP16_dense_router_physically_billed":True,
        "S11_every_original_6M_matrix_uses_same_native_C_CB59":True,
        "S11_router_in_the_six_real_relation_GRU_cycles":True,
        "S11_evidence_features_causal_no_gold_in_forward":True,
        "S11_no_original_Calibri_or_GT_weight_updates":True,
        "S11_no_original_NRA3_6M_weight_updates":True,
        "S11_only_new_train72_teacher_is_weak_not_semantic_truth":True,
        "S11_new_TRUE_43M_TRAIN72_rows":len(samples),
        "S11_new_Russian_TRAIN72_total_words":len(y),
        "S11_weak_heuristic_teacher_counts":teacher,
        "S8b_hard_router_prediction_COUNTS_on_NEW_train72":hard_old,
        "S11_loaded_native_hard_router_prediction_COUNTS_on_NEW_train72":hard_new,
        "S8b_existing_native_weak_train_accuracy_on_NEW_train72":old_train_accuracy,
        "S11_loaded_native_weak_train_accuracy_on_NEW_train72":actual_train_accuracy,
        "S11_loaded_native_weak_train_ce_on_NEW_train72":actual_train_ce,
        "S11_route_use_all_three_over_5_percent":all_branches_used,
        "S11_native_weak_train_accuracy_gain_over_S8b_3pp":enough_train_teacher,
        "S11_train_router_physically_snapped_reloaded":True,
        "S11_only_NRA_component_S8b_median_ms":v2_ms,
        "S11_only_NRA_component_router_median_ms":v3_ms,
        "S11_microbenchmark_not_whole_43M_CPU":True,
        "independent_true_Russian_generalization_eval_measured":False,
        "new_independent_Russian_EVAL_sealed_before_train":False,
        "full_answer_EOS_candidate_quality_measured":False,
        "real_full_parent_43M_plus_GT_plus_NRA_CPU_speed_certified":False,
        "all_parent_GT_NRA_physical_bpp_below_point7_certified":False,
        "parent_616_J7_from_S10_remains_original_frozen_baseline":True,
        "protected_616_J7_with_S11_active_generation_performed":False,
        "old_five_consumed_EVAL128_not_loaded":True,
        "sixth_reserved_S3CI_EVAL128_not_loaded":True,
        "no_LIVE_Render_Supabase_APK_or_Jessica_chat":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
        **fit,
    }
    report=args.out/"NRA3_S11_PAID_3WAY_SELECTOR_REAL43M_TRAIN72_NO_PROMOTION.json"
    report.write_text(json.dumps(result,ensure_ascii=False,sort_keys=True,
                                 indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        k:v for k,v in result.items()
        if k not in ("router_loss_trace",)
    },ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
