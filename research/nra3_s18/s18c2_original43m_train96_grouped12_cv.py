"""S18c2: strict 12-fold TRAIN-internal Russian group CV, no spent EVAL.

S18c1 genuine original 43M+Green Tunnel CPU 96 TRAIN hidden320,
43 frozen packed CB59 5-bit C matrices + 39 FP16 S11 selector.
4 lexical-family-disjoint folds and 8 construction-disjoint folds,
declared by S18c0 before this training. Each fold initializes FROM
SCRATCH, runs exactly 150 fixed AdamW steps on its own TRAIN rows
and evaluates heldout only after exact physical FP16 reload.
Never train on S17c/S15c/S12c/old EVAL128, never select by heldout.
Only original 2309 FP16 new rank8 role parameters can be updated.
This is TRAIN-internal group transfer, NOT independent generalization.
No LIVE, output gate stays zero; full-sentence S95 means strict
online causality has NOT been established.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s13.real43m_train72_readouts import (
    S8B_43_CB59_SHA, S11_39_FP16_SHA,
)
from research.nra3_s16.paid_directed_valence import (
    PaidDirectedValence, PARAMETERS, PHYSICAL_BYTES, ROLES,
)
from research.nra3_s16.real43m_train72_directed_valence import (
    DirectedPairTrain, frozen_paired_logits, metrics, score_weighted_train,
)
from research.nra3_s18.real43m_train96_physical_source import (
    S18B_TRUE_CPU_RUN, checked_real_train96, grouped_train96_split_contract,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
)
from research.nra3_s18.s18c1_original43m_train96_nativeC_rank8 import (
    native_pair_train96, TOTAL_PAIRS, TOTAL_ROLES,
)

STEPS=150
LEARNING_RATE=.025
SEED=318018
WEIGHTS=torch.tensor([.20,1.,1.,1.,1.5],dtype=torch.float32)


def require(ok,msg):
    if not bool(ok):
        raise ValueError("S18c2 only NEW96 original native-C GROUPED TRAIN internal: "+msg)


def select_pairs(x:DirectedPairTrain,indices):
    require(len(indices)>0 and len(set(indices))==len(indices)
            and all(type(i) is int and 0<=i<96 for i in indices),
            "unknown or duplicate TRAIN96 row in group split")
    keep=torch.zeros(96,dtype=torch.bool)
    keep[list(indices)]=True
    select=keep[x.source_index]
    sl=lambda t:t[select].detach().clone().contiguous()
    target=sl(x.target)
    return DirectedPairTrain(
        sl(x.predicate),sl(x.argument),
        sl(x.predicate_positions),sl(x.argument_positions),
        sl(x.current_positions),target,sl(x.orientation),sl(x.source_index),
        len(indices),int((target!=0).sum()))


def rounded_exact(model,x):
    """All physical FP16 bytes must be reloaded before reporting metrics."""
    raw=model.physical_bytes()
    loaded=PaidDirectedValence.from_physical(raw).eval()
    require(len(raw)==PHYSICAL_BYTES==4665 and loaded.physical_bytes()==raw
            and all(not p.requires_grad for p in loaded.parameters()),
            "physical charged 2309 FP16 rank8 operator reload mismatch")
    with torch.inference_mode():
        scored=metrics(frozen_paired_logits(loaded,x),x)
    return scored,raw


def one_fold(x, fold, out:Path):
    kind,fold_id,held_groups, train_rows,held_rows=fold
    require(kind in ("lexical","syntax") and train_rows and held_rows
            and set(train_rows).isdisjoint(held_rows)
            and sorted((*train_rows,*held_rows))==list(range(96)),
            "train vs heldout source identifiers overlap")
    train=select_pairs(x,train_rows)
    hold=select_pairs(x,held_rows)
    require(train.total_pairs+hold.total_pairs==x.total_pairs
            and train.positive_links+hold.positive_links==x.positive_links
            and set(train.source_index.tolist()).isdisjoint(hold.source_index.tolist())
            and train.rows+hold.rows==96,
            "heldout row or gold leaked into physical TRAIN subset")
    torch.manual_seed(SEED)
    model=PaidDirectedValence().eval()
    require(sum(p.numel() for p in model.parameters())==PARAMETERS==2309
            and model.output_gate==0 and model.candidate_accepted is False,
            "trainable model shape or safety gate drift")
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,weight_decay=.0002)
    for step in range(1,STEPS+1):
        opt.zero_grad(set_to_none=True)
        loss=score_weighted_train(model,train,WEIGHTS)
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "physical paid rank8 input gradients absent/nonfinite")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    # No validation query prior to ALL 150 optimizer steps.
    final_train,physical=rounded_exact(model,train)
    final_hold,second=rounded_exact(model,hold)
    require(physical==second, "heldout score changed model bytes")
    dest=out/(fold_id+".n3s16")
    require(not dest.exists(), "never overwrite frozen fold physical parameters")
    dest.write_bytes(physical)
    require(dest.read_bytes()==physical, "FP16 fold source physical file drift")
    a=final_train["all_two_direction_wordpair_exact_role"]
    b=final_hold["all_two_direction_wordpair_exact_role"]
    return {
        "fold_id":fold_id,
        "kind":"TRAIN_INTERNAL_"+kind.upper(),
        "heldout_lexical_or_syntactic_groups":list(held_groups),
        "train_sentence_count":train.rows,
        "heldout_sentence_count":hold.rows,
        "train_all_directed_candidate_pairs":train.total_pairs,
        "heldout_all_directed_candidate_pairs":hold.total_pairs,
        "train_positive_directed_role_gold":train.positive_links,
        "heldout_positive_directed_role_gold":hold.positive_links,
        "optimizer_steps_exact":STEPS,
        "fold_from_scratch":True,
        "selected_checkpoint_using_heldout":False,
        "heldout_features_never_entered_optimizer":True,
        "no_oracle_POS_or_gold_positive_pair_filtering":True,
        "final_train_only_F1":a["f1"],
        "final_holdout_TRAIN_internal_F1":b["f1"],
        "final_holdout_TRAIN_internal_P":b["precision"],
        "final_holdout_TRAIN_internal_R":b["recall"],
        "final_holdout_TRAIN_internal_TP":b["exact_directed_role_true_positive"],
        "final_holdout_TRAIN_internal_FP":b["false_positive"],
        "final_holdout_TRAIN_internal_FN":b["false_negative"],
        "per_role_heldout":final_hold["per_role"],
        "by_orientation_heldout":final_hold["by_surface_orientation"],
        "physical_fp16_filename":dest.name,
        "physical_fp16_sha256":hashlib.sha256(physical).hexdigest(),
        "physical_fp16_bytes":len(physical),
    }


def aggregate_folds(folds,expected_count):
    tp=sum(f["final_holdout_TRAIN_internal_TP"] for f in folds)
    fp=sum(f["final_holdout_TRAIN_internal_FP"] for f in folds)
    fn=sum(f["final_holdout_TRAIN_internal_FN"] for f in folds)
    require(tp+fn==expected_count==300
            and sum(f["heldout_positive_directed_role_gold"] for f in folds)==expected_count,
            "heldout micro F1 axis must cover all 300 unique true roles ONCE")
    p=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    f1=2*p*recall/(p+recall) if p+recall else 0.
    return {
        "folds":len(folds),
        "all_300_gold_roles_once_per_axis":expected_count,
        "micro_TP":tp,"micro_FP":fp,"micro_FN":fn,
        "micro_precision":p,"micro_recall":recall,"micro_F1":f1,
        "fold_F1":[f["final_holdout_TRAIN_internal_F1"] for f in folds],
    }


def main():
    p=argparse.ArgumentParser()
    for arg in ("s18b-hidden","s18b-report","s8-cb59","s11-router","out"):
        p.add_argument("--"+arg,type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists(), "never overwrite old grouped CV result")
    samples,s18b=checked_real_train96(args.s18b_hidden,args.s18b_report)
    require(S18B_TRUE_CPU_RUN==43910479658
            and s18b["all_616_native_CPU_generated_token_digests_unchanged"]
            and S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256==
                "abde63c34534d90e00a6ac61808d2453ce0cfc966c83d45377c1fb620999cca9",
            "original physical source 616 J7 or locked gold identity drift")
    original=args.s8_cb59.read_bytes()
    router=args.s11_router.read_bytes()
    require(len(original)==653798 and hashlib.sha256(original).hexdigest()==S8B_43_CB59_SHA
            and len(CB59(original).handles)==43 and CB59(original).meta[
                "original_unique_parameter_count"]==6_106_154,
            "original 43 packed five-bit native-C weights modified")
    require(len(router)==123 and hashlib.sha256(router).hexdigest()==S11_39_FP16_SHA
            and PaidThreeWayRouter.from_physical(router).physical_bytes()==router,
            "immutable S11 39FP16 parallel selector changed")
    folds=grouped_train96_split_contract(tuple(s.row for s in samples))
    require(len(folds)==12 and len([f for f in folds if f[0]=="lexical"])==4
            and len([f for f in folds if f[0]=="syntax"])==8,
            "precommitted 12 folds missing")
    args.out.mkdir(parents=True,exist_ok=False)
    so=args.out/"s18c2_original_43_CB59_native.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
        "research/calibri/direct_kernel_v59.c","-o",str(so)],check=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    native=NRA3NativePackedCPU(
        original,NRA3Config(parallel_hypotheses=3,
            parallel_selector_version=3),so,paid_router_physical=router).eval()
    require(native.source_sha256==S8B_43_CB59_SHA
            and native.paid_router.physical_bytes()==router
            and not native.native_cpu_weights_dense_resident
            and native.output_gate==0 and native.candidate_accepted is False
            and all(not q.requires_grad for q in native.parameters()),
            "frozen original 43 CB59 native-C/39FP16 violated")
    x,shas=native_pair_train96(samples,native)
    require(x.rows==96 and x.positive_links==TOTAL_ROLES==300
            and x.total_pairs==TOTAL_PAIRS==3288 and len(shas)==96,
            "all 96 physical native-C rows / 300 linked labels / ordered pairs invalid")
    report=[]
    for k,fold in enumerate(folds,1):
        f=one_fold(x,fold,args.out)
        report.append(f)
        print(json.dumps({
            "fold_number":k,"fold_id":f["fold_id"],
            "train_only_F1":f["final_train_only_F1"],
            "heldout_TRAIN_internal_F1":f["final_holdout_TRAIN_internal_F1"],
            "model_fp16_SHA256":f["physical_fp16_sha256"],
        },sort_keys=True),flush=True)
    lex=[f for f in report if f["kind"]=="TRAIN_INTERNAL_LEXICAL"]
    syn=[f for f in report if f["kind"]=="TRAIN_INTERNAL_SYNTAX"]
    require(len(lex)==4 and len(syn)==8
            and sum(f["heldout_sentence_count"] for f in lex)==96
            and sum(f["heldout_sentence_count"] for f in syn)==96,
            "one per axis: all 96 original TRAIN rows must be held out exactly once")
    aggregate={
        "TRAIN_INTERNAL_LEXICAL":aggregate_folds(lex,300),
        "TRAIN_INTERNAL_SYNTAX":aggregate_folds(syn,300),
    }
    result={
        "stage":"NRA3_S18C2_TRUE_ORIGINAL43M_GT_NATIVE43_S11_NEW96_GROUPED12_TRAIN_INTERNAL_CV",
        "original_S18b_43M_postGT_hidden_FP16_sha":s18b["actual_fp16_feature_sha256"],
        "original_S18b_cpu_run":S18B_TRUE_CPU_RUN,
        "original_616_J7_generated_tokens_unchanged":True,
        "original_Calibri43M_GT_unmodified":True,
        "original_CB59_43_packed_5bit_C_SHA":S8B_43_CB59_SHA,
        "original_S11_39_FP16_SHA":S11_39_FP16_SHA,
        "original_S18_96_TRAIN_source_SHA":S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
        "original_S18c1_FULL_TRAIN_run":43913448392,
        "native_C_original_all_96_true43M_rows":True,
        "all_bidirectional_unfiltered_wordpair_candidates":x.total_pairs,
        "all_300_gold_positive_role_edges":x.positive_links,
        "lexical_heldout_FOLDS":4,
        "syntax_heldout_FOLDS":8,
        "exact_optimizer_steps_per_fold":STEPS,
        "from_scratch_each_fold":True,
        "no_heldout_checkpoint_selection":True,
        "new_rank8_2309_physically_paid_FP16_only":True,
        "new_rank8_FP16_physical_bytes_per_fold":PHYSICAL_BYTES,
        "S95_full_prompt_context_strict_online_word_causality_not_proven":True,
        "TRAIN_INTERNAL_CV_not_new_independent_EVAL":True,
        "no_any_consumed_S17c_S15c_S12c_EVAL_or_EVAL128_used":True,
        "original_43M_GT_CB59_S11_all_frozen":True,
        "new_candidate_active_616_J7_EOS_not_tested":True,
        "all_inclusive_bpp_at_most_07_not_certified":True,
        "whole_CPU_speed_at_least_90pct_CB59_not_certified":True,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
        "LIVE_Jessica_Render_Supabase_APK_unchanged":True,
        "micro_grouped_metrics":aggregate,
        "individual_fold_results":report,
    }
    dest=args.out/"NRA3_S18C2_NEW96_ORIGINAL43M_NATIVE43_CV12_F1_REPORT.json"
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="individual_fold_results"},
                     sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
