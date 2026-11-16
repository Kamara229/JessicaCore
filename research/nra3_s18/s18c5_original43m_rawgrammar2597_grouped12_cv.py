"""S18c5: 12 previously frozen TRAIN-internal lexical/syntax group CV folds.

Compare research S18c4's paid 2,597-FP16 Russian raw-text morphology
+ original native43-C 128d role head against previously executed S18c2's
2,309-FP16 head on IDENTICAL original NEW96 TRAIN and 12 fixed splits.
Each fold: new rank8 head initialized from scratch, 150 FIXED AdamW steps
on TRAIN-side pairs only, physical FP16 bytes reloaded, then heldout
scored exactly once. No heldout monitoring/checkpoint/parameter selection.
The predeclared original TRAIN-internal folds are NOT a pristine EVAL
(the grammar feature architecture was proposed after inspecting them).
Never claim independent language generalization or activate LIVE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch.nn import functional as F

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s13.real43m_train72_readouts import (
    S8B_43_CB59_SHA, S11_39_FP16_SHA,
)
from research.nra3_s16.real43m_train72_directed_valence import metrics
from research.nra3_s18.real43m_train96_physical_source import (
    S18B_TRUE_CPU_RUN, checked_real_train96, grouped_train96_split_contract,
)
from research.nra3_s18.ru96_multiverb_train_source import (
    S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256, CONSTRUCTIONS,
)
from research.nra3_s18.s18c1_original43m_train96_nativeC_rank8 import (
    native_pair_train96, TOTAL_PAIRS, TOTAL_ROLES,
)
from research.nra3_s18.s18c2_original43m_train96_grouped12_cv import (
    select_pairs, aggregate_folds,
)
from research.nra3_s18.s18c4_paid_rawgrammar_rank8 import (
    PaidRawGrammarDirectedValence, PARAMETERS, PHYSICAL_BYTES, DIM,
    EXTRA_PARAMETERS,
)
from research.nra3_s18.s18c4_original43m_rawgrammar_new96_train import (
    create_only_raw_text_pair_features, score,
)

STEPS=150
LEARNING_RATE=.025
SEED=318018
WEIGHTS=torch.tensor([.20,1.,1.,1.,1.5],dtype=torch.float32)
FIXED_S18C2_RUN=43914384102
FIXED_S18C4_RUN=43916875445
EXPECTED_BASE_S18C2_LEXICAL_F1=0.3670411985018726
EXPECTED_BASE_S18C2_SYNTAX_F1=0.338785046728972


def require(ok, msg):
    if not bool(ok):
        raise ValueError("S18c5 TRAIN-internal 12-fold raw grammar original native CPU: "+msg)


def subset(x, rawp, rawa, rowids):
    keep=torch.zeros(96, dtype=torch.bool)
    require(len(rowids)>0 and all(type(v) is int and 0<=v<96 for v in rowids)
            and len(set(rowids))==len(rowids),
            "fold IDs cannot contain duplicate or out-of-range source")
    keep[list(rowids)]=True
    selected=keep[x.source_index]
    x_slice=select_pairs(x,rowids)
    gp=rawp[selected].detach().clone().contiguous()
    ga=rawa[selected].detach().clone().contiguous()
    require(x_slice.total_pairs==gp.shape[0]==ga.shape[0]
            and gp.shape[1]==ga.shape[1]==DIM
            and torch.equal(x_slice.source_index,x.source_index[selected]),
            "raw-only model inputs or physical ordered wordpair split misaligned")
    return x_slice,gp,ga


def physical_rounded_metrics(model,x,gp,ga):
    physical=model.physical_bytes()
    frozen=PaidRawGrammarDirectedValence.from_physical(physical).eval()
    require(len(physical)==PHYSICAL_BYTES==5286
            and frozen.physical_bytes()==physical
            and all(not p.requires_grad for p in frozen.parameters())
            and frozen.output_gate==0 and frozen.candidate_accepted is False,
            "all 2597 FP16 values must round-trip physically before scoring")
    with torch.inference_mode():
        measured=metrics(score(frozen,x,gp,ga),x)
    return measured,physical


def one_fixed_fold(x,gp,ga,fold,out:Path):
    kind,ident,held_groups,train_rows,held_rows=fold
    require(kind in ("lexical","syntax")
            and ident in tuple(f"lex_{k}" for k in range(4))+tuple(
                f"syntax_{k}" for k in range(8))
            and set(train_rows).isdisjoint(held_rows)
            and sorted((*train_rows,*held_rows))==list(range(96)),
            "precommitted source fold independence violated")
    train,trainp,traina=subset(x,gp,ga,train_rows)
    held,heldp,helda=subset(x,gp,ga,held_rows)
    require(train.total_pairs+held.total_pairs==x.total_pairs==3288
            and train.positive_links+held.positive_links==x.positive_links==300
            and set(train.source_index.tolist()).isdisjoint(
                set(held.source_index.tolist()))
            and train.rows+held.rows==96,
            "model input/source split includes forbidden validation wordpairs")
    torch.manual_seed(SEED)
    model=PaidRawGrammarDirectedValence().eval()
    require(sum(p.numel() for p in model.parameters())==PARAMETERS==2597
            and EXTRA_PARAMETERS==288
            and model.output_gate==0 and model.candidate_accepted is False,
            "paid raw morphology rank8 fields changed or gate opened")
    opt=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,
                          weight_decay=.0002)
    # Crucially: NO heldout forward, loss, checkpoint selection or data access
    # inside optimizer; all 150 steps fixed BEFORE even one heldout score.
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        logits=score(model,train,trainp,traina)
        loss=F.cross_entropy(logits,train.target,weight=WEIGHTS)
        loss=loss+1e-5*sum(p.square().mean() for p in model.parameters())
        require(bool(torch.isfinite(loss)), "TRAIN-side loss nonfinite")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "paid original rank8 or grammar projection missing gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    train_metrics,physical=physical_rounded_metrics(
        model,train,trainp,traina)
    hold_metrics,again=physical_rounded_metrics(model,held,heldp,helda)
    require(physical==again,
            "heldout evaluation must not alter FP16 parameter bytes")
    path=out/(ident+".n3s18c4")
    require(not path.exists(), "old heldout FP16 checkpoint cannot be overwritten")
    path.write_bytes(physical)
    require(path.read_bytes()==physical, "fold physical artifact write mismatch")
    a=train_metrics["all_two_direction_wordpair_exact_role"]
    b=hold_metrics["all_two_direction_wordpair_exact_role"]
    return {
        "fold_id":ident,
        "kind":"TRAIN_INTERNAL_"+kind.upper(),
        "heldout_lexical_or_syntactic_groups":list(held_groups),
        "train_sentence_count":train.rows,
        "heldout_sentence_count":held.rows,
        "train_all_directed_candidate_pairs":train.total_pairs,
        "heldout_all_directed_candidate_pairs":held.total_pairs,
        "train_positive_directed_role_gold":train.positive_links,
        "heldout_positive_directed_role_gold":held.positive_links,
        "optimizer_steps_exact":STEPS,
        "fold_from_scratch":True,
        "selected_checkpoint_using_heldout":False,
        "heldout_features_never_entered_optimizer":True,
        "no_gold_UPOS_UD_lemma_role_entered_raw_grammar_model_features":True,
        "all_negative_pairs_included":True,
        "final_train_only_F1":a["f1"],
        "final_holdout_TRAIN_internal_F1":b["f1"],
        "final_holdout_TRAIN_internal_P":b["precision"],
        "final_holdout_TRAIN_internal_R":b["recall"],
        "final_holdout_TRAIN_internal_TP":b["exact_directed_role_true_positive"],
        "final_holdout_TRAIN_internal_FP":b["false_positive"],
        "final_holdout_TRAIN_internal_FN":b["false_negative"],
        "per_role_heldout":hold_metrics["per_role"],
        "by_orientation_heldout":hold_metrics["by_surface_orientation"],
        "physical_fp16_filename":path.name,
        "physical_fp16_sha256":hashlib.sha256(physical).hexdigest(),
        "physical_fp16_bytes":len(physical),
    }


def authentic_baseline(path:Path):
    baseline=json.loads(path.read_text(encoding="utf-8"))
    require(baseline["stage"]==
            "NRA3_S18C2_TRUE_ORIGINAL43M_GT_NATIVE43_S11_NEW96_GROUPED12_TRAIN_INTERNAL_CV"
            and baseline["original_S18b_cpu_run"]==S18B_TRUE_CPU_RUN
            and baseline["original_43M_GT_CB59_S11_all_frozen"]
            and baseline["all_300_gold_positive_role_edges"]==300
            and baseline["lexical_heldout_FOLDS"]==4
            and baseline["syntax_heldout_FOLDS"]==8,
            "original S18c2 TRAIN-internal comparison run mismatch")
    m=baseline["micro_grouped_metrics"]
    require(abs(m["TRAIN_INTERNAL_LEXICAL"]["micro_F1"]-
                EXPECTED_BASE_S18C2_LEXICAL_F1)<1e-12
            and abs(m["TRAIN_INTERNAL_SYNTAX"]["micro_F1"]-
                    EXPECTED_BASE_S18C2_SYNTAX_F1)<1e-12,
            "prior original 2309 rank8 12-fold baseline frozen F1 changed")
    old=baseline["individual_fold_results"]
    require(len(old)==12 and
            set(z["fold_id"] for z in old)=={
                f"lex_{k}" for k in range(4)} | {
                f"syntax_{k}" for k in range(8)},
            "prior 12 fold IDs differ from immutable S18c0 groups")
    return baseline


def main():
    parser=argparse.ArgumentParser()
    for name in ("s18b-hidden","s18b-report","s8-cb59","s11-router",
                 "s18c2-baseline-report","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    require(not a.out.exists(),"never overwrite earlier physical frozen results")
    samples,original_s18b=checked_real_train96(a.s18b_hidden,a.s18b_report)
    old=authentic_baseline(a.s18c2_baseline_report)
    require(S18B_TRUE_CPU_RUN==43910479658
            and original_s18b["all_616_native_CPU_generated_token_digests_unchanged"]
            and S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256==
                "abde63c34534d90e00a6ac61808d2453ce0cfc966c83d45377c1fb620999cca9",
            "original 96 TRAIN Calibri source/616 J7 provenance drift")
    packed=a.s8_cb59.read_bytes()
    router=a.s11_router.read_bytes()
    require(len(packed)==653798
            and hashlib.sha256(packed).hexdigest()==S8B_43_CB59_SHA
            and len(CB59(packed).handles)==43
            and CB59(packed).meta["original_unique_parameter_count"]==6_106_154
            and len(router)==123
            and hashlib.sha256(router).hexdigest()==S11_39_FP16_SHA
            and PaidThreeWayRouter.from_physical(router).physical_bytes()==router,
            "original packed native-C or frozen S11 source SHA changed")
    folds=grouped_train96_split_contract(tuple(row.row for row in samples))
    require(len(folds)==12 and len([f for f in folds if f[0]=="lexical"])==4
            and len([f for f in folds if f[0]=="syntax"])==8,
            "exactly 4 lexical and 8 syntax groups required")
    a.out.mkdir(parents=True,exist_ok=False)
    library=a.out/"s18c5_original43_CB59_C.so"
    subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                   "research/calibri/direct_kernel_v59.c","-o",str(library)],check=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    native=NRA3NativePackedCPU(
        packed,NRA3Config(parallel_hypotheses=3,
            parallel_selector_version=3),library,
        paid_router_physical=router).eval()
    require(native.source_sha256==S8B_43_CB59_SHA
            and native.paid_router.physical_bytes()==router
            and not native.native_cpu_weights_dense_resident
            and native.output_gate==0 and native.candidate_accepted is False
            and all(not v.requires_grad for v in native.parameters()),
            "original native-C frozen43 original S11 role gate not immutable")
    pairs,relation_sha=native_pair_train96(samples,native)
    require(pairs.rows==96 and pairs.total_pairs==TOTAL_PAIRS==3288
            and pairs.positive_links==TOTAL_ROLES==300
            and len(relation_sha)==96,
            "original frozen native C 96 source rows all directed candidate pairs missing")
    gp,ga=create_only_raw_text_pair_features(samples,pairs)
    records=[]
    for idx,fold in enumerate(folds,1):
        result=one_fixed_fold(pairs,gp,ga,fold,a.out)
        records.append(result)
        print(json.dumps({
            "S18c5_fold":idx,"fold_id":result["fold_id"],
            "train_only_F1":result["final_train_only_F1"],
            "heldout_TRAIN_internal_F1":result["final_holdout_TRAIN_internal_F1"],
            "paid_FP16_SHA256":result["physical_fp16_sha256"],
        },sort_keys=True),flush=True)
    lexical=[z for z in records if z["kind"]=="TRAIN_INTERNAL_LEXICAL"]
    syntax=[z for z in records if z["kind"]=="TRAIN_INTERNAL_SYNTAX"]
    require(len(lexical)==4 and len(syntax)==8
            and sum(z["heldout_sentence_count"] for z in lexical)==96
            and sum(z["heldout_sentence_count"] for z in syntax)==96
            and sum(z["heldout_all_directed_candidate_pairs"] for z in lexical)==3288
            and sum(z["heldout_all_directed_candidate_pairs"] for z in syntax)==3288,
            "old controlled source groups and pair totals must cover all 96 once per axis")
    grouped={
        "TRAIN_INTERNAL_LEXICAL":aggregate_folds(lexical,300),
        "TRAIN_INTERNAL_SYNTAX":aggregate_folds(syntax,300),
    }
    compare={}
    for axis in ("TRAIN_INTERNAL_LEXICAL","TRAIN_INTERNAL_SYNTAX"):
        new=grouped[axis]
        before=old["micro_grouped_metrics"][axis]
        compare[axis]={
            "older_S18c2_2309FP16_MICRO_F1":before["micro_F1"],
            "new_S18c5_2597FP16_MICRO_F1":new["micro_F1"],
            "delta_MICRO_F1_new_minus_baseline":new["micro_F1"]-before["micro_F1"],
            "older_MICRO_precision":before["micro_precision"],
            "new_MICRO_precision":new["micro_precision"],
            "older_MICRO_recall":before["micro_recall"],
            "new_MICRO_recall":new["micro_recall"],
            "better_than_S18c2_TRAIN_internal":new["micro_F1"]>before["micro_F1"],
        }
    previous={z["fold_id"]:z for z in old["individual_fold_results"]}
    named={}
    for mode,foldid in ((CONSTRUCTIONS[2],"syntax_2"),
                        (CONSTRUCTIONS[3],"syntax_3")):
        cur=next(z for z in records if z["fold_id"]==foldid)
        prior=previous[foldid]
        require(cur["heldout_lexical_or_syntactic_groups"]==[mode]
                and prior["heldout_lexical_or_syntactic_groups"]==[mode],
                "old and new passive/recipient groups must be identical")
        named[mode]={
            "old_S18c2_F1":prior["final_holdout_TRAIN_internal_F1"],
            "new_S18c5_F1":cur["final_holdout_TRAIN_internal_F1"],
            "old_S18c2_P":prior["final_holdout_TRAIN_internal_P"],
            "new_S18c5_P":cur["final_holdout_TRAIN_internal_P"],
            "old_S18c2_R":prior["final_holdout_TRAIN_internal_R"],
            "new_S18c5_R":cur["final_holdout_TRAIN_internal_R"],
            "new_S18c5_per_role_heldout":cur["per_role_heldout"],
        }
    summary={
        "stage":"NRA3_S18C5_TRUE43M_GT_NATIVE43_S11_NEW96_RAWGRAMMAR2597_GROUPED12_TRAIN_INTERNAL_CV",
        "S18b_original43M_gt_true_CPU_train96_run":S18B_TRUE_CPU_RUN,
        "S18b_original43M_FP16_hidden320_SHA":original_s18b["actual_fp16_feature_sha256"],
        "S18c2_frozen_prior_plain_rank8_cv_run":FIXED_S18C2_RUN,
        "S18c4_physical_train96_grammar_paid2597_run":FIXED_S18C4_RUN,
        "original_616_J7_unchanged_only_for_frozen_parent":True,
        "original43M_GT_native43_CB59_S11_unchanged":True,
        "original43M_native_C_96_full_pairs_executed":True,
        "original_CB59_native43_SHA":S8B_43_CB59_SHA,
        "original_S11_39FP16_SHA":S11_39_FP16_SHA,
        "S18_96_TRAIN_full_source_SHA":S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,
        "original_native_C_per_sentence_relation_SHA":relation_sha,
        "all_bidirectional_candidate_pairs_unfiltered":pairs.total_pairs,
        "exact_300_directed_gold_roles_train96":pairs.positive_links,
        "fixed_12_group_folds_from_S18c0":True,
        "lexical_heldout_folds":4,
        "syntax_heldout_folds":8,
        "each_fold_from_scratch_fixed_optimizer_steps":STEPS,
        "optimizer_selected_checkpoint_using_heldout":False,
        "true_raw_text_18_indicator_words_only_no_oracle_UPOS_UD_lemma_gold_role":True,
        "new_model_2597_paid_FP16_physical_bytes_per_fold":PHYSICAL_BYTES,
        "extra_paid_288_FP16_grammar_projection_params":EXTRA_PARAMETERS,
        "each_fold_physical_source_SHA_verified":True,
        "S18c4_100pct_full_TRAIN_F1_NOT_GENERALIZATION":True,
        "train_internal_holdouts_seen_during_prior_architecture_design":True,
        "this_grouped_cv_NOT_pristine_independent_EVAL":True,
        "no_old_S17c_S15c_S12c_consumed_EVALs_loaded":True,
        "new_candidate_active_616_J7_EOS_not_tested":True,
        "strict_online_causal_Russian_word_parser_unproven":True,
        "inclusive_bpp_le07_NOT_CERTIFIED":True,
        "whole_CPU_ge90pct_CB59_NOT_CERTIFIED":True,
        "output_gate":0,"candidate_accepted":False,
        "activation_allowed":False,
        "LIVE_Jessica_Render_Supabase_APK_unchanged":True,
        "micro_grouped_metrics":grouped,
        "comparison_to_S18c2_TRAIN_internal":compare,
        "heldout_passive_and_recipient_subject_comparison":named,
        "individual_fold_results":records,
    }
    output=a.out/"NRA3_S18C5_ORIGINAL43M_RAWGRAMMAR2597_NEW96_CV12_F1_REPORT.json"
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items()
        if k not in ("individual_fold_results","original_native_C_per_sentence_relation_SHA")},
         ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
