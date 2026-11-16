"""NRA3 S18a: REAL Calibri43M+GT frozen native CPU TRAIN72 grouped CV.

S17c (#43902361580) independently rejected S16b direct-role candidate:
on 24 NEW already-CONSUMED Russian sentences, hard directed link F1
15.93% vs 86.60% on 72 TRAIN, so NO RETUNING from those EVAL24.
This file imports none of S17 or prior S12c/S15c EVAL data. It is
scientific TRAIN-internal diagnostic, NOT independent generalization.

Two PREDECLARED GROUPED CONTROL schemes on exactly S9 old72 TRAIN:
  lexical: four folds holding out 3 entire lexical families per fold;
  syntax: six folds holding out one entire construction per fold.
Each fold trains FROM SCRATCH on subset only, 150 FIXED AdamW steps,
no validation-target updates or checkpoint selection, then reloads
all 2,309 original S16 FP16 physical parameters from exact bytes.
Every pair including negatives and both directions is included.
NO model activation and no original checkpoint writes.
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
from research.nra3_s9.new_ru_train72 import build_s9_train72,TEMPLATES,S9_TRAIN72_CANONICAL_SHA256
from research.nra3_s11.paid_router import PaidThreeWayRouter
from research.nra3_s11.real_ru72_source import checked_real_train72,S10_FEATURE_SHA,S10_CAPTURE_RUN
from research.nra3_s13.real43m_train72_readouts import S8B_43_CB59_SHA,S11_39_FP16_SHA
from research.nra3_s16.direct_train72_edges import S16_TRAIN_SOURCE_SHA,sealed_directed_train72
from research.nra3_s16.paid_directed_valence import PaidDirectedValence,PARAMETERS,PHYSICAL_BYTES,ROLES
from research.nra3_s16.real43m_train72_directed_valence import (
    DirectedPairTrain,build_all_paired_native_train72,
    frozen_paired_logits,metrics,score_weighted_train
)

STEPS=150
LEARNING_RATE=.025
WEIGHTS=torch.tensor([.20,1.,1.,1.,1.5],dtype=torch.float32)
LEXICAL_FOLDS=tuple(tuple(i for i in range(12) if i%4==k) for k in range(4))
SYNTAX_FOLDS=tuple((name,) for name in TEMPLATES)
EXPECTED_LEXICAL_FOLDS=4
EXPECTED_SYNTAX_FOLDS=6


def require(ok,msg):
    if not bool(ok):
        raise ValueError("NRA3 S18a strictly old72 TRAIN-grouped CV: "+msg)


def controlled_group_folds(rows=None):
    rows=build_s9_train72() if rows is None else rows
    require(len(rows)==72 and
            len({(r.lexical_family,r.construction) for r in rows})==72,
            "original 12x6 fixed TRAIN matrix required")
    folds=[]
    for k, families in enumerate(LEXICAL_FOLDS):
        train=tuple(i for i,r in enumerate(rows)
                    if r.lexical_family not in families)
        heldout=tuple(i for i,r in enumerate(rows)
                      if r.lexical_family in families)
        require(len(train)==54 and len(heldout)==18
                and set(train).isdisjoint(heldout)
                and {rows[i].lexical_family for i in train}.isdisjoint(
                    {rows[i].lexical_family for i in heldout})
                and len({rows[i].construction for i in heldout})==6,
                "lexical families must be completely isolated")
        folds.append({
           "id":f"lexical_family_mod4_{k}",
           "kind":"TRAIN_INTERNAL_LEXICAL",
           "heldout_groups":list(families),
           "train_rows":train,"holdout_rows":heldout
        })
    for c, (name,) in enumerate(SYNTAX_FOLDS):
        train=tuple(i for i,r in enumerate(rows) if r.construction!=name)
        heldout=tuple(i for i,r in enumerate(rows) if r.construction==name)
        require(len(train)==60 and len(heldout)==12
                and set(train).isdisjoint(heldout)
                and {rows[i].construction for i in train}.isdisjoint(
                    {rows[i].construction for i in heldout})
                and len({rows[i].lexical_family for i in heldout})==12,
                "syntax construction must be completely isolated")
        folds.append({
           "id":f"syntactic_construction_{c}_{name}",
           "kind":"TRAIN_INTERNAL_SYNTAX",
           "heldout_groups":[name],
           "train_rows":train,"holdout_rows":heldout
        })
    require(len(folds)==10
            and set(i for f in folds[:4] for i in f["holdout_rows"])==
                set(range(72))
            and set(i for f in folds[4:] for i in f["holdout_rows"])==
                set(range(72)),
            "cross-validation must cover every original TRAIN sentence once per axis")
    return folds


def select_pairs(x:DirectedPairTrain,source_rows)->DirectedPairTrain:
    """Use precommitted original sentence ID, NOT gold to choose split."""
    allowed=torch.zeros(72,dtype=torch.bool)
    for k in source_rows:
        require(type(k) is int and 0<=k<72,
                "never select unknown/duplicate old TRAIN split row")
        allowed[k]=True
    require(int(allowed.sum())==len(source_rows),"duplicate or unknown training row")
    mask=allowed[x.source_index]
    def sl(t):return t[mask].detach().clone().contiguous()
    y=sl(x.target)
    return DirectedPairTrain(
      sl(x.predicate),sl(x.argument),
      sl(x.predicate_positions),sl(x.argument_positions),
      sl(x.current_positions),y,sl(x.orientation),sl(x.source_index),
      len(source_rows),int((y!=0).sum())
    )


def score_rounded(model,x):
    physical=model.physical_bytes()
    reload=PaidDirectedValence.from_physical(physical).eval()
    require(len(physical)==PHYSICAL_BYTES and
            reload.physical_bytes()==physical,
            "all FP16 checkpoint sources must be physically reloaded")
    with torch.inference_mode():
        logits=frozen_paired_logits(reload,x)
        return metrics(logits,x),physical


def one_precommitted_fold(x:DirectedPairTrain,fold,out:Path)->dict:
    train=select_pairs(x,fold["train_rows"])
    held=select_pairs(x,fold["holdout_rows"])
    require(train.total_pairs+held.total_pairs==x.total_pairs
            and train.positive_links+held.positive_links==x.positive_links
            and set(train.source_index.tolist()).isdisjoint(
                set(held.source_index.tolist()))
            and train.rows+held.rows==72,
            "TRAIN/validation physical native wordpairs overlap")
    torch.manual_seed(318018)
    model=PaidDirectedValence().eval()
    require(sum(q.numel() for q in model.parameters())==PARAMETERS==2309,
            "only 2309 new physically paid FP16 source fields")
    initial_train,initial_blob=score_rounded(model,train)
    initial_held,_=score_rounded(model,held)
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE,
                                 weight_decay=.0002)
    for step in range(1,STEPS+1):
        optimizer.zero_grad(set_to_none=True)
        loss=score_weighted_train(model,train,WEIGHTS)
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "all rank8 physical params receive only TRAIN gradients")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    final_train,physical=score_rounded(model,train)
    final_held,again=score_rounded(model,held)
    require(physical==again and len(physical)==PHYSICAL_BYTES,
            "validation must NEVER refit model or change physical bytes")
    name=fold["id"]+".n3s16"
    file=out/name
    file.write_bytes(physical)
    require(file.read_bytes()==physical,
            "trained FP16 source physical ABI mismatch")
    def compact(report):
        return report["all_two_direction_wordpair_exact_role"]
    a=compact(initial_held);b=compact(final_held)
    return {
        "fold_id":fold["id"],"split_kind":fold["kind"],
        "heldout_original_S9_TRAIN_groups":fold["heldout_groups"],
        "train_s9_rows":train.rows,
        "internal_holdout_s9_rows":held.rows,
        "train_all_ordered_wordpairs":train.total_pairs,
        "internal_holdout_all_ordered_wordpairs":held.total_pairs,
        "train_gold_positive_links":train.positive_links,
        "internal_holdout_gold_positive_links":held.positive_links,
        "no_gold_POS_filtered_candidate_pairs":True,
        "fixed_optimizer_steps":STEPS,
        "selected_checkpoint_using_holdout":False,
        "physical_new_FP16_file":name,
        "physical_new_FP16_full_SHA256":hashlib.sha256(physical).hexdigest(),
        "physical_new_FP16_bytes":len(physical),
        "initial_untrained_train_only_f1":compact(initial_train)["f1"],
        "initial_untrained_heldout_TRAIN_f1":a["f1"],
        "final_training_f1":compact(final_train)["f1"],
        "final_internal_train_holdout_f1":b["f1"],
        "final_internal_train_holdout_precision":b["precision"],
        "final_internal_train_holdout_recall":b["recall"],
        "final_internal_train_holdout_TP":b["exact_directed_role_true_positive"],
        "final_internal_train_holdout_FP":b["false_positive"],
        "final_internal_train_holdout_FN":b["false_negative"],
        "final_heldout_TRAIN_by_role":final_held["per_role"],
        "final_heldout_TRAIN_by_orientation":final_held["by_surface_orientation"],
        "initial_model_not_prior_S16b_all72_learned_source":True,
        "S17c_consumed_EVAL24_not_loaded":True,
        "S15c_consumed_EVAL24_not_loaded":True
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("s10-hidden","s10-report","s8-cb59","s11-router","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    require(not a.out.exists(),
            "do not overwrite any old72 grouped CV or expose spent EVAL source")
    samples,s10=checked_real_train72(a.s10_hidden,a.s10_report)
    require(len(samples)==72
            and s10["all_616_native_CPU_generated_token_digests_unchanged"] is True
            and S9_TRAIN72_CANONICAL_SHA256==
                "b09d60a77a17d30c2dbf846e40ab97860def8635e2bf31a5b0ab8981832b744d"
            and S16_TRAIN_SOURCE_SHA==
                "9843c35ee9aa75b8f5c9c963e441589dd2f0292e41ec7933971d5aa3dcd365d6"
            and sealed_directed_train72()["S16_gold_directed_links"]==216,
            "original old72-only source J7 or signed gold SHA changed")
    original=a.s8_cb59.read_bytes()
    router=a.s11_router.read_bytes()
    require(len(original)==653798
            and hashlib.sha256(original).hexdigest()==S8B_43_CB59_SHA
            and len(CB59(original).handles)==43
            and CB59(original).meta["original_unique_parameter_count"]==6_106_154
            and len(router)==123
            and hashlib.sha256(router).hexdigest()==S11_39_FP16_SHA
            and PaidThreeWayRouter.from_physical(router).physical_bytes()==router,
            "original frozen native43 CB59 C or S11 39FP16 source integrity")
    a.out.mkdir(parents=True,exist_ok=False)
    so=a.out/"original_native43_5bit_CB59_crossfamily_C.so"
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
            and native.output_gate==0 and native.candidate_accepted is False,
            "original 43 native-C matrices and paid selector must be frozen")
    corpus=build_all_paired_native_train72(samples,native)
    require(corpus.rows==72 and corpus.positive_links==216
            and corpus.total_pairs==1920,
            "all 72 original Russian TRAIN, 216 direct gold, 1920 full ordered wordpairs")
    folds=controlled_group_folds()
    fold_reports=[]
    for k,fold in enumerate(folds,1):
        report=one_precommitted_fold(corpus,fold,a.out)
        fold_reports.append(report)
        print(json.dumps({
            "S18a_fold":k, "fold_id":report["fold_id"],
            "train_F1":report["final_training_f1"],
            "heldout_TRAIN_internal_F1":report["final_internal_train_holdout_f1"],
            "source_SHA":report["physical_new_FP16_full_SHA256"]
        },sort_keys=True,ensure_ascii=False),flush=True)
    stats={}
    for kind in ("TRAIN_INTERNAL_LEXICAL","TRAIN_INTERNAL_SYNTAX"):
        sub=[z for z in fold_reports if z["split_kind"]==kind]
        tp=sum(z["final_internal_train_holdout_TP"] for z in sub)
        fp=sum(z["final_internal_train_holdout_FP"] for z in sub)
        fn=sum(z["final_internal_train_holdout_FN"] for z in sub)
        n_gold=tp+fn
        p=tp/(tp+fp) if tp+fp else 0.
        rec=tp/(tp+fn) if tp+fn else 0.
        f1=2*p*rec/(p+rec) if p+rec else 0.
        require(n_gold==216 and all(z["fixed_optimizer_steps"]==150 for z in sub),
                "each axis must evaluate all 216 old TRAIN links once with blind fold training")
        stats[kind]={
            "folds":len(sub),"original_S9_TRAIN_gold_once_per_axis":216,
            "micro_tp":tp,"micro_fp":fp,"micro_fn":fn,
            "precision":p,"recall":rec,"f1":f1,
            "per_fold_F1":[z["final_internal_train_holdout_f1"] for z in sub]
        }
    summary={
        "stage":"NRA3_S18A_OLD72_ONLY_GENUINE43M_NATIVE_C_LEXEME_AND_CONSTRUCTION_GROUPED_CV",
        "S18a_original_S10_43M_GT_layer8_real72_FP16_SHA":S10_FEATURE_SHA,
        "S18a_original_S10_616_J7_readonly_provenance_run":S10_CAPTURE_RUN,
        "S18a_original_43_native_C_CB59_physical_SHA":S8B_43_CB59_SHA,
        "S18a_original_S11_39FP16_sixcycle_selector_SHA":S11_39_FP16_SHA,
        "S18a_original_S9_TRAIN72_complete_gold_SHA":S9_TRAIN72_CANONICAL_SHA256,
        "S18a_original_S16_DIRECTED_TRAIN216_SHA":S16_TRAIN_SOURCE_SHA,
        "S18a_exact_existing_S9_TRAIN_rows":72,
        "S18a_full_bidirectional_candidate_wordpairs":1920,
        "S18a_exact_directed_gold_positive_links":216,
        "S18a_four_family_group_folds_precommitted":True,
        "S18a_six_construction_group_folds_precommitted":True,
        "S18a_each_fold_from_scratch_only_150_train_optimizer_steps":True,
        "S18a_fixed_step_no_validation_checkpoint_selection":True,
        "S18a_test_group_gold_never_entered_optimizer":True,
        "S18a_no_oracle_POS_or_verb_filtered_candidates":True,
        "S18a_trained_only_2309_physically_paid_FP16_params_each_fold":True,
        "S18a_raw_FP16_bytes_per_fold":PHYSICAL_BYTES,
        "S18a_original43M_GT_CB59_S11_all_frozen":True,
        "S18a_grouped_cv_heldout_is_OLD_TRAIN_not_independent_NEW_EVAL":True,
        "S18a_no_OLD_EVAL12c_S15c_or_consumed_S17c_loaded":True,
        "S18a_frozen_S95_prompt_scope_uses_full_sentence_context":True,
        "S18a_end_to_end_online_word_causality_certified":False,
        "S18a_active_new_candidate_616_J7_or_eos_tested":False,
        "S18a_whole_Calibri43M_GT_NRA_speed_ge90pct_tested":False,
        "S18a_whole_inclusive_bpp_le07_certified":False,
        "S18a_no_LIVE_Jessica_chat_Render_Supabase_APK":True,
        "research_acceptance":False,"candidate_accepted":False,
        "output_gate":0,"activation_allowed":False,
        "S18a_grouped_CV_metrics":stats,
        "S18a_precommitted_folds_all_physical_S16_Rank8":[
            z for z in fold_reports
        ]
    }
    out=a.out/"NRA3_S18A_ORIGINAL43M_TRAIN72_GROUPED10_CV_F1_REPORT.json"
    out.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                   encoding="utf-8")
    print(json.dumps({k:v for k,v in summary.items()
        if k!="S18a_precommitted_folds_all_physical_S16_Rank8"},
        sort_keys=True,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
