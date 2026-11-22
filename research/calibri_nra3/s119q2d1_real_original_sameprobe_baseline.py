"""S119q2d1: original frozen S142 43M+GT source-matched native baseline.

Exactly the SAME 30 previously exposed, authored SOURCE-only diagnostic
prefixes/targets measured by deeper S119q2d candidate and equal-param local.
One original 12layer physical forward for every prefix, NO optimizer,
new trainable adapter, teacher labels update, GOLD or independent quality.
This compares frozen original native full 49,152 head fairly with each
trained child for a specific next-token conditional metric only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    load_actual_original,sha,
)
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import (
    old_original_param_digest,
)
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    first_original_pass,
)
from research.calibri_nra3.s119q2b_real43m_onepass_native_train import (
    verify_q2a_artifact,
)
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    STAGE as Q2D_STAGE,REPORT as Q2D_REPORT,
    CONTEXT_CKPT,LOCAL_CKPT,source_inventory,pack_sources,
    VOCAB_SIZE,PROBE_COUNTS,
)

STAGE="CALIBRI_S119Q2D1_REAL_S142_43M_GT_ORIGINAL_SOURCE_PROBE_BASELINE"
REPORT="S119Q2D1_ORIGINAL_FROZEN_CORE43M_GT_30_NATIVE_SOURCE_PROBES.json"


def load_q2d_proof(report_path:Path,context_path:Path,local_path:Path,
                   s93:Path,s142:Path):
    check(report_path.name==Q2D_REPORT and
          context_path.name==CONTEXT_CKPT and local_path.name==LOCAL_CKPT,
          "fixed genuine q2d physical research child files and report only")
    result=json.loads(report_path.read_text(encoding="utf8"))
    check(result["stage"]==Q2D_STAGE and
          result["original_S93_SHA256"]==sha(s93) and
          result["original_S142_43M_GT_SHA256"]==sha(s142) and
          result["real_original_full_native_BPE_classes"]==VOCAB_SIZE and
          result["frozen_original_12layer_Core_GreenTunnel_unchanged"]
          and result["full_old_original_53751_NRA3_FP16_unchanged"]
          and result["calibration_18_and_source_probe_12_NEVER_used_for_gradient"]
          and result["output_gate"]==0 and
          result["candidate_accepted"] is False and
          result["source_probes_NOT_blind_or_independent"],
          "q2d physical TRAIN-only provenance and hard acceptance failclose")
    for path,key in ((context_path,"new_research_checkpoint_contextual"),
                     (local_path,"new_research_checkpoint_local")):
        metadata=result[key]
        check(sha(path)==metadata["sha256"] and
              path.stat().st_size==metadata["file_bytes"] and
              metadata["saved_total_child_FP16_params"]>150000,
              "require exact real persisted A/B trained tensors; not fabricated")
    return result


def summarize_original(rows):
    check(len(rows)==30 and
          Counter(x["source_split"] for x in rows)==PROBE_COUNTS,
          "only 18 CAL + 12 old SOURCE_PROBE matched original source prompts")
    counts=Counter(x["kind"] for x in rows)
    check(len(counts)==6 and all(v==5 for v in counts.values()),
          "six domains exactly five source-only old observations each")
    grouped=defaultdict(list)
    for row in rows:
        grouped[row["kind"]].append(row["next_native_CE"])
    return {
        "count_source_probes":len(rows),
        "source_only_greedy_correct":sum(
            x["correct_native_greedy_token"] for x in rows),
        "mean_source_only_native_CE":
            sum(x["next_native_CE"] for x in rows)/len(rows),
        "mean_source_only_by_domain":{
            k:sum(arr)/len(arr) for k,arr in sorted(grouped.items())},
        "source_details":rows,
        "not_blind_independent_or_full_answer_generation":True,
    }


def run(s93,side,s142,q2a_dir,q2d_dir,out):
    check(not out.exists(),"new immutable source-only diagnostic file only")
    torch.manual_seed(119134)
    torch.set_num_threads(2)
    q2d_report=load_q2d_proof(
        q2d_dir/Q2D_REPORT,
        q2d_dir/CONTEXT_CKPT,
        q2d_dir/LOCAL_CKPT,
        s93,s142)
    q2a_rows,q2a_manifest=verify_q2a_artifact(q2a_dir)
    check(q2a_manifest["synthetic_rows_SHA256"]==
          q2d_report["q2a_source_SHA256"],
          "same original q2a SOURCE SHA as actual q2d training")
    train_raw,probe_raw=source_inventory(q2a_rows)
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    check(tok.vocab_size==VOCAB_SIZE,
          "same immutable original native 49152 token classes")
    train_unused,probes=pack_sources(tok,train_raw,probe_raw)
    check(len(train_unused)==96 and len(probes)==30 and
          not {r["prompt_sha256"] for r in train_unused}&
          {r["prompt_sha256"] for r in probes},
          "one exact comparison set that was never optimized")
    model,sources,bridge,full,expansion=load_actual_original(
        s93,side,s142)
    frozen_digest=old_original_param_digest(model)
    check(all(not p.requires_grad for p in model.parameters()),
          "ALL original Core12+GT pretrained weights frozen")
    notes=[]
    for ex in probes:
        pos=ex["first_distinct_position"] if ex["source_family"]=="q2c1" else 0
        check(0<=pos<len(ex["completion_ids"]),
              "original model tested on SAME causal target native BPE index")
        prefix=ex["prefix"]+ex["completion_ids"][:pos]
        out_logits,unused_old_h12=first_original_pass(
            model,sources,bridge,full,expansion,prefix,
            ex["prompt_features"])
        check(out_logits.shape==(1,VOCAB_SIZE) and
              bool(torch.isfinite(out_logits).all()),
              "genuine original frozen Core12 GT full-native lm head")
        target=int(ex["completion_ids"][pos])
        ce=F.cross_entropy(
            out_logits.detach(),
            torch.tensor([target],dtype=torch.long))
        notes.append({
            "kind":ex["kind"],
            "source_split":ex["source_split"],
            "source_family":ex["source_family"],
            "target_position":pos,
            "next_native_CE":float(ce.item()),
            "correct_native_greedy_token":
                int(out_logits.detach().argmax(-1))==target,
            "original_native_full_vocab_49152":True,
            "prior_true_teacher_forced_answer_tokens_only":pos,
            "source_id":ex["id"],
        })
        del out_logits,unused_old_h12
    baseline=summarize_original(notes)
    check(old_original_param_digest(model)==frozen_digest and
          baseline["count_source_probes"]==30,
          "all 30 old actual source diagnostic with identical original weights")
    contextual=q2d_report["source_diagnostic_contextual"]
    local=q2d_report["source_diagnostic_local"]
    for prior in (contextual,local):
        check(prior["count_source_probes"]==30
              and len(prior["native_source_details"])==30
              and prior["not_blind_independent_or_natural_full_answer_generation"],
              "cannot substitute a different Q2D probe or skill benchmark")
        for old,new in zip(baseline["source_details"],
                           prior["native_source_details"]):
            check(old["source_id"]==new["source_id"]
                  and old["kind"]==new["kind"]
                  and old["target_position"]==new["target_position"],
                  "exact same 30 physical originals and candidate target positions")
    summary={
        "original_frozen_native_CE":baseline["mean_source_only_native_CE"],
        "trained_contextual_native_CE":contextual["mean_source_only_native_CE"],
        "trained_local_native_CE":local["mean_source_only_native_CE"],
        "contextual_MINUS_original_CE":
            contextual["mean_source_only_native_CE"]-
            baseline["mean_source_only_native_CE"],
        "local_MINUS_original_CE":
            local["mean_source_only_native_CE"]-
            baseline["mean_source_only_native_CE"],
        "contextual_MINUS_local_CE":
            contextual["mean_source_only_native_CE"]-
            local["mean_source_only_native_CE"],
        "original_native_exact_greedy_count":
            baseline["source_only_greedy_correct"],
        "contextual_native_exact_greedy_count":
            contextual["source_only_greedy_correct"],
        "local_native_exact_greedy_count":
            local["source_only_greedy_correct"],
        "all_results_only_previously_used_same_author_probes":True,
    }
    result={
        "stage":STAGE,
        "q2d_real_research_source_report_SHA256":sha(q2d_dir/Q2D_REPORT),
        "q2d_physical_original_training_96x4_reused_unmodified":True,
        "original_S93_sha256":sha(s93),
        "original_S142_43M_GT_sha256":sha(s142),
        "original_true_12layer_all_30_full_native_physical_CPU_forwards":True,
        "native_original_full_classes":VOCAB_SIZE,
        "old_S142_original_weights_unchanged":True,
        "no_optimization_or_retraining":True,
        "same_30_source_only_probes_already_previously_exposed":True,
        "q2a_calibration_AND_q2c1_source_probes_not_independent":True,
        "no_other_original_J7_golden_or_spent_EVAL128":True,
        "no_LIVE_Render_Supabase_Jessica_main_APK":True,
        "candidate_accepted":False,
        "output_gate":0,
        "summary":summary,
        "full_original_source_native_baseline":baseline,
    }
    out.mkdir(parents=True)
    (out/REPORT).write_text(json.dumps(
        result,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("S119Q2D1_REAL_ORIGINAL43M_NATIVE_SOURCE_BASELINE "+
          json.dumps(summary,sort_keys=True),flush=True)


def main():
    parser=argparse.ArgumentParser()
    for name in ("s93","side","s142","q2a_dir","q2d_dir","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args=parser.parse_args()
    run(args.s93,args.side,args.s142,args.q2a_dir,args.q2d_dir,args.out)


if __name__=="__main__":
    main()
