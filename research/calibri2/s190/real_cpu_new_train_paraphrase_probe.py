"""S190: REAL frozen compressed 43M CPU J7 on 32 NEW TRAIN paraphrase probes.

Runs full original S187 456+128 first, with strict S187 no-forgetting and bpp
gate. Independently adds 24 source-grounded owner paraphrases + 8 numeric
rephrasings, all declared NEW TRAIN, no heldout QA/TEST, no neural training.
The selector is UNMODIFIED S187. No expected IDs, question SHA, or correctness
are consulted by inference. Freeze predictions BEFORE posthoc source scoring.
This is a diagnostic, NOT evidence of general language skill.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import torch
from calibri2.s160 import real_parallel_slots_j7 as s160
from calibri2.s187 import real_cpu_j7_history_s69_guard as s187
from calibri2.s182 import real_cpu_j7_shadow_guard as s182
from calibri2.s186.history_s69_prompt_only_shadow import source_candidate

BASE_SPEED=s160.paired_cpu_ratio
PROBE_RECORDS=None
SOURCE_FILES=(
    "research/calibri2/s188/paired_samecpu_s182_vs_s187.py",
    "research/calibri2/s188/test_paired_samecpu_s182_vs_s187.py",
    "research/calibri2/s189/source_metamorphic_train_only.py",
    "research/calibri2/s189/test_source_metamorphic_train_only.py",
    "research/calibri2/s190/real_cpu_new_train_paraphrase_probe.py",
    "research/calibri2/s190/test_real_cpu_new_train_paraphrase_probe.py",
)


def sha(q):
    return hashlib.sha256(q.encode("utf-8")).hexdigest()


def new_train_paraphrase_probes():
    """Predeclare 32 source-grounded NEW TRAIN items, never QA/TEST."""
    colors=("blue","green","red","yellow")
    nouns=("desk","lamp","vase","book","bowl")
    probes=[]
    for j in range(8):
        a=f"Owner{100+j}"
        b=f"Owner{200+j}"
        color=colors[j%4]
        left=f"{color} {nouns[j%5]}"
        right=f"{color} {nouns[(j+1)%5]}"
        cases=(
            (f"Parcel note: {a} ordered {left}; {b} ordered {right}. Which two words describe {a}'s item? Write just the item.",
             left, "owner_A_paraphrase"),
            (f"Inventory line: {a} has {left}. Separately, {b} has {right}. What item belongs to {b}? Two words only.",
             right, "owner_B_paraphrase"),
            (f"Dispatch list: item for {b} is {right}; item for {a} is {left}. State the color and noun of {a}'s item.",
             left, "reordered_owner_paraphrase"),
        )
        probes.extend(cases)
    for a,b in ((412,409),(300,301),(1001,999),(78,79),
                (101,100),(23,25),(999,2),(2,900)):
        q=f"Answer Yes or No: does {a} exceed {b}?"
        probes.append((q,"Yes" if a>b else "No","numeric_new_wording"))
    if len(probes)!=32 or len({sha(q) for q,_,_ in probes})!=32:
        raise AssertionError("S190 32 unique new TRAIN paraphrases must be frozen")
    for q,_,_ in probes:
        if source_candidate(q) is not None:
            raise AssertionError("S190 new TRAIN accidentally matches original memorized source templates")
    return probes


def probe_true_frozen_cpu(cpu,old,old_ids,routes,tok,bank,copy_net,
                          bridge,full,expansion,parent_old,head):
    """Runs after original protected S187 audit; no labels at routing."""
    global PROBE_RECORDS
    if PROBE_RECORDS is not None:
        raise AssertionError("S190 real CPU probe cannot run twice")
    from calibri2.s175 import negative_owner_guard_real_j7 as s175
    if s175.OLD_S172 is not s187.select_with_s186:
        raise AssertionError("S190 exact S187 selector must remain in place")
    model,sources,first,later,pointer,lm_head,*_=cpu
    if model.training:
        raise AssertionError("S190 CPU neural checkpoint must be frozen eval")
    probes=new_train_paraphrase_probes()
    s160.apply_profile(sources,cpu[6],"S93",bank,cpu[7],cpu[8])
    responses={}
    # Real neural CPU J7 and EOS on each NEW TRAIN question; freeze first.
    for q,_expected,_group in probes:
        feats=torch.as_tensor(s160.prompt_only_vector(tok,q),dtype=torch.float32)
        with torch.inference_mode(),expansion.prefill_only(model),s160.prefill_only_tunnel(model,full):
            with s160.frozen_cpu_phase_scope(model,bridge,"S93",feats,phase="prefill_only"):
                generated,_note=s160.cached_full_cpu_parallel(
                    model,sources,first,later,lm_head,tok,q,pointer,copy_net,head)
        responses[sha(q)]=list(generated)
    if len(responses)!=32:
        raise AssertionError("S190 new TRAIN real CPU J7 full outputs incomplete")
    # ONLY AFTER all CPU inference: derive expected from literal NEW TRAIN
    # construction (not stored original target labels nor SHA answer lookup).
    grouped={}
    summary={}
    for q,answer,group in probes:
        ids=tok.encode(answer,add_special_tokens=False)
        expected_len=1 if group=="numeric_new_wording" else 2
        if len(ids)!=expected_len or any(x==tok.eos_token_id for x in ids):
            raise AssertionError("S190 new TRAIN expected source vocabulary tokenization drift")
        produced=responses[sha(q)]
        expected=ids+[tok.eos_token_id]
        row={
            "prompt_sha256":sha(q),
            "group":group,
            "generated_tokens":len(produced),
            "eos_present":bool(produced and produced[-1]==tok.eos_token_id),
            "first_token_correct":bool(produced and produced[0]==ids[0]),
            "full_EOS_correct":produced==expected,
        }
        grouped.setdefault(group,[]).append(row)
    for group,rows in grouped.items():
        summary[group]={
            "count":len(rows),
            "first_correct":sum(x["first_token_correct"] for x in rows),
            "full_EOS_correct":sum(x["full_EOS_correct"] for x in rows),
            "eos_present":sum(x["eos_present"] for x in rows),
        }
    PROBE_RECORDS={
        "NEW_TRAIN_group_metrics":summary,
        "NEW_TRAIN_first_correct":sum(x["first_token_correct"] for r in grouped.values() for x in r),
        "NEW_TRAIN_full_EOS_correct":sum(x["full_EOS_correct"] for r in grouped.values() for x in r),
        "NEW_TRAIN_eos_present":sum(x["eos_present"] for r in grouped.values() for x in r),
        "hashed_per_prompt_audit":sorted(
            [x for r in grouped.values() for x in r],key=lambda x:x["prompt_sha256"]),
        "real_CPU_full_J7_new_TRAIN_all32":True,
        "all_new_TRAIN_unmatched_by_exact_old_source_grammars":True,
        "no_gold_or_SHA_at_inference_or_routing":True,
    }


def patched_speed(*args,**kwargs):
    base=BASE_SPEED(*args,**kwargs)
    probe_true_frozen_cpu(*args,**kwargs)
    return base


def main():
    global PROBE_RECORDS
    if len(sys.argv)<5:
        raise SystemExit("S190 uses complete S187 original 456+128 CLI frozen sources")
    out=Path(sys.argv[-1])
    if (out/"S190_REAL_CPU_32_NEW_TRAIN_PARAPHRASE_PROBES.json").exists():
        raise FileExistsError("S190 immutable new TRAIN inference record cannot overwrite")
    s160.paired_cpu_ratio=patched_speed
    try:
        s187.main() # FULL protected original 456+128 REAL CPU J7 and source bpp.
    finally:
        s160.paired_cpu_ratio=BASE_SPEED
    if PROBE_RECORDS is None:
        raise AssertionError("S190 actual CPU J7 new TRAIN probes did not execute")
    original=json.loads((out/"S187_REAL_CPU_S69_HISTORY_456_128_STRICT_J7.json").read_text())
    if (not original["research_candidate_pass"]
            or original["true_real_CPU_full_EOS_correct_original456"]!=456
            or original["true_real_CPU_new128_full_EOS"]!=128
            or original["lost_prior_full_EOS"]!=0
            or original["lost_prior_first_token"]!=0):
        raise AssertionError("S190 original 456+128 true J7 protected model drift")
    charged=sum(Path(p).stat().st_size for p in SOURCE_FILES)
    bpp=original["all_in_physical_bpp"]+8*charged/s182.DENOM
    if bpp>.7 or original["paired_CPU_baseline_vs_S101_ratio"]<.9:
        raise AssertionError("S190 physical all-in compressed bpp or frozen CPU baseline failed")
    if sum(x["count"] for x in PROBE_RECORDS["NEW_TRAIN_group_metrics"].values())!=32:
        raise AssertionError("S190 32 new TRAIN family probe records incomplete")
    report={
        "event":"S190_REAL_CPU_32_UNSEEN_NEW_TRAIN_PARAPHRASES_STRICT_SOURCE_ABSTAIN",
        "S187_pinned_real_CPU_run":43336858062,
        "S188_pinned_independent_same_CPU_run":43338233327,
        "original_real_CPU_FULL_EOS_456":456,
        "new128_real_CPU_FULL_EOS":128,
        "original_first_full_forgetting":0,
        "all_in_physical_bpp_charging_S188_S189_S190_source":bpp,
        "original_CPU_S101_ratio":original["paired_CPU_baseline_vs_S101_ratio"],
        "new_TRAIN_paraphrases_are_neither_QA_nor_TEST":True,
        "no_neural_weights_updated":True,
        "model_accepted":False,
        "Jessica_LIVE_Render_Supabase_APK_untouched":True,
        **PROBE_RECORDS,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"S190_REAL_CPU_32_NEW_TRAIN_PARAPHRASE_PROBES.json").write_text(
        json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="hashed_per_prompt_audit"},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    main()
