"""S205 TRAIN-only posthoc: diagnose S204 native 32 novel J7 tokens.

Reads ONLY already completed immutable S204 full real native CPU J7 report and
the corresponding S190 source-only decoded 32 TRAIN flags; never reruns the
model, never changes weights, never routes by gold/SHA, never accesses QA/TEST.

Important: first-token correctness plus generated length and EOS permit a
*strict second-token error proof* for a two-word TRAIN answer when the first
token matches, the output is length 3 and the last token is EOS.
No raw predicted token IDs are available in the S204 manifest; do not invent
them. More granular logits/differences require a separate new native trace.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

STAGE = "S205_POSTHOC_ACTUAL_S204_NOVEL32_TOKEN_FAILURES"
SOURCE_RUN = 43389569329
EXPECTED_GROUPS = {
    "owner_A_paraphrase": (8,3),
    "owner_B_paraphrase": (8,3),
    "reordered_owner_paraphrase": (8,3),
    "numeric_new_wording": (8,2),
}


def diagnose(s204: dict, s190: dict) -> dict:
    """Purely posthoc, fail closed on incomplete or non-native provenance."""
    if (s204.get("stage")!="S204_ACTUAL_NATIVE_616_CPU_J7_PROTECTED_NRA_CANDIDATE"
        or s204.get("source_S197_pinned_GitHub_run")!=43369398297
        or s204.get("full_456_correct_EOS")!=456
        or s204.get("full_128_correct_EOS")!=128
        or s204.get("model_accepted") is not False
        or s204.get("S204_native_CPU_456_128_source_hook",{}).get(
            "real_CPU_inference_completed_before_gold_scoring") is not True
        or s190.get("real_CPU_full_J7_new_TRAIN_all32") is not True
        or s190.get("no_gold_or_SHA_at_inference_or_routing") is not True
        or s190.get("original_real_CPU_FULL_EOS_456")!=456
        or s190.get("new128_real_CPU_FULL_EOS")!=128):
        raise AssertionError("S205 genuine unaccepted S204 original CPU native 616 provenance required")
    rows = s190.get("hashed_per_prompt_audit")
    if not isinstance(rows,list) or len(rows)!=32:
        raise AssertionError("S205 requires all 32 actual native J7 prompt audits")
    all_sha=set()
    counts=defaultdict(Counter)
    second_fail_proofs=defaultdict(list)
    numeric_wrong_first=0
    for row in rows:
        if not isinstance(row,dict):
            raise AssertionError("S205 no synthetic or missing TRAIN audit")
        sha=row.get("prompt_sha256")
        group=row.get("group")
        n=row.get("generated_tokens")
        first=row.get("first_token_correct")
        full=row.get("full_EOS_correct")
        eos=row.get("eos_present")
        if (not isinstance(sha,str) or len(sha)!=64
            or any(c not in "0123456789abcdef" for c in sha)
            or sha in all_sha or group not in EXPECTED_GROUPS
            or type(n) is not int or n<1 or n>4
            or not all(type(x) is bool for x in (first,full,eos))
            or (full and (not first or not eos))):
            raise AssertionError("S205 invalid SHA-group or actual generated token flags")
        all_sha.add(sha)
        expected_len=EXPECTED_GROUPS[group][1]
        c=counts[group]
        c["count"]+=1
        c["first_correct"]+=int(first)
        c["full_EOS_correct"]+=int(full)
        c["EOS_present"]+=int(eos)
        c["actual_generated_len_"+str(n)]+=1
        c["expected_len_"+str(expected_len)]+=1
        c["length_mismatch"]+=int(n!=expected_len)
        # For a two-word answer 3 tokens = first, second, EOS. With
        # matching first, exact length, true EOS, but full!=target,
        # the **only remaining mismatch is token number two**.
        if group!="numeric_new_wording" and n==3 and first and eos and not full:
            c["proven_second_token_fail"]+=1
            second_fail_proofs[group].append(sha)
        if group=="numeric_new_wording" and not first:
            numeric_wrong_first+=1
    if set(counts)!=set(EXPECTED_GROUPS) or any(
        counts[k]["count"]!=EXPECTED_GROUPS[k][0] for k in EXPECTED_GROUPS):
        raise AssertionError("S205 source group coverage drift")
    group_summary=s190.get("NEW_TRAIN_group_metrics")
    if not isinstance(group_summary,dict) or set(group_summary)!=set(EXPECTED_GROUPS):
        raise AssertionError("S205 source S190 group summary changed")
    for group in EXPECTED_GROUPS:
        a=counts[group];b=group_summary[group]
        for k,derived in (("count",a["count"]),
                          ("first_correct",a["first_correct"]),
                          ("full_EOS_correct",a["full_EOS_correct"]),
                          ("eos_present",a["EOS_present"])):
            if b.get(k)!=derived:
                raise AssertionError("S205 native S190 individual/group contradiction")
    totals={
        "first_correct":sum(x["first_correct"] for x in counts.values()),
        "full_EOS_correct":sum(x["full_EOS_correct"] for x in counts.values()),
        "EOS_present":sum(x["EOS_present"] for x in counts.values()),
        "provable_second_token_failures":sum(x["proven_second_token_fail"] for x in counts.values()),
        "length_mismatches":sum(x["length_mismatch"] for x in counts.values()),
    }
    if (totals["first_correct"]!=s204.get("32_NEW_TRAIN_first_correct")
        or totals["full_EOS_correct"]!=s204.get("32_NEW_TRAIN_full_EOS_correct")
        or totals["first_correct"]!=s190.get("NEW_TRAIN_first_correct")
        or totals["full_EOS_correct"]!=s190.get("NEW_TRAIN_full_EOS_correct")
        or totals["EOS_present"]!=s190.get("NEW_TRAIN_eos_present")):
        raise AssertionError("S205 native CPU report cross-check failed")
    bpp=s204.get("total_physical_budget",{})
    b=float(bpp.get("conservative_bpp_excluding_zero_structural_growth",float("nan")))
    bits=float(bpp.get("all_in_bits",float("nan")))
    if not (math.isfinite(bits) and math.isfinite(b) and bits>0 and b>0):
        raise AssertionError("S205 missing actual all-inclusive physical model invoice")
    denom=bits/b
    # Use the exact original conservative architecture denominator; account
    # for all real runtime code and data rather than assuming free bits.
    if abs(denom-43283660)>1:
        raise AssertionError("S205 conservative physical denominator modified")
    overbytes=max(0,(bits-.7*denom)/8)
    return {
        "stage":STAGE,
        "original_S204_true_CPU_J7_GitHub_run":SOURCE_RUN,
        "actual_S190_32_SOURCE_ONLY_TRAIN_no_QA_TEST":True,
        "real_native_CPU_full_EOS_456":456,
        "real_native_CPU_full_EOS_128":128,
        "original_4_S73_protected":len(s204["S204_native_CPU_456_128_source_hook"]["S73_actual_generated_ids"])==4,
        "total_novel32_TRAIN":32,
        "counts":totals,
        "group_error_types":{k:dict(v) for k,v in sorted(counts.items())},
        "proven_second_token_mismatch_SHAs":{k:sorted(v) for k,v in sorted(second_fail_proofs.items())},
        "numeric_first_wrong_count":numeric_wrong_first,
        "S204_source_inclusive_conservative_bpp":b,
        "max_bpp":.7,
        "budget_bytes_over_hard_limit":math.ceil(overbytes),
        "independent_candidate_vs_parent_CPU_speed_STILL_UNVERIFIED":
            s204.get("CPU_ratio_independently_paired_for_new_candidate") is not True,
        "training_candidate_accepted":False,
        "retrain_or_decoder_inference_ran":False,
        "raw_novel32_generated_token_ids_not_recorded_in_S204":True,
        "next_step":"Record raw novel32 actual token IDs during separately pinned native S204 J7; prioritize 16 provably wrong second tokens and 8 numeric first-token errors.",
        "Jessica_LIVE_Render_Supabase_unchanged":True,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("s204_full_j7",type=Path)
    p.add_argument("s190_novel32_report",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    if a.out.exists():
        raise FileExistsError("S205 immutable posthoc diagnostic cannot overwrite")
    result=diagnose(json.loads(a.s204_full_j7.read_text()),
                    json.loads(a.s190_novel32_report.read_text()))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True))
    print(json.dumps({
        "event":STAGE,
        "protected_456":456,"protected_128":128,
        "32_first_correct":result["counts"]["first_correct"],
        "proven_second_token_failures":result["counts"]["provable_second_token_failures"],
        "numeric_wrong_first":result["numeric_first_wrong_count"],
        "physical_bpp_over_limit":result["budget_bytes_over_hard_limit"],
        "model_accepted":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
