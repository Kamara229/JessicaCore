"""S97 TRAIN-only postmortem of S96's failed novel-prompt evaluation.

Uses immutable previously recorded per-ID TRAIN token outputs from S87,
S91, S93 and the already completed S96 aggregate result (no S96 prompts).
No second inspection of S96 examples, no models trained or decoded, no
QA/TEST read. This diagnoses *where* copying and EOS break, rather than
optimizing on a once-inspected novel evaluation.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

STAGE="S97_TRAIN_ONLY_TOKEN_POSITION_CAUSAL_DIAGNOSTIC"
PROFILES=("S82","S87","S91","S93")

def read_report(path):
    q=json.loads(Path(path).read_text())
    if (q.get("stage")!="S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT"
        or len(q.get("all_456_per_identity_results",[]))!=456):
        raise AssertionError("An immutable physical 456 CPU report is mandatory")
    rows=q["all_456_per_identity_results"]
    if len({x["id"] for x in rows})!=456:
        raise AssertionError("Duplicate frozen TRAIN identity")
    return {x["id"]:x for x in rows}

def classify(actual, expected):
    if len(expected)<2:
        raise AssertionError("A complete target needs at least one word and EOS")
    first=bool(actual) and actual[0]==expected[0]
    second=first and len(expected)>2 and len(actual)>1 and actual[1]==expected[1]
    eos_exact=second and len(actual)==len(expected) and actual==expected
    if actual==expected: label="full_and_EOS_correct"
    elif not first:label="first_token_wrong"
    elif len(expected)>2 and not second:
        label="early_EOS_instead_of_second" if len(actual)>1 and actual[1]==expected[-1] else "second_token_wrong"
    elif not eos_exact:
        label="correct_two_tokens_wrong_suffix_or_EOS"
    else:raise AssertionError("Unclassifiable response")
    return label,first,second,actual==expected

def analyze(refs):
    records={}
    for identity in sorted(refs["S87"]):
        root=refs["S87"][identity]
        for name in ("S91","S93"):
            item=refs[name][identity]
            if any(item[field]!=root[field] for field in ("id","question_sha256","target_ids","S82_ids")):
                raise AssertionError("Frozen TRAIN identity / S82 baseline drift")
        expected=root["target_ids"]
        if len(expected)<2 or expected[-1] in expected[:-1]:
            raise AssertionError("Expected full answer/EOS geometry differs")
        records[identity]={
            "family":identity.split("/")[0],"goal":expected,
            "actual":{
                "S82":root["S82_ids"],
                "S87":refs["S87"][identity]["S87_ids"],
                "S91":refs["S91"][identity]["S87_ids"],
                "S93":refs["S93"][identity]["S87_ids"],
            }}
    result={}
    for profile in PROFILES:
        by_family=defaultdict(Counter)
        total=Counter()
        failing_second=Counter()
        for identity,row in records.items():
            actual=row["actual"][profile]
            label,first,second,full=classify(actual,row["goal"])
            total[label]+=1
            total["total"]+=1
            total["first_correct"]+=int(first)
            total["first_and_second_correct"]+=int(second)
            total["first_correct_in_multiword_targets"]+=int(first and len(row["goal"])>2)
            total["one_word_targets"]+=int(len(row["goal"])==2)
            total["strict_full_with_EOS"]+=int(full)
            total["has_terminal_EOS"]+=int(bool(actual) and actual[-1]==row["goal"][-1])
            by_family[row["family"]][label]+=1
            by_family[row["family"]]["total"]+=1
            by_family[row["family"]]["strict_full_with_EOS"]+=int(full)
            if first and not second and len(actual)>1:
                # Numeric token IDs are diagnostics only; these are *not*
                # stored in the runtime classifier or shipped as answers.
                failing_second[int(actual[1])]+=1
        result[profile]={
            "token_failure_counters":dict(total),
            "per_TRAIN_family":{k:dict(v) for k,v in sorted(by_family.items())},
            "most_common_wrong_second_token_ids_when_first_correct":
                failing_second.most_common(8),
            "conditional_second_correct_given_first":
                total["first_and_second_correct"]/max(1,total["first_correct_in_multiword_targets"]),
            "conditional_full_EOS_given_first_second":
                total["strict_full_with_EOS"]/max(1,total["first_and_second_correct"]),
        }
    return result,records

def main():
    ap=argparse.ArgumentParser()
    for n in ("S87_full","S91_full","S93_full","S95_train_router_audit","S96_once_inspected_novel_aggregate"):
        ap.add_argument(n,type=Path)
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    refs={"S87":read_report(args.S87_full),
          "S91":read_report(args.S91_full),
          "S93":read_report(args.S93_full)}
    profiles,records=analyze(refs)
    if {k:profiles[k]["token_failure_counters"]["strict_full_with_EOS"] for k in ("S87","S91","S93")} != {"S87":196,"S91":257,"S93":268}:
        raise AssertionError("Frozen full+EOS published counts changed")
    old_union=set(k for k,r in records.items()
                  if r["actual"]["S87"]==r["goal"] or r["actual"]["S91"]==r["goal"])
    all_union=set(k for k,r in records.items()
                  if any(r["actual"][name]==r["goal"] for name in ("S87","S91","S93")))
    if len(old_union)!=282 or len(all_union)!=287:
        raise AssertionError("S97 union history contradicts prior actual source reports")
    s95=json.loads(args.S95_train_router_audit.read_text())
    if (s95.get("stage")!="S95_PROMPT_ONLY_NEURAL_THREE_PROFILE_ROUTER_TRAIN_RESEARCH"
        or s95.get("candidate_accepted") is not False):
        raise AssertionError("Frozen S95 TRAIN router verification unavailable")
    alltrain=s95["frozen_report_routed_TRAIN_prediction"]
    if (alltrain["correct_full_plus_EOS"]!=287 or
        alltrain["lost_previously_full_count"]!=0 or
        alltrain["lost_previously_first_count"]!=0):
        raise AssertionError("S95 actual TRAIN regression baseline changed")
    folds=s95["fourfold_HELD_OUT_TRAIN_not_QA"]
    if len(folds)!=4 or sum(x["heldout_TRAIN_rows"] for x in folds)!=456:
        raise AssertionError("S95 heldout-TRAIN folds do not cover 456")
    s96=json.loads(args.S96_once_inspected_novel_aggregate.read_text())
    if (s96.get("stage")!="S96_ONE_TIME_FROZEN_PREVIOUSLY_UNSEEN_PROMPTS_PHYSICAL_CPU_EOS"
        or s96.get("unseen_prompts_count")!=64 or
        not s96.get("S95_neural_router_and_all_profile_weights_frozen")):
        raise AssertionError("S96 immutable original 64 novel-prompt aggregate absent")
    routed=s96["unseen_routed_strict"]
    old=s96["unseen_S82_baseline_strict"]
    oracle=s96["unseen_three_profile_oracle_upper_bound_only"]
    if (routed["full_with_EOS"],old["full_with_EOS"],oracle)!=(2,0,2):
        raise AssertionError("S96 known independent negative result changed")
    report={
        "stage":STAGE,
        "TRAIN_only_source_456_tokens_per_identity":True,
        "prior_one_time_S96_aggregate_readonly":True,
        "NO_S96_question_or_gold_labels_loaded":True,
        "sealed_previously_inspected_QA_TEST_not_loaded":True,
        "no_training_no_new_CPU_generation":True,
        "LIVE_untouched":True,
        "candidate_accepted":False,
        "legacy_S87_or_S91_union":len(old_union),
        "theoretical_3profile_frozen_union":len(all_union),
        "frozen_TRAIN_profiles":profiles,
        "frozen_S95_fullTRAIN_neural_router":{
            "strict_full_plus_EOS":alltrain["correct_full_plus_EOS"],
            "lost_legacy_full":alltrain["lost_previously_full_count"],
            "lost_legacy_first":alltrain["lost_previously_first_count"],
            "physical_bpp_conservative":s95["physical_bpp_conservative"]},
        "frozen_S95_fourfold_TRAIN_heldout":{
            "full_answer_losses":sum(v["forgot_full"] for v in folds),
            "first_token_losses":sum(v["forgot_first"] for v in folds),
            "full_correct":sum(v["strict_full"] for v in folds),
            "rows":sum(v["heldout_TRAIN_rows"] for v in folds)},
        "frozen_S96_independent_novel_negative_result":{
            "strict_full_plus_EOS":routed["full_with_EOS"],
            "original_S82_strict_full_plus_EOS":old["full_with_EOS"],
            "theoretical_3profile_oracle":oracle,
            "routed_first_token":routed["first_token"],
            "routed_terminal_EOS":routed["ends_in_EOS"],
            "CPU_speed_ratio_vs_S82":s96["median_CPU_speed_ratio_S82_over_routed"],
            "passing_90pct_CPU_guard":s96["routed_CPU_at_least_90pct_S82"]},
        "next_experiment_design":"S98: repair general second-token prompt-copy on disjoint TRAIN only, zero-loss legacy TRAIN and newly preregistered unused nouns withheld; do NOT tune/retest S96.",
    }
    output=args.out/"S97_TRAIN_TOKEN_POSITION_AND_S96_NEGATIVE_EVIDENCE.json"
    output.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2))
    for k,v in profiles.items():
        t=v["token_failure_counters"]
        print(json.dumps({"event":"S97_IMMUTABLE_TRAIN_TOKEN_BREAKDOWN",
            "profile":k,"full_EOS":t["strict_full_with_EOS"],
            "first_ok":t["first_correct"],
            "second_given_first_ok":t["first_and_second_correct"],
            "early_EOS_instead_second":t.get("early_EOS_instead_of_second",0),
            "wrong_second":t.get("second_token_wrong",0),
            "correct_two_tokens_wrong_suffix_EOS":t.get("correct_two_tokens_wrong_suffix_or_EOS",0)}),flush=True)
    print(json.dumps({"event":"S97_FROZEN_NOVEL_GENERALIZATION_FAILURE",
        "S95_TRAIN_full":287,
        "S95_heldout_TRAIN_full":report["frozen_S95_fourfold_TRAIN_heldout"]["full_correct"],
        "S96_previously_unseen_full":routed["full_with_EOS"],
        "S96_unseen_oracle":oracle,
        "S96_CPU_speed_ratio":s96["median_CPU_speed_ratio_S82_over_routed"],
        "candidate_accepted":False}),flush=True)

if __name__=="__main__":
    main()
