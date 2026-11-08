"""S23: explain S20 learned multiword syntax but failed unseen color+noun binding.

Purely read-only examination of the 144+16 S20 independent full
answer artifacts and 32 TRAIN conditional-next-token S22 artifacts.
No models, weight updates, QA-led checkpoint selection or user chat
tests. Reports full literal answers and distinct diagnostic metrics:
first color word, second noun, both fields, full EOS, output length,
TRAIN/QA gap, per-withheld-combination generalization, memorized noun
bias, and whether correct color is often paired with a wrong object.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

EXPECTED_SUITE="ac992f3670c50e53bfe1711159ad330e67037291191b85ca108895d26d6fda97"
EXPECTED_SHA={
    "S7":"70512bc374c1846631f635998f9e035e2e25f147da185c2b7b0dd663799d9f8a",
    "S20":"4b62bcec5e40c2b7b73d3be38e4c365d1cf9a62614420497490795534477e93d",
}
COLORS=("red","blue","green","yellow")
NOUNS=("cup","book","bag","lamp","chair","hat")
FULL=re.compile(r"^\s*(red|blue|green|yellow)\s+(cup|book|bag|lamp|chair|hat)\s*[.!]?\s*$",re.I)


def read_qa(path,label):
    data=json.loads(path.read_text())
    if data["precommitted_suite_sha256"]!=EXPECTED_SUITE:
        raise AssertionError("S23 suite drift")
    if data["model_sha256"]!=EXPECTED_SHA[label]:
        raise AssertionError("S23 actual physical model SHA mismatch")
    if data["complete_bytes"]!=3141513 or data["unique_parameters"]!=36380480:
        raise AssertionError("S23 physical byte count mismatch")
    if not data["physical_bpp"]<.7:
        raise AssertionError("S23 physical 0.7 bits/param violation")
    rows=data["actual_144_complete_greedy_answers"]
    opened=data["actual_16_open_answers"]
    if len(rows)!=144 or len(opened)!=16:
        raise AssertionError("S23 missing actual independent answers")
    seen={r["id"] for r in rows}
    if len(seen)!=144 or sum(r["kind"]=="unseen_color_object_binding" for r in rows)!=96:
        raise AssertionError("S23 missing independent or duplicated QA cases")
    return data,{x["id"]:x for x in rows}


def stats(rows):
    buckets=defaultdict(lambda:Counter())
    labels=Counter()
    predictions=Counter()
    examples=[]
    for case in rows.values():
        gold=case["gold"].split()
        text=case["actual_text"].strip()
        domain=case["kind"]
        v=buckets[domain]
        v["cases"]+=1
        v["strict_full_EOS"]+=int(case["full_plus_EOS"])
        v["generated_EOS"]+=int(case["EOS_emitted"])
        v["stopped_too_early"]+=int(case["stopped_too_early"])
        if domain!="unseen_color_object_binding":
            continue
        m=FULL.fullmatch(text)
        prefix=re.match(r"^\s*(red|blue|green|yellow)(?:\s|$)",text,re.I)
        noun=re.search(r"\b(cup|book|bag|lamp|chair|hat)\b",text,re.I)
        first=(prefix.group(1).lower() if prefix else None)
        second=(m.group(2).lower() if m else (noun.group(1).lower() if noun else None))
        color_ok=first==gold[0]
        noun_ok=second==gold[1]
        v["valid_2word_format"]+=int(m is not None)
        v["color_word_correct"]+=int(color_ok)
        v["noun_word_correct"]+=int(noun_ok)
        v["both_words_semantically_correct"]+=int(m is not None and color_ok and noun_ok)
        v["color_correct_noun_wrong"]+=int(color_ok and not noun_ok)
        v["noun_correct_color_wrong"]+=int(noun_ok and not color_ok)
        v["neither_word_correct"]+=int(not noun_ok and not color_ok)
        v["valid_2word_no_EOS"]+=int(m is not None and not case["EOS_emitted"])
        labels[case["gold"]]+=1
        predictions[text]+=1
        combo=case["gold"]
        pairlabel=labels[combo]
        examples.append({
            "id":case["id"],"gold":combo,
            "real_answer":text,
            "two_fields_parsed":bool(m),
            "color_correct":color_ok,"noun_correct":noun_ok,
            "both_semantic":bool(m and color_ok and noun_ok),
            "strict_EOS":bool(case["full_plus_EOS"]),
            "EOS_emitted":bool(case["EOS_emitted"]),
            "ground_truth_heldout_combo_index":case["combo_index"],
            "answer_token_ids":case["actual_token_ids"],
        })
    if len(examples)!=96 or len(labels)!=8:
        raise AssertionError("Unexpected withheld QA composition")
    return dict(buckets),predictions.most_common(25),examples,labels


def read_s22(path):
    j=json.loads(path.read_text())
    if j["SOURCE_S7_SHA"]!=EXPECTED_SHA["S7"] or j["SOURCE_S20_SHA"]!=EXPECTED_SHA["S20"]:
        raise AssertionError("S22 diagnostic output from untrusted checkpoints")
    if j["total_paired_samples"]!=32:
        raise AssertionError("Missing S22 TRAIN conditional records")
    a=j["S7_before"];b=j["S20_after"]
    if len(a["records"])!=32 or len(b["records"])!=32:
        raise AssertionError("Not 32 TRAIN controls")
    return {
        "S7":{key:value for key,value in a.items() if key not in ("records","model_label","model_sha")},
        "S20":{key:value for key,value in b.items() if key not in ("records","model_label","model_sha")},
        "actual_TRAIN_examples":[{
            "gold":x["gold_answer"],
            "S7_actual":x["free_greedy_answer"],
            "S20_actual":y["free_greedy_answer"],
            "S7_gold_second_rank":x["forced_second_token"]["target_rank"],
            "S20_gold_second_rank":y["forced_second_token"]["target_rank"],
            "S7_EOS_was_top1_after_gold_first":x["forced_second_token"]["eos_top1"],
            "S20_EOS_was_top1_after_gold_first":y["forced_second_token"]["eos_top1"],
        } for x,y in list(zip(a["records"],b["records"]))[:12]]
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("s7_qa",type=Path)
    p.add_argument("s20_qa",type=Path)
    p.add_argument("s22_train",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    before,b=read_qa(args.s7_qa,"S7")
    after,a=read_qa(args.s20_qa,"S20")
    if b.keys()!=a.keys():
        raise AssertionError("QA case identity changed between S7 and S20")
    for key in b:
        if b[key]["question"]!=a[key]["question"] or b[key]["gold"]!=a[key]["gold"]:
            raise AssertionError("Paired QA question or gold mutated")
    bstats,bpopular,brows,bgold=stats(b)
    astats,apopular,arows,agold=stats(a)
    if bgold!=agold:
        raise AssertionError("QA composition distribution changed")
    comparative=[]
    for pre,post in zip(brows,arows):
        if pre["id"]!=post["id"] or pre["gold"]!=post["gold"]:
            raise AssertionError("Alignment failed")
        comparative.append({
            "id":pre["id"],
            "correct_gold":pre["gold"],
            "S7_real":pre["real_answer"],
            "S20_real":post["real_answer"],
            "S7_color":pre["color_correct"],
            "S20_color":post["color_correct"],
            "S7_noun":pre["noun_correct"],
            "S20_noun":post["noun_correct"],
            "S20_full_with_EOS":post["strict_EOS"]
        })
    diagnostic=read_s22(args.s22_train)
    report={
        "stage":"S23_S20_SECOND_WORD_BINDING_AND_EOS_FAILURE_ANALYSIS_ONLY",
        "precommitted_new_QA_sha":EXPECTED_SUITE,
        "before_SHA":EXPECTED_SHA["S7"],"after_SHA":EXPECTED_SHA["S20"],
        "complete_physical_bytes":3141513,
        "complete_bpp":before["physical_bpp"],
        "new_withheld_2word_combinations":dict(agold),
        "before_per_domain":bstats,"after_per_domain":astats,
        "before_top_answers":bpopular,
        "after_top_answers":apopular,
        "all_96_before_after_ACTUAL_QA_answers":comparative,
        "S22_TRAIN_only_forced_second_token_diagnostics":diagnostic,
        "all_16_open_before_after":[{
            "question":bq["question"],
            "S7_answer":bq["actual_text"],
            "S20_answer":aq["actual_text"],
            "S7_EOS":bq["EOS"],
            "S20_EOS":aq["EOS"],
        } for bq,aq in zip(before["actual_16_open_answers"],after["actual_16_open_answers"])],
        "not_a_checkpoint_selection_training_signal":True,
        "QA_never_used_to_choose_or_modify_model_weights":True,
        "no_FP32_reference_or_135m_teacher":True,
        "no_LIVE_changes":True,
    }
    out=args.out/"ACTUAL_CALIBRI2_S23_S20_ERROR_CAUSE_96_16_32.json"
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "event":"CALIBRI2_S23_REAL_UNSEEN_2WORD_ERROR_AUDIT_DONE",
        "baseline":bstats,"S20":astats,
        "top_before":bpopular[:8],"top_after":apopular[:12],
        "TRAIN_conditional":{name:diagnostic[name] for name in ("S7","S20")},
        "real_8_new_examples":comparative[:8],
        "real_3_open_examples":report["all_16_open_before_after"][:3],
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
