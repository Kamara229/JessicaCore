"""CALIBRI S46: forensic S45 paid-CB59 noun bottleneck (TRAIN outputs only).

Consume the real immutable S45 Actions JSON. Do NOT train, decode a new model,
inspect protected QA or ordinary Jessica chat, change LIVE, or claim quality
from teacher-forced noun ranks. Identify what stopped an actual full answer:
free greedy tokens, target noun ranks, and paid update guard vetoes.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
from pathlib import Path

def classify_trial(t):
    if not t["both_noun_sides_improved"]:
        return "noun_global_winner_margin_not_both_improved"
    if t["Yes_historical_margin"] is None:
        return "Yes_or_No_history_margin_not_safe"
    if t["lost_34_previously_correct_colors"] is None:
        return "Yes_or_No_history_margin_not_safe"
    if t["lost_34_previously_correct_colors"]:
        return "regressed_protected_first_color"
    if t["free_first_Yes_top1_count"]:
        return "reintroduced_free_first_Yes"
    if t["lost_7_historical_full_ids"] is None:
        return "historical_old7_not_checked"
    if t["lost_7_historical_full_ids"]:
        return "regressed_historical_full_answer"
    return "accepted"

def summarize_stage(stage):
    ans=stage["new_generated"]
    ranks=stage["new_noun_gold_ranks"]
    return {
        "new_free_full_correct":stage["new_64_full_correct"],
        "gold_noun_top1_teacher_forced":stage["new_gold_noun_top1"],
        "gold_noun_top10_teacher_forced":stage["new_gold_noun_top10"],
        "gold_noun_median_rank_teacher_forced":stage["new_gold_noun_median_rank"],
        "old_exact_full_correct":stage["old_8_full_correct"],
        "old_correct_ids":stage["old_correct_ids"],
        "free_text_top10":collections.Counter(x["text"] for x in ans.values()).most_common(10),
        "case_ids_rank_at_most_25":sorted(
            [{"id":k,"rank":n,"actual":ans[k]["text"]}
             for k,n in ranks.items() if n<=25],key=lambda x:x["rank"]),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("source",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    data=args.source.read_bytes()
    root=json.loads(data)
    if root["stage"]!="S45_REAL_PHYSICAL_CB59_GUARDED_NOUN_RECOVERY_TRAIN":
        raise AssertionError("wrong S45 source")
    if (root["physical_bytes"]!=3141513
        or not (0<root["complete_physical_bpp"]<0.7)
        or root["accepted_steps"]!=2
        or root["attempted_steps"]!=12):
        raise AssertionError("original S45 CI physical facts mismatch")
    if (root["first_color_correct_before"]!=34
        or root["first_color_correct_after"]!=34
        or root["first_Yes_top1_after"]!=0
        or not root["old_7_preserved"]):
        raise AssertionError("S45 safety/signature drift")
    if root["new_full_64_after"]!=0 or root["gold_noun_top1_after"]!=0:
        raise AssertionError("S45 full-answer bottleneck facts changed")
    if not all(root[k] for k in (
      "only_TRAIN_for_training_and_guarding",
      "paid_5bit_indices_fixed",
      "S29_QA_and_previously_inspected_S26_QA_not_used",
      "physical_cb59_saved")):
        raise AssertionError("guard or physical source contract not met")
    if not root["physical_reload"]["all_64_exact_full_and_noun_ranks_reloaded"]:
        raise AssertionError("missing real model physical reload")
    before,after=root["before"],root["after"]
    if len(before["new_generated"])!=64 or len(after["new_generated"])!=64:
        raise AssertionError("expected all 64 new TRAIN answers")
    if set(before["new_generated"])!=set(after["new_generated"]):
        raise AssertionError("case identities shifted")
    if before["old_correct_ids"]!=after["old_correct_ids"]:
        raise AssertionError("old full skill identities shifted")
    accepted=[]; rejected=[]
    trials=collections.Counter()
    for step in root["all_candidates"]:
        decisions=[]
        for t in step["attempts"]:
            reason=classify_trial(t)
            trials[reason]+=1
            decisions.append({"lr":t["learning_rate"],"reason":reason})
        event={"step":step["step"],"TRAIN_pair":step["TRAIN_pair"],
               "lr":step["selected_learning_rate"],"attempts":decisions}
        (accepted if step["selected_learning_rate"] is not None else rejected).append(event)
    if len(accepted)!=2 or len(rejected)!=10:
        raise AssertionError("S45 accepted/rejected step mismatch")
    per_case=[]
    for key in sorted(after["new_generated"]):
        b=before["new_generated"][key]; a=after["new_generated"][key]
        per_case.append({
          "TRAIN_id":key,
          "before_generated_text":b["text"],
          "after_generated_text":a["text"],
          "before_noun_rank_teacher_forced":before["new_noun_gold_ranks"][key],
          "after_noun_rank_teacher_forced":after["new_noun_gold_ranks"][key],
          "rank_improvement":before["new_noun_gold_ranks"][key]-after["new_noun_gold_ranks"][key],
          "real_free_tokens_changed":b["actual"]!=a["actual"],
          "full_answer_correct_after":a["full_correct"]
        })
    report={
      "stage":"S46_READONLY_REAL_S45_FREE_NOUN_BOTTLENECK",
      "source_S45_json_sha256":hashlib.sha256(data).hexdigest(),
      "source_real_GitHub_CI":42502640523,
      "source_physical_reload_sha256":root["physical_reload"]["sha256"],
      "physical_bpp":root["complete_physical_bpp"],
      "no_training_and_no_model_selection":True,
      "Jessica_chat_test_NOT_RUN":True,
      "S29_QA_and_S26_inspected_QA_not_used":True,
      "LIVE_not_modified":True,
      "before":summarize_stage(before),
      "after":summarize_stage(after),
      "top1_real_first_colors_preserved":34,
      "old7_full_answers_preserved":True,
      "free_first_yes_recurrence":0,
      "new_64_full_answers_correct_after":0,
      "changed_free_answer_count":sum(x["real_free_tokens_changed"] for x in per_case),
      "rank_better_count":sum(x["rank_improvement"]>0 for x in per_case),
      "rank_worse_count":sum(x["rank_improvement"]<0 for x in per_case),
      "rejection_attribution":dict(trials),
      "accepted_training_attempts":accepted,
      "rejected_training_attempts":rejected,
      "casewise_actual_greedy_vs_teacher_rank":per_case,
      "conclusion_limited_to_observed_TRAIN":
        "Noun rank gains alone have not produced full correct free answers; do not infer FP32 parity.",
      "next_safe_option":
        "Before changing physical weights again, test whether a general non-memorizing green-tunnel decoder can improve next-token noun selection without changing first-token color and old7 answers, with a bounded fixed paid size and separate heldout QA."
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({
      "event":report["stage"],
      "rejections":dict(trials),
      "rank_better":report["rank_better_count"],
      "rank_worse":report["rank_worse_count"],
      "real_full":report["new_64_full_answers_correct_after"],
      "changed_free":report["changed_free_answer_count"],
      "old7":True,
      "bpp":report["physical_bpp"],
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
