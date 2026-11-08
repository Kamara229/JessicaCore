"""S42 strictly read-only diagnosis of the 64 paired TRAIN observations in S41.

Read ONLY the published S41 TRAIN JSON. No model loading, gradient, test,
validation, held-out S29 QA, S26 inspected QA or Jessica ordinary-chat test.
No LIVE, Render, Supabase, or APK changes. Creates a reproducible JSON report.
"""
from __future__ import annotations
import argparse
import collections
import json
import statistics
from pathlib import Path

def analyze(data):
    if data.get("stage")!="S41_REAL_CB59_PAID_CENTROID_GLOBAL_NOUN_WINNER_RECOVERY":
        raise AssertionError("Not the sealed S41 physical CB59 report")
    if data.get("source_S39_SHA256")!="92d87d6b2fce6871c98a99257ae752e4a777604a10e1be60bfca058cd61e5bba":
        raise AssertionError("Wrong S39 physical source for S41")
    if abs(data["physical_bpp"]-0.6908128754760795)>1e-12:
        raise AssertionError("S41 complete physical bitrate changed")
    assert data["S29_immutable_QA_and_S26_previously_used_QA_UNTOUCHED"] is True
    assert data["only_TRAIN_for_all_learning_selection_and_reporting"] is True
    assert data["Jessica_ordinary_chat_TEST_NOT_RUN"] is True
    assert data["LIVE_UNMODIFIED"] is True
    assert data["accepted_steps"]==18
    b=data["before"]["all_64_new_TRAIN"]
    a=data["after"]["all_64_new_TRAIN"]
    assert len(a)==len(b)==64
    assert [r["id"] for r in b]==[r["id"] for r in a]
    assert len(set(r["id"] for r in a))==64
    assert all(r["gold"]==s["gold"] for r,s in zip(a,b))
    def stage(rows):
        rank=[r["noun_gold_rank"] for r in rows]
        margin=[r["noun_gold_minus_global_winner_logit"] for r in rows]
        return {
           "count":len(rows),
           "predicted_winner_token_counts":dict(collections.Counter(r["predicted_winner_token"] for r in rows).most_common()),
           "free_generated_text_counts":dict(collections.Counter(r["generated_text"] for r in rows).most_common()),
           "greedy_starts_yes_count":sum(r["generated_text"].strip()=="Yes" for r in rows),
           "gold_noun_rank_min":min(rank),
           "gold_noun_rank_median":statistics.median(rank),
           "gold_noun_rank_max":max(rank),
           "gold_noun_rank_top1":sum(x==1 for x in rank),
           "gold_noun_rank_top10":sum(x<=10 for x in rank),
           "gold_minus_global_wrong_winner_logit_mean":statistics.mean(margin),
           "full_answers_with_EOS_correct":sum(bool(r["full_correct_with_eos"]) for r in rows),
        }
    before=stage(b)
    after=stage(a)
    if before["gold_noun_rank_top1"]!=0 or after["gold_noun_rank_top1"]!=0:
        raise AssertionError("S41 top-1 baseline differs")
    if before["full_answers_with_EOS_correct"]!=0 or after["full_answers_with_EOS_correct"]!=0:
        raise AssertionError("S41 zero full-answer baseline differs")
    if data["before"]["historical_8_TRAIN_full_answers"]!=7 or data["after"]["historical_8_TRAIN_full_answers"]!=7:
        raise AssertionError("Historical seven TRAIN skill guard changed")
    changed=[{"id":x["id"],"gold":x["gold"],
              "old_winner":y["predicted_winner_token"],
              "new_winner":x["predicted_winner_token"],
              "rank_before":y["noun_gold_rank"],"rank_after":x["noun_gold_rank"],
              "free_before":y["generated_text"],"free_after":x["generated_text"],
              "free_after_is_yes":x["generated_text"].strip()=="Yes"}
             for x,y in zip(a,b)]
    result={
      "stage":"S42_S41_64_TRAIN_NOUN_WINNER_AND_FREE_GENERATION_COLLAPSE",
      "input_source":"S41 physical TRAIN-only CI JSON, same 64 rows before and after",
      "S41_physical_model_sha256":data["physical_reload"]["sha256"],
      "physical_bpp":data["physical_bpp"],
      "historical_old_correct_full_answers_before_after":[7,7],
      "S41_accepted_paid_centroid_steps":18,
      "before":before,"after":after,
      "per_row_transitions":changed,
      "diagnostic_hypothesis_not_proven":"Teacher-forced gold noun ranking improved, but unconstrained greedy generation often emits Yes as a complete response; training objective neglected free-decoding first-token distribution and sequence-level correctness.",
      "next_research_step":"Measure paired actual greedy first-token winner, gold color rank, noun rank after generated first token and EOS for all 64 TRAIN prompts; protect all seven old answers and Yes/No; only then design global-distribution-aware full-answer paid-CB59 S43. Do not tune on held-out QA.",
      "no_training_or_model_selection":True,
      "no_S29_or_previously_inspected_S26_QA":True,
      "no_Jessica_chat_test":True,
      "no_LIVE_changes":True,
    }
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    report=analyze(json.loads(args.source.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({"event":"S42_TRAIN_COLLAPSE_FACTS",
      "before_winner":list(report["before"]["predicted_winner_token_counts"].items())[:3],
      "after_winner":list(report["after"]["predicted_winner_token_counts"].items())[:8],
      "free_yes_count_after":report["after"]["greedy_starts_yes_count"],
      "noun_top1_after":report["after"]["gold_noun_rank_top1"],
      "full_new_after":report["after"]["full_answers_with_EOS_correct"],
      "old_protected":7,"physical_bpp":report["physical_bpp"]},ensure_ascii=False))

if __name__=="__main__":
    main()
