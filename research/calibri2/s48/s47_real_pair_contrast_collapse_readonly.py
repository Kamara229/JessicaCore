"""CALIBRI S48: audit S47 full TRAIN answers for opposing-noun collapse.

Read ONLY immutable S47 actual GitHub report produced by real S47 CI.
Do NOT train, rank checkpoints, use independently sealed QA or inspect
Jessica ordinary chat. Claiming 17/64 exact matches is misleading if both
opposed sides of each controlled TRAIN pair receive one identical answer.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

SOURCE_STAGE="S47_REAL_ACTIVE_TINY_LOW_RANK_HIDDEN_GREEN_TUNNEL"
EXPECTED_S47_CI=42505575247

def audit(data):
    if data.get("stage")!=SOURCE_STAGE or not data["accepted_actual_new_full_improvement"]:
        raise AssertionError("expected approved S47 training-only actual model report")
    if data["baseline"]!={"new_full":0,"new_correct_first_color":34,"old_full":7}:
        raise AssertionError("S47 original physical TRAIN baseline drift")
    if abs(float(data["all_inclusive_bpp_including_tunnel"])-0.695841726112)>1e-11:
        raise AssertionError("S47 all-inclusive physical bpp drift")
    after=data["after"]
    assert after["new_full"]==17 and after["old_full"]==7
    assert after["new_correct_first_color"]==34
    assert data["physical_reload"]["all_64_new_full_sequences_identical"]
    assert data["physical_reload"]["all_8_old_full_sequences_identical"]
    rows=after["all_64_new"]
    before=data["all_64_before"]
    if len(rows)!=64 or set(rows)!=set(before):
        raise AssertionError("64 original TRAIN example keys differ")
    ids=sorted({k.rsplit("_",1)[0] for k in rows})
    if len(ids)!=32:raise AssertionError("expected exactly 32 TRAIN contrasting noun pairs")
    result=[]
    for id in ids:
        l=rows[id+"_left"];r=rows[id+"_right"]
        lb=before[id+"_left"];rb=before[id+"_right"]
        if l["expected"]==r["expected"]:
            raise AssertionError("nominally contrasted pair has identical gold response")
        result.append({
          "TRAIN_pair_id":int(id),
          "both_full_correct":bool(l["full_correct"] and r["full_correct"]),
          "exactly_one_correct":bool(l["full_correct"])!=bool(r["full_correct"]),
          "identical_free_text":l["text"]==r["text"],
          "identical_free_token_sequence":l["actual"]==r["actual"],
          "before_free_text_identical":lb["text"]==rb["text"],
          "left_free":l["text"],"right_free":r["text"],
          "left_full_correct":bool(l["full_correct"]),
          "right_full_correct":bool(r["full_correct"]),
          "left_expected_token_ids":l["expected"],
          "right_expected_token_ids":r["expected"]
        })
    predictions=Counter(row["text"] for row in rows.values())
    counts={
       "new_train_total":64,
       "pairs_total":len(result),
       "full_correct_new_before":sum(x["full_correct"] for x in before.values()),
       "full_correct_new_after":sum(x["full_correct"] for x in rows.values()),
       "both_contrast_sides_full_correct":sum(x["both_full_correct"] for x in result),
       "exactly_one_side_full_correct":sum(x["exactly_one_correct"] for x in result),
       "same_text_on_two_opposing_gold_noun_sides":sum(x["identical_free_text"] for x in result),
       "same_actual_token_seq_on_opposing_gold_noun_sides":sum(x["identical_free_token_sequence"] for x in result),
       "distinct_free_texts":len(predictions),
       "max_one_text_share":max(predictions.values())/64,
       "old_historical_full_correct_retained":after["old_full"],
       "original_first_gold_colors_correct_retained":after["new_correct_first_color"],
       "all_inclusive_physical_bpp":data["all_inclusive_bpp_including_tunnel"]
    }
    assert counts["full_correct_new_before"]==0
    assert counts["full_correct_new_after"]==17
    if counts["both_contrast_sides_full_correct"]==0 and counts["same_text_on_two_opposing_gold_noun_sides"]==32:
        interpretation="TRAIN prefix answer collapse: 17/64 matched by luck of one identical output per contrast; model does not discriminate paired nouns. No quality parity claim."
    else:
        interpretation="Measure actual paired noun discrimination; never equate aggregate correct count to learned task generalization."
    return {
       "stage":"S48_READONLY_REAL_S47_PAIRED_NOUN_COLLAPSE_AUDIT",
       "source_real_S47_CI":EXPECTED_S47_CI,
       "S47_only_TRAIN_report":True,
       "NO_MODEL_TRAINING_NO_QA_NO_JESSICA_TEST":True,
       "LIVE_Render_Supabase_APK_UNMODIFIED":True,
       "not_used_to_select_original_S47_checkpoint":True,
       "source_S47_real_controller_physically_paid":True,
       "real_exact_full_equals_not_semantic_parity":True,
       "counts":counts,
       "dominant_free_texts":predictions.most_common(12),
       "per_predeclared_TRAIN_contrast":result,
       "interpretation":interpretation,
       "next_research_tactic":"S49: optimize contrast BOTH sides of a paired noun as one indivisible TRAIN target, require both full+EOS answers, preserve all historical 7, no regression in 34 first colors, and charge complete Green Tunnel/controller bytes within 0.7 bpp. Do not tune with S29 heldout or S26 inspected QA."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("input",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    raw=a.input.read_bytes()
    data=json.loads(raw)
    result=audit(data)
    result["source_S47_json_sha256"]=hashlib.sha256(raw).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"event":"S48_REAL_PAIRED_NOUN_COLLAPSE_AUDIT",
       **result["counts"],"interpretation":result["interpretation"]},
       ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
