"""Paired error slices, margin shifts and generation collapse; no selection on TEST."""
from __future__ import annotations

def paired_diagnosis(base, corrected):
    if not base or len(base) != len(corrected):
        raise AssertionError("diagnosis requires nonempty paired samples")
    def identity(x): return x["record_index"], x["gold"], x["prompt_SHA"]
    if list(map(identity, base)) != list(map(identity, corrected)):
        raise AssertionError("CB59 and CB88 evaluated different TRAIN inputs")
    categories = {"both_correct": 0, "CB59_only_correct": 0,
                  "CB88_only_correct": 0, "both_wrong": 0}
    regressions, improvements = [], []
    by_gold = {"yes": {"base":0,"cb88":0,"count":0},
               "no": {"base":0,"cb88":0,"count":0}}
    signed_margin_changes=[]
    for a, b in zip(base, corrected):
        category = ("both_correct" if a["correct"] and b["correct"] else
                    "CB59_only_correct" if a["correct"] else
                    "CB88_only_correct" if b["correct"] else "both_wrong")
        categories[category] += 1
        gold_label = "yes" if a["gold"] == 0 else "no"
        by_gold[gold_label]["count"] += 1
        by_gold[gold_label]["base"] += a["correct"]
        by_gold[gold_label]["cb88"] += b["correct"]
        sign = 1 if a["gold"] == 0 else -1
        signed_margin_changes.append(sign*(b["yes_no_logit_margin"]-a["yes_no_logit_margin"]))
        row = {"record_index": a["record_index"], "prompt_SHA": a["prompt_SHA"],
               "gold": a["gold"], "CB59_pred": a["pred"], "CB88_pred": b["pred"],
               "CB59_signed_margin": sign*a["yes_no_logit_margin"],
               "CB88_signed_margin": sign*b["yes_no_logit_margin"],
               "CB59_confidence": a["prediction_confidence"],
               "CB88_confidence": b["prediction_confidence"]}
        if category == "CB59_only_correct": regressions.append(row)
        if category == "CB88_only_correct": improvements.append(row)
    n=len(base)
    return {"count": n, "CB59_correct": sum(r["correct"] for r in base),
            "CB88_correct": sum(r["correct"] for r in corrected),
            "categories": categories, "by_original_gold": by_gold,
            "mean_gold_signed_margin_shift_CB88_minus_CB59": sum(signed_margin_changes)/n,
            "regressed_records": regressions, "recovered_records": improvements,
            "all_example_level_predictions_and_confidence": {"CB59": base, "CB88": corrected}}

def greedy_summary(sequences):
    if not sequences or any(not x for x in sequences):
        raise AssertionError("nonempty generation samples required")
    scored=[]
    for tokens in sequences:
        ratio=len(set(tokens))/len(tokens)
        largest=max(tokens.count(t) for t in set(tokens))/len(tokens)
        scored.append({"generated_ids": tokens, "distinct_ratio": ratio,
                       "most_frequent_share": largest, "repeats": len(tokens)-len(set(tokens))})
    return {"mean_distinct_ratio": sum(x["distinct_ratio"] for x in scored)/len(scored),
            "mean_most_frequent_share": sum(x["most_frequent_share"] for x in scored)/len(scored),
            "prompts": scored}
