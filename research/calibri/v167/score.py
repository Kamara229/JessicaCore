"""Behavior-first comparison. No KL, NLL, logits-to-teacher, or PPL gates."""
import math
import re
from collections import Counter
from .protocol import ARMS,DOMAINS,counts

def wilson(k,n):
    if n<1:raise ValueError("empty evaluation")
    z=1.96
    p=k/n
    c=(p+z*z/(2*n))/(1+z*z/n)
    d=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/(1+z*z/n)
    return [round(max(0,c-d),5),round(min(1,c+d),5)]

def choices(tasks,results):
    if tuple(results)!=ARMS:
        raise ValueError("changed model order or missing FP32")
    out={}
    for dom in DOMAINS:
        cases=tasks[dom]
        if len(cases)!=counts()[dom]:
            raise ValueError("missing preregistered questions")
        for arm in ARMS:
            records=results[arm][dom]
            if len(records)!=len(cases):
                raise ValueError("silent questions dropped")
            for case,row in zip(cases,records):
                if row["prompt_sha"]!=case.sha or row["index"]!=case.index:
                    raise ValueError("unpaired example or reordered dataset")
        per={}
        teacher=results["FP32"][dom]
        for arm in ARMS:
            rows=results[arm][dom]
            hits=sum(bool(x["correct"]) for x in rows)
            per[arm]={
                "questions":len(rows),"correct":hits,"accuracy":hits/len(rows),
                "wilson95_descriptive":wilson(hits,len(rows)),
                "chosen_label_histogram":dict(Counter(x["chosen_label"] for x in rows)),
                "agree_with_FP32_choices":sum(x["chosen_index"]==t["chosen_index"] for x,t in zip(rows,teacher)),
            }
        for arm in ARMS[1:]:
            rows=results[arm][dom]
            preserved=sum(t["correct"] and s["correct"] for t,s in zip(teacher,rows))
            lost=sum(t["correct"] and not s["correct"] for t,s in zip(teacher,rows))
            gained=sum(not t["correct"] and s["correct"] for t,s in zip(teacher,rows))
            both_wrong=sum(not t["correct"] and not s["correct"] for t,s in zip(teacher,rows))
            per[arm]["FP32_correct_preserved"]=preserved
            per[arm]["FP32_correct_lost"]=lost
            per[arm]["FP32_wrong_but_packed_correct"]=gained
            per[arm]["both_incorrect"]=both_wrong
            per[arm]["retention_conditional_on_FP32_correct"]=(preserved/per["FP32"]["correct"]
                if per["FP32"]["correct"] else None)
            per[arm]["net_correct_delta_vs_FP32"]=gained-lost
        out[dom]=per
    return {"by_domain":out,"physical_quality_gate":"NOT_DECIDED_BY_KL_NLL",
            "interpretation":"Measured capability of this FP32 teacher on specified prompts only, not an upper bound on all possible tasks. Preserve FP32 successes; separately report compressed correct when FP32 fails.",
            "no_scoring_by_teacher_probability_distribution":True}

def parse_strict(text,gold):
    """Deliberately strict automatic lexical check; do not call semantic review."""
    if gold is None:
        return None
    tokens=re.findall(r"[a-z0-9]+",text.casefold())
    expected=re.findall(r"[a-z0-9]+",gold.casefold())
    # Correct if the *whole* answer consists of just the reference, ignoring punctuation.
    return tokens==expected

def generation(probes,results):
    if tuple(results)!=ARMS:
        raise ValueError("missing FP32 actual-generation control")
    for arm in ARMS:
        if len(results[arm])!=len(probes):
            raise ValueError("model skipped some free-response examples")
        for q,r in zip(probes,results[arm]):
            if r["prompt_sha"]!=q["sha"] or r["domain"]!=q["domain"]:
                raise ValueError("generation mismatch between arms")
    out={}
    for arm in ARMS:
        rows=results[arm]
        counts_by_domain=Counter(x["domain"] for x in rows)
        by_dom={}
        for dom in counts_by_domain:
            selected=[x for x in rows if x["domain"]==dom]
            graded=[x for x in selected if x["strict_exact_gold"] is not None]
            by_dom[dom]={
                "questions":len(selected),
                "strict_exact_gold_count":sum(x["strict_exact_gold"] for x in graded),
                "strict_exact_denominator":len(graded),
                "generated_token_count":sum(x["token_count"] for x in selected),
                "eos_count":sum(x["eos_generated"] for x in selected),
                "mean_distinct_token_fraction":sum(x["distinct_ratio"] for x in selected)/len(selected),
            }
        total_tokens=sum(x["token_count"] for x in rows)
        total_wall=sum(x["full_wall_seconds"] for x in rows)
        if not 0<total_wall or not total_tokens:
            raise ValueError("invalid real model timing")
        out[arm]={
            "by_domain":by_dom,"total_tokens":total_tokens,
            "total_full_model_seconds":total_wall,
            "measured_full_model_tokens_per_second":total_tokens/total_wall,
            "mean_distinct_ratio":sum(x["distinct_ratio"] for x in rows)/len(rows),
            "complete_actual_texts_and_ids_recorded":True,
        }
    baseline=out["CB59"]["measured_full_model_tokens_per_second"]
    out["CB161"]["speed_ratio_vs_CB59"]=out["CB161"]["measured_full_model_tokens_per_second"]/baseline
    out["FP32"]["speed_ratio_vs_CB59"]=out["FP32"]["measured_full_model_tokens_per_second"]/baseline
    pair_a=results["FP32"]
    for arm in ARMS[1:]:
        pair_b=results[arm]
        out[arm]["same_strict_correct_as_FP32"]=sum(
            a["strict_exact_gold"] is True and b["strict_exact_gold"] is True
            for a,b in zip(pair_a,pair_b))
        out[arm]["FP32_strict_correct_lost"]=sum(
            a["strict_exact_gold"] is True and b["strict_exact_gold"] is False
            for a,b in zip(pair_a,pair_b))
        out[arm]["FP32_strict_wrong_packed_right"]=sum(
            a["strict_exact_gold"] is False and b["strict_exact_gold"] is True
            for a,b in zip(pair_a,pair_b))
    return {"by_arm":out,"strict_exact_is_NOT_a_semantic_human_grade":True,
            "all_48_verbatim_outputs_in_report":True,
            "single_CPU_RUN_not_hardware_generalization":True,
            "no_KL_NLL_PPL_for_quality_or_selection":True}
