"""Actual conditional answer accuracy and teacher choice agreement.

Gold label is never used in generation or a codebook update. No posthoc
exclusion of incorrect answers; all 16 questions per domain are reported.
"""
from torch.nn.functional import softmax
import torch
from v146.scoring import label_tokens,encode_prompt
from .protocol import COUNT,ARMS,DOMAINS,LABELS


def make_score(logits,label_ids,gold):
    if logits.ndim!=1 or not torch.isfinite(logits).all():
        raise ValueError("last-token scores invalid")
    if not 0<=gold<len(label_ids):
        raise ValueError("invalid gold option position")
    selected=logits[torch.tensor(label_ids,dtype=torch.long)].to(torch.float64)
    p=softmax(selected,dim=-1)
    win=int(selected.argmax())
    rest=selected.clone()
    rest[gold]=-torch.inf
    return {
        "chosen_index":win,
        "correct":win==gold,
        "gold_conditional_probability":float(p[gold]),
        "gold_minus_best_other_logit":float(selected[gold]-rest.max()),
        "conditional_choice_probabilities":[float(x) for x in p],
    }


def audit(all_arms):
    if tuple(all_arms)!=ARMS:
        raise ValueError("not the four locked FP32/CB59/half/paid arms")
    result={}
    for domain in DOMAINS:
        if any(len(all_arms[arm][domain])!=COUNT for arm in ARMS):
            raise ValueError("omitted new validation question")
        expected=[(r["row"],r["prompt_sha"]) for r in all_arms[ARMS[0]][domain]]
        if len(set(expected))!=COUNT or expected[0][0]!=400 or expected[-1][0]!=415:
            raise ValueError("changed fixed validation row IDs")
        for arm in ARMS[1:]:
            if [(r["row"],r["prompt_sha"]) for r in all_arms[arm][domain]]!=expected:
                raise ValueError("arms did not see identical source questions")
        summaries={}
        for arm in ARMS:
            cases=all_arms[arm][domain]
            summaries[arm]={
                "tasks_scored":COUNT,
                "correct":sum(r["correct"] for r in cases),
                "accuracy":sum(r["correct"] for r in cases)/COUNT,
                "mean_gold_conditional_probability":sum(r["gold_conditional_probability"] for r in cases)/COUNT,
                "teacher_choice_agreements":sum(
                    r["chosen_index"]==t["chosen_index"]
                    for r,t in zip(cases,all_arms["FP32"][domain])),
                "mean_gold_choice_margin":sum(r["gold_minus_best_other_logit"] for r in cases)/COUNT,
            }
        summaries["CB147_minus_CB59_correct"]=summaries["CB147_int8_paid"]["correct"]-summaries["CB59"]["correct"]
        summaries["CB147_minus_preprior_half_correct"]=summaries["CB147_int8_paid"]["correct"]-summaries["CB140_head_half"]["correct"]
        result[domain]=summaries
    return {
        "fixed_validation_only":result,
        "CB147_no_choice_accuracy_regression_in_both":all(
            result[d]["CB147_minus_CB59_correct"]>=0 for d in DOMAINS),
        "CB147_no_regression_against_half_head_in_both":all(
            result[d]["CB147_minus_preprior_half_correct"]>=0 for d in DOMAINS),
        "IMPORTANT":"Small 16+16 forced-choice validation only; never a complete QA, free-generation, independent TEST, or semantic pass.",
    }
