"""Explicit semantic regression reporting: count correct and FP32-choice agreement."""
from v150.protocol import ARMS,DATASETS,QUESTION_COUNT,ROW_BEGIN,ROW_END


def summarize(rows):
    if tuple(rows)!=ARMS:
        raise ValueError("incorrect or incomplete precommitted model arms")
    outputs={}
    for domain in DATASETS:
        controls=rows["FP32"][domain]
        expected=[(r["question_row"],r["prompt_SHA256"]) for r in controls]
        if expected!=[(r,s) for r,s in expected] or (
                len(expected)!=QUESTION_COUNT or
                [r for r,s in expected]!=list(range(ROW_BEGIN,ROW_END)) or
                len(set(expected))!=QUESTION_COUNT):
            raise ValueError("incomplete or duplicated source validation questions")
        scores={}
        for arm in ARMS:
            values=rows[arm][domain]
            if [(r["question_row"],r["prompt_SHA256"]) for r in values]!=expected:
                raise ValueError("non-matched paired validation sources")
            scores[arm]={
                "correct":sum(v["correct"] for v in values),
                "questions":QUESTION_COUNT,
                "forced_choice_accuracy":sum(v["correct"] for v in values)/QUESTION_COUNT,
                "teacher_forced_choice_agreement":sum(
                    v["chosen_index"]==t["chosen_index"]
                    for v,t in zip(values,controls)),
                "mean_gold_choice_probability":sum(
                    v["gold_conditional_probability"] for v in values)/QUESTION_COUNT,
                "mean_gold_margin":sum(
                    v["gold_minus_best_other_logit"] for v in values)/QUESTION_COUNT,
            }
        scores["CB149_minus_CB147_correct"]=(
            scores["CB149_contextual"]["correct"]-scores["CB147_prior"]["correct"])
        scores["CB149_minus_CB59_correct"]=(
            scores["CB149_contextual"]["correct"]-scores["CB59"]["correct"])
        outputs[domain]=scores
    no_regression=all(outputs[d]["CB149_minus_CB147_correct"]>=0 for d in DATASETS)
    return {
        "fresh_SciQ_and_BoolQ_validation":outputs,
        "contextual_int4_no_accuracy_regression_in_both":no_regression,
        "semantic_quality_still_UNPROVEN_even_if_true":True,
        "not_frozen_v89_TEST_or_full_free_response_benchmark":True,
    }
