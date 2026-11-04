"""Teacher-aligned predeclared metrics on the identical 16 context questions."""
from statistics import mean
from .oracles import ARMS


def summary(arms):
    if tuple(arms)!=ARMS:
        raise ValueError("missing/extra oracle arms or different order")
    reports={}
    for name,records in arms.items():
        if len(records)!=16:
            raise ValueError("incomplete paired question set")
        reports[name]={
            "mean_FP32_teacher_KL":mean(r["teacher_to_student_KL"] for r in records),
            "first_argmax_matches_teacher":sum(r["argmax_equal"] for r in records),
            "first_EOS":sum(r["student_first_EOS"] for r in records),
            "mean_relative_embedding_error":mean(
                r["layer_trajectory"][0]["relative_L2"] for r in records),
            "mean_relative_layer0_error":mean(
                r["layer_trajectory"][1]["relative_L2"] for r in records),
            "mean_relative_layer4_error":mean(
                r["layer_trajectory"][5]["relative_L2"] for r in records),
            "mean_relative_layer29_error":mean(
                r["layer_trajectory"][30]["relative_L2"] for r in records),
        }
    return {
        "by_arm":reports,
        "limits":"Read-only pretrained FP32 matrix oracle diagnostic: NOT 0.7bpp, NOT a new model, NOT QA pass, NOT measured runtime speed.",
    }
