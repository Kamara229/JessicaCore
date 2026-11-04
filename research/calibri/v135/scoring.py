"""Ablation table from frozen FP32 teacher logits and hidden activations."""
from statistics import mean

def summarize(arms):
    expected=("CB59","CB59_FP32_input","CB59_FP32_head",
              "CB59_FP32_input_and_head","v119_neutral",
              "v119_FP32_input","v119_no_0to4","v119_no_0to4_FP32_input")
    if tuple(arms)!=expected: raise ValueError("missing or reordered probes")
    result={}
    for arm,records in arms.items():
        if len(records)!=16: raise ValueError("incomplete matching prompts")
        result[arm]={
            "mean_teacher_to_student_KL":mean(r["teacher_to_student_KL"] for r in records),
            "first_EOS_count":sum(r["student_first_EOS"] for r in records),
            "teacher_first_token_argmax_match":sum(r["argmax_equal"] for r in records),
            "mean_embedding_relative_L2":mean(r["layer_trajectory"][0]["relative_L2"]
                                               for r in records),
            "mean_layer0_relative_L2":mean(r["layer_trajectory"][1]["relative_L2"]
                                            for r in records),
        }
    result["interpretation_limit"]="FP32 oracle matrix substitutions are NOT deployable and do NOT meet 0.7bpp; this is fault-localization only, and KL/EOS are NOT QA accuracy."
    return result
