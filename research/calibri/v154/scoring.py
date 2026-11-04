"""Compare teacher KL + argmax, count all matched diagnostic prompts."""
from v152.scoring import aggregate as first_token_summary
from .protocol import ARMS,MATRIX_KEYS


def aggregate(rows,elapsed):
    if tuple(rows)!=ARMS or tuple(elapsed)!=ARMS:
        raise ValueError("not full preregistered 16-arm original operator ablation")
    if any(len(rows[key])!=8 for key in ARMS):
        raise ValueError("unpaired fixed first-token probes")
    base={k:sum(x["teacher_to_student_KL"] for x in rows[k])/8
          for k in ARMS}
    original=base["CB59_original"]
    full=base["CB153_all7"]
    each=[]
    for key in MATRIX_KEYS:
        short=key.split(".")[-2]
        each.append({
            "original_operator":key,
            "single_operator_KL_vs_CB59":base[f"only_{short}"]-original,
            "leave_one_out_KL_vs_complete_CB153":base[f"except_{short}"]-full,
            "any_single_operator_teacher_argmax_match":sum(
                x["FP32_first_token_argmax_agreement"] for x in rows[f"only_{short}"]),
            "cannot_infer_this_operator_alone_is_sufficient":True,
        })
    return {
        "original_CB59_mean_teacher_KL":original,
        "complete_CB153_mean_teacher_KL":full,
        "all7_KL_delta_vs_CB59":full-original,
        "per_original_operator_diagnostic":
            sorted(each,key=lambda x:x["single_operator_KL_vs_CB59"]),
        "all_16_arm_means":base,
        "all_16_arm_full_CPU_seconds":elapsed,
        "no_new_VALID_TEST_or_QA":True,
        "not_proof_that_subsets_generalize_to_new_data":True,
    }
