"""KL/argmax on identical predefined first-token probes; no gold used."""
from statistics import mean
import torch
from torch.nn.functional import log_softmax

def comparison(reference,student,eos):
    teacher=reference.to(torch.float64).flatten()
    trial=student.to(torch.float64).flatten()
    if teacher.shape!=trial.shape or teacher.numel()!=49152:
        raise ValueError("wrong original 135M output logits")
    if not teacher.isfinite().all() or not trial.isfinite().all():
        raise ValueError("nonfinite source output")
    logp=log_softmax(teacher,dim=-1)
    logq=log_softmax(trial,dim=-1)
    kl=float((logp.exp()*(logp-logq)).sum())
    if kl< -1e-6:
        raise ValueError("negative teacher KL")
    return {
        "teacher_to_student_KL":max(kl,0.0),
        "FP32_first_token_argmax_agreement":
            int(teacher.argmax())==int(trial.argmax()),
        "first_EOS":int(trial.argmax())==eos,
        "teacher_first_argmax":int(teacher.argmax()),
        "candidate_first_argmax":int(trial.argmax()),
    }


def aggregate(arms):
    from .protocol import ARMS,DIAGNOSTIC_QUESTION_INDICES
    if tuple(arms)!=ARMS:
        raise ValueError("incomplete fixed original 30-block diagnostic")
    length=len(DIAGNOSTIC_QUESTION_INDICES)
    result={}
    for arm,rows in arms.items():
        if len(rows)!=length:
            raise ValueError("unmatched original eight prompts")
        result[arm]={
            "mean_FP32_teacher_KL":mean(x["teacher_to_student_KL"] for x in rows),
            "FP32_first_token_agreement_count":
                sum(x["FP32_first_token_argmax_agreement"] for x in rows),
            "first_EOS_count":sum(x["first_EOS"] for x in rows),
        }
    original=result["genuine_packed_CB59"]["mean_FP32_teacher_KL"]
    layers=[]
    for index in range(30):
        name=f"oracle_FP32_layer_{index:02d}"
        stats=result[name]
        layers.append({
            "layer":index,"borrowed_F32_single_layer":True,
            "mean_KL_delta_vs_original_CB59":
                stats["mean_FP32_teacher_KL"]-original,
            "teacher_argmax_match_delta_vs_CB59":
                stats["FP32_first_token_agreement_count"]-
                result["genuine_packed_CB59"]["FP32_first_token_agreement_count"],
        })
    return {
        "controls":result,
        "all_30_single_layer_FP32_oracle_ablation_ranking":
            sorted(layers,key=lambda x:x["mean_KL_delta_vs_original_CB59"]),
        "not_a_validated_rate_allocation_or_semantic_QA_result":True,
        "all_FP32_layer_oracle_arms_borrow_UNPAID_large_dense_tensors":True,
        "NOT_0_7bpp_except_genuine_CB59_control":True,
    }
