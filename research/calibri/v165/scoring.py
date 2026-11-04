"""Teacher top-1 in packed top-K: exact optimistic Green Tunnel upper bound.

If the teacher top1 token is outside a student's existing topK, NO
reranker limited to those K output IDs can recover this teacher choice,
even if it knows perfect original teacher preference for those K.
The upper bound is NOT an actual trained Green Tunnel; assigning
teacher top1 is an unpaid oracle and cannot prove semantic quality.

For equal packed logit ties, strict greater-than ranks are an OPTIMISTIC
minimum rank; the actual implementation might lose ties.
"""
import torch
from .protocol import TOPK,VOCAB,WINDOW,TOKENS_PER_DOMAIN


@torch.inference_mode()
def one_window(teacher,student,ids):
    if (teacher.shape,student.shape)!=( (1,WINDOW,VOCAB),(1,WINDOW,VOCAB) ):
        raise ValueError("original 135m full 49152 vocab not matched")
    if ids.shape!=(1,WINDOW):
        raise ValueError("original matching gold context window missing")
    if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
        raise ValueError("source next-token logits contain inf or nan")
    t=teacher[0,:-1,:].to(torch.float32)
    s=student[0,:-1,:].to(torch.float32)
    target_teacher=t.argmax(dim=-1).to(torch.long)
    target_gold=ids[0,1:].to(torch.long)
    rows=torch.arange(WINDOW-1)
    student_at_teacher=s[rows,target_teacher]
    student_at_gold=s[rows,target_gold]
    rank_teacher=1+(s>student_at_teacher[:,None]).sum(-1)
    rank_gold=1+(s>student_at_gold[:,None]).sum(-1)
    tied_teacher=(s==student_at_teacher[:,None]).sum(-1)
    teacher_best=student_at_teacher
    gap_best=(s.max(-1).values-teacher_best)
    return {
        "tokens":WINDOW-1,
        "strict_gt_rank_for_teacher_TOP1_optimistic_ties":rank_teacher.cpu().tolist(),
        "strict_gt_rank_for_true_GOLD_token_optimistic_ties":rank_gold.cpu().tolist(),
        "student_logit_gap_to_original_teacher_TOP1":gap_best.cpu().tolist(),
        "student_logit_ties_at_teacher_TOP1":tied_teacher.cpu().tolist(),
        "student_TOP1_matches_original_teacher_TOP1":
            int((rank_teacher==1).sum().item()),
        "student_TOP1_is_true_gold_token":
            int((s.argmax(-1)==target_gold).sum().item()),
        "teacher_TOP1_is_true_gold_token":
            int((target_teacher==target_gold).sum().item()),
        "teacher_original_TOP1_ids_sha_material":[int(x) for x in target_teacher.cpu().tolist()],
        "original_true_GOLD_ids_sha_material":[int(x) for x in target_gold.cpu().tolist()],
        "an_unpaid_teacher_oracle_is_not_a_trained_reranker":True,
    }


def aggregate(rows):
    if len(rows)!=4 or any(x["tokens"]!=WINDOW-1 for x in rows):
        raise ValueError("missing four source TRAIN FIT forward windows")
    ranks_teacher=[r for x in rows
                   for r in x["strict_gt_rank_for_teacher_TOP1_optimistic_ties"]]
    ranks_gold=[r for x in rows
                for r in x["strict_gt_rank_for_true_GOLD_token_optimistic_ties"]]
    if len(ranks_teacher)!=TOKENS_PER_DOMAIN or len(ranks_gold)!=TOKENS_PER_DOMAIN:
        raise ValueError("incomplete original 252 next-token positions")
    teacher_top1=sum(x["student_TOP1_matches_original_teacher_TOP1"] for x in rows)
    positive_gold=sum(x["student_TOP1_is_true_gold_token"] for x in rows)
    real_teacher_gold=sum(x["teacher_TOP1_is_true_gold_token"] for x in rows)
    support={}
    for k in TOPK:
        a=sum(r<=k for r in ranks_teacher)
        b=sum(r<=k for r in ranks_gold)
        support[str(k)]={
            "teacher_TOP1_recoverable_by_UNPAID_PERFECT_TOPK_oracle_UPPER_BOUND":a,
            "teacher_TOP1_recovery_upper_bound_fraction":a/TOKENS_PER_DOMAIN,
            "true_gold_in_student_topK_oracle_upper_bound_count":b,
            "gold_in_student_topK_upper_bound_fraction":b/TOKENS_PER_DOMAIN,
            "topK_reranker_can_never_recover_missed_teacher_TOP1_count":
                TOKENS_PER_DOMAIN-a,
        }
    return {
        "total_actual_next_token_positions":TOKENS_PER_DOMAIN,
        "original_student_teacher_TOP1_agreement":teacher_top1,
        "original_student_true_gold_TOP1_match":positive_gold,
        "original_teacher_TOP1_is_TRUE_gold":real_teacher_gold,
        "optimistic_tie_handling_important":True,
        "topK_support_information_theoretic_reranker_ceiling_not_performance":support,
        "largest_optimistic_tied_teacher_logit_count":
            max(max(x["student_logit_ties_at_teacher_TOP1"]) for x in rows),
        "mean_logit_gap_to_teacher_top1":sum(
            v for x in rows for v in x["student_logit_gap_to_original_teacher_TOP1"])/TOKENS_PER_DOMAIN,
        "not_an_actual_Green_Tunnel_run":True,
        "unpaid_FP32_teacher_oracle_not_a_0p7bpp_candidate":True,
        "no_quality_or_conversational_recovery_claim":True,
    }
