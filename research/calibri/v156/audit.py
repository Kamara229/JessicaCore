"""Unhidden per-window, per-domain improvement sign and uncertainty controls."""
from math import comb
from .protocol import ARMS,DOMAINS,WINDOW,WINDOWS


def exact_sign_tail(improved,total):
    """One-sided under symmetric signs, exploratory only, NOT model confidence."""
    if total<1 or not 0<=improved<=total:
        raise ValueError("invalid paired window signs")
    return sum(comb(total,k) for k in range(improved,total+1))/(2**total)


def paired_window_stats(student,baseline):
    if len(student)!=WINDOWS or len(baseline)!=WINDOWS:
        raise ValueError("fixed eight source windows required for paired validation")
    if any(a["tokens_scored"]!=WINDOW-1 or b["tokens_scored"]!=WINDOW-1
           for a,b in zip(student,baseline)):
        raise ValueError("unequal next-token denominators")
    d_nll=[(a["sum_NLL"]-b["sum_NLL"])/(WINDOW-1)
           for a,b in zip(student,baseline)]
    d_kl=[(a["sum_teacher_to_student_KL"]-
           b["sum_teacher_to_student_KL"])/(WINDOW-1)
          for a,b in zip(student,baseline)]
    nll_better=sum(x<0 for x in d_nll)
    kl_better=sum(x<0 for x in d_kl)
    return {
        "paired_window_mean_NLL_deltas":d_nll,
        "paired_window_mean_teacher_KL_deltas":d_kl,
        "NLL_better_windows_of_8":nll_better,
        "KL_better_windows_of_8":kl_better,
        "NLL_improvement_exact_sign_tail_exploratory":exact_sign_tail(nll_better,WINDOWS),
        "KL_improvement_exact_sign_tail_exploratory":exact_sign_tail(kl_better,WINDOWS),
    }


def summarize(aggregate,raw):
    if tuple(aggregate)!=ARMS or tuple(raw)!=ARMS:
        raise ValueError("missing complete preregistered original source arms")
    domains={}
    for domain in DOMAINS:
        if any(len(raw[arm][domain])!=WINDOWS for arm in ARMS):
            raise ValueError("one predeclared validation window not scored")
        base=aggregate["native_CB59"][domain]
        v=aggregate["paid_v_proj_only"][domain]
        o=aggregate["paid_o_proj_only"][domain]
        all7=aggregate["paid_all7"][domain]
        pair=paired_window_stats(raw["paid_v_proj_only"][domain],
                                 raw["native_CB59"][domain])
        domains[domain]={
            "v_proj_minus_CB59_NLL":v["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
            "v_proj_minus_CB59_KL":v["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
            "o_proj_minus_CB59_NLL":o["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
            "all7_minus_CB59_NLL":all7["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
            "v_proj_teacher_next_token_top1_agreement":v["teacher_next_token_argmax_agreement_fraction"],
            "paired_eight_windows":pair,
        }
    passed=all(x["v_proj_minus_CB59_NLL"]<0 and
               x["v_proj_minus_CB59_KL"]<=0 for x in domains.values())
    return {
        "v_proj_improves_NLL_and_FP32_teacher_KL_on_both_third_holdout_domains":passed,
        "new_domain_deltas_and_pairing":domains,
        "HOLDOUT_selected_AFTER_v155_so_not_an_independent_TEST":True,
        "full_original_frozen_v89_TEST_NOT_touched":True,
        "even_PASS_would_not_restore_semantic_QA_or_language":True,
    }
