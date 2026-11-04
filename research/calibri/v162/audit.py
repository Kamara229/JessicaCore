"""Paired 32-window NLL/KL: reject lucky one-window aggregate gains.

The gate is preregistered, cannot be tuned to the actual source outcomes:
both corpora, both true NLL and FP32 teacher KL must improve mean, at
least 20/32 windows must improve each metric in each corpus, and dropping
the SINGLE most favorable NLL or KL window must not reverse that gain.
Sign tails are exploratory only: windows from natural text are dependent.
"""
import math
from v142.scoring import aggregate
from .protocol import (DOMAINS,NATIVE_ARMS,WINDOW,WINDOWS,
                       SCORED_TOKENS_EACH,MIN_BETTER_WINDOWS)


def paired(student,base,metric):
    if len(student)!=WINDOWS or len(base)!=WINDOWS:
        raise ValueError("all 32 windows from both source arms must be present")
    if metric not in ("true_NLL","FP32_teacher_KL"):
        raise ValueError("only real original teacher/gold quality can be audited")
    field="sum_NLL" if metric=="true_NLL" else "sum_teacher_to_student_KL"
    if any(x["tokens_scored"]!=WINDOW-1 or y["tokens_scored"]!=WINDOW-1
           for x,y in zip(student,base)):
        raise ValueError("mismatched physical input tokens")
    deltas=[(x[field]-y[field])/(WINDOW-1)
            for x,y in zip(student,base)]
    if any(not math.isfinite(x) for x in deltas):
        raise ValueError("source holdout produced invalid comparison")
    n_improved=sum(x<0 for x in deltas)
    mean=sum(deltas)/WINDOWS
    retained=(sum(deltas)-min(deltas))/(WINDOWS-1)
    ordered=sorted(deltas)
    robust=bool(mean<0 and retained<0 and n_improved>=MIN_BETTER_WINDOWS)
    return {
       "paired_32_per_window_delta":deltas,
       "mean_candidate_minus_CB59":mean,
       "median_candidate_minus_CB59":(ordered[15]+ordered[16])/2,
       "improved_windows_out_of_32":n_improved,
       "after_excluding_single_strongest_improvement_mean_delta":retained,
       "at_least_20_of_32_help_and_trimmed_mean_help":robust,
       "one_sided_window_sign_tail_exploratory_not_a_p_value":
           sum(math.comb(WINDOWS,i) for i in range(n_improved,WINDOWS+1))/2**WINDOWS,
    }


def summarize(measurements,timings):
    if tuple(measurements)!=NATIVE_ARMS or tuple(timings)!=NATIVE_ARMS:
        raise ValueError("precommitted original CB59 and paid first-block only")
    aggregate_by_arm={}
    for arm in NATIVE_ARMS:
        aggregate_by_arm[arm]={}
        if tuple(measurements[arm])!=DOMAINS:
            raise ValueError("missing original text corpus")
        for domain in DOMAINS:
            rows=measurements[arm][domain]
            if len(rows)!=WINDOWS or any(row["tokens_scored"]!=WINDOW-1 for row in rows):
                raise ValueError("original natural text tokens were skipped")
            summary=aggregate(rows)
            if summary["tokens_scored"]!=SCORED_TOKENS_EACH:
                raise ValueError("wrong independent next-token count")
            aggregate_by_arm[arm][domain]=summary

    deltas={}
    for domain in DOMAINS:
        a=aggregate_by_arm[NATIVE_ARMS[0]][domain]
        b=aggregate_by_arm[NATIVE_ARMS[1]][domain]
        nll=paired(measurements[NATIVE_ARMS[1]][domain],
                   measurements[NATIVE_ARMS[0]][domain],"true_NLL")
        kl=paired(measurements[NATIVE_ARMS[1]][domain],
                  measurements[NATIVE_ARMS[0]][domain],"FP32_teacher_KL")
        mean_nll=b["mean_TRUE_next_token_NLL"]-a["mean_TRUE_next_token_NLL"]
        mean_kl=b["teacher_to_candidate_KL"]-a["teacher_to_candidate_KL"]
        if abs(nll["mean_candidate_minus_CB59"]-mean_nll)>1e-4 or (
                abs(kl["mean_candidate_minus_CB59"]-mean_kl)>1e-4):
            raise ValueError("source aggregate and paired windows disagree")
        deltas[domain]={
            "NLL_new_minus_original_CB59":mean_nll,
            "teacher_KL_new_minus_original_CB59":mean_kl,
            "new_PPL_over_CB59_PPL":b["perplexity"]/a["perplexity"],
            "teacher_top1_matches_new_out_of_2016":int(
                b["exact_FP32_next_token_argmax_count"]),
            "teacher_top1_matches_original_CB59_out_of_2016":int(
                a["exact_FP32_next_token_argmax_count"]),
            "paired_TRUE_NLL":nll,
            "paired_teacher_KL":kl,
            "original_full_model_CPU_seconds":timings[NATIVE_ARMS[0]][domain],
            "paid_full_model_CPU_seconds":timings[NATIVE_ARMS[1]][domain],
            "rough_single_trial_full_model_CPU_speed_ratio":(
                timings[NATIVE_ARMS[0]][domain]/
                max(timings[NATIVE_ARMS[1]][domain],1e-9)),
            "per_domain_robust_gate":nll["at_least_20_of_32_help_and_trimmed_mean_help"]
               and kl["at_least_20_of_32_help_and_trimmed_mean_help"],
        }
    return {
      "both_domain_precommitted_STRONG_NLL_KL_robustness_pass":all(
          deltas[d]["per_domain_robust_gate"] for d in DOMAINS),
      "per_domain_robustness":deltas,
      "real_paired_true_NLL_PPL_KL_and_top1":aggregate_by_arm,
      "even_STRONG_VALID_PASS_NOT_automatically_good_semantic_QA":True,
      "this_source_is_post_selection_VALID_not_frozen_independent_TEST":True,
      "CPU_speed_is_single_noisy_trial_NOT_a_90_percent_release_benchmark":True,
    }
