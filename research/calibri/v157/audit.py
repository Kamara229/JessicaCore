"""Strong per-domain and per-window gate resists a single outlier improvement."""
from math import comb
from .protocol import ARMS,DOMAINS,WINDOWS,WINDOW,MIN_IMPROVED_WINDOWS


def deltas(student,baseline,measure):
    if len(student)!=WINDOWS or len(baseline)!=WINDOWS:
        raise ValueError("pre-registered 32 source windows not fully scored")
    key=("sum_NLL" if measure=="nll" else "sum_teacher_to_student_KL"
         if measure=="kl" else None)
    if key is None:raise ValueError("only true NLL and FP32 teacher KL")
    if any(a["tokens_scored"]!=WINDOW-1 or b["tokens_scored"]!=WINDOW-1
           for a,b in zip(student,baseline)):
        raise ValueError("inconsistent actual gold token denominators")
    return [(a[key]-b[key])/(WINDOW-1) for a,b in zip(student,baseline)]


def exact_sign_tail(positive,N):
    if not 0<=positive<=N:raise ValueError("invalid source sign comparison")
    return sum(comb(N,i) for i in range(positive,N+1))/2**N


def paired_audit(student,baseline,metric):
    x=deltas(student,baseline,metric)
    sorted_x=sorted(x)
    wins=sum(v<0 for v in x)
    avg=sum(x)/len(x)
    # A favorable extreme negative delta was the v156 WikiText driver.
    # Drop the single LARGEST improvement (most negative window) up front.
    trimmed=(sum(x)-min(x))/(len(x)-1)
    return {
       "paired_32_window_deltas":x,
       "all_32_window_mean":avg,
       "median_window_delta":(sorted_x[15]+sorted_x[16])/2,
       "windows_improved":wins,
       "fraction_windows_improved":wins/32,
       "one_sided_sign_tail_exploratory":exact_sign_tail(wins,32),
       "worst_single_favorable_window_dropped_mean":trimmed,
       "predeclared_majority_min_improved":MIN_IMPROVED_WINDOWS,
       "robust_enough":bool(avg<0 and trimmed<0 and wins>=MIN_IMPROVED_WINDOWS),
    }


def summarize(aggregate,raw):
    if tuple(aggregate)!=ARMS or tuple(raw)!=ARMS:
        raise ValueError("not exact preregistered original model arms")
    result={}
    for domain in DOMAINS:
        if any(len(raw[arm][domain])!=WINDOWS for arm in ARMS):
            raise ValueError("missing original source fourth valid windows")
        base=aggregate["native_CB59"][domain]
        refined=aggregate["paid_layer6_v_only"][domain]
        other=aggregate["paid_layer6_o_only"][domain]
        all7=aggregate["paid_layer6_all7"][domain]
        x_nll=paired_audit(raw["paid_layer6_v_only"][domain],raw["native_CB59"][domain],"nll")
        x_kl=paired_audit(raw["paid_layer6_v_only"][domain],raw["native_CB59"][domain],"kl")
        result[domain]={
           "v_proj_minus_CB59_gold_NLL":refined["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
           "v_proj_minus_CB59_teacher_KL":refined["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
           "v_proj_teacher_argmax_count":int(round(
               refined["teacher_next_token_argmax_agreement_fraction"]*refined["tokens_scored"])),
           "o_proj_minus_CB59_gold_NLL":other["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
           "all7_minus_CB59_gold_NLL":all7["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
           "per_window_NLL_robustness":x_nll,
           "per_window_teacher_KL_robustness":x_kl,
           "precommitted_STRONG_pass":x_nll["robust_enough"] and x_kl["robust_enough"],
        }
    return {
        "v_proj_STRONG_two_domain_robustness_pass":all(
            result[d]["precommitted_STRONG_pass"] for d in DOMAINS),
        "honest_fourth_NEW_source_results":result,
        "even_if_passes_still_NO_true_semantic_QA_or_good_language":True,
        "sign_tail_descriptive_not_independent_observation_p_value":True,
    }
