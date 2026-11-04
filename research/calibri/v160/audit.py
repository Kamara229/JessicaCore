"""Matched head vs embedding effects on real source next-token NLL/KL."""
from v142.scoring import aggregate
from .protocol import ARMS,DOMAINS,WINDOWS,WINDOW


def summarize(per_windows):
    if tuple(per_windows)!=ARMS:
        raise ValueError("not exactly frozen physical source gain arms")
    total=WINDOWS*(WINDOW-1)
    aggregated={}
    for arm in ARMS:
        aggregated[arm]={}
        for domain in DOMAINS:
            rows=per_windows[arm][domain]
            if len(rows)!=WINDOWS or any(row["tokens_scored"]!=WINDOW-1 for row in rows):
                raise ValueError("lost paired original source next-token measurements")
            value=aggregate(rows)
            if value["tokens_scored"]!=total:
                raise ValueError("scored tokens changed")
            aggregated[arm][domain]=value

    differences={}
    base=ARMS[0]
    for domain in DOMAINS:
        ref=aggregated[base][domain]
        differences[domain]={}
        for arm in ARMS[1:]:
            t=aggregated[arm][domain]
            differences[domain][arm]={
                "vs_CB59_TRUE_next_token_NLL":t["mean_TRUE_next_token_NLL"]-ref["mean_TRUE_next_token_NLL"],
                "vs_CB59_FP32_teacher_KL":t["teacher_to_candidate_KL"]-ref["teacher_to_candidate_KL"],
                "teacher_next_token_argmax_agreement":t["teacher_next_token_argmax_agreement_fraction"],
            }
        differences[domain]["input_effect_with_HEAD0_vs_HEAD1"]={
            "delta_NLL_input_only_minus_baseline":
                aggregated[ARMS[1]][domain]["mean_TRUE_next_token_NLL"]-ref["mean_TRUE_next_token_NLL"],
            "delta_NLL_paired_minus_headonly":
                aggregated[ARMS[3]][domain]["mean_TRUE_next_token_NLL"]-
                aggregated[ARMS[2]][domain]["mean_TRUE_next_token_NLL"],
        }
    input_head0=ARMS[1]
    input_quality_both=all(
        differences[d][input_head0]["vs_CB59_TRUE_next_token_NLL"]<0 and
        differences[d][input_head0]["vs_CB59_FP32_teacher_KL"]<=0
        for d in DOMAINS)
    return {
        "true_source_next_token_FP32_teacher_and_golden_metrics":aggregated,
        "paired_per_domain_causal_input_head_effects":differences,
        "paid_input_ONLY_improves_NLL_and_KL_on_both_new_domains":input_quality_both,
        "do_not_claim_semantics_QA_or_compressed_release_from_this_test":True,
    }
