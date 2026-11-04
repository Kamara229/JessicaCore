"""Compare genuinely fully packed model vs separately UNPAID diagnostic oracle."""
from v142.scoring import aggregate
from .protocol import ARMS,PHYSICAL_ARMS,DOMAINS,WINDOWS,WINDOW,ORACLE_NAME


def summarize(rows):
    if tuple(rows)!=ARMS:
        raise ValueError("all genuine and oracle arms must be separately present")
    aggregated={}
    for arm in ARMS:
        aggregated[arm]={}
        for domain in DOMAINS:
            source=rows[arm][domain]
            if len(source)!=WINDOWS or any(x["tokens_scored"]!=WINDOW-1 for x in source):
                raise ValueError("not complete precommitted 8x64 original source VALID")
            aggregated[arm][domain]=aggregate(source)
    effects={}
    for domain in DOMAINS:
        base=aggregated[ARMS[0]][domain]
        paid=aggregated[ARMS[1]][domain]
        oracle=aggregated[ORACLE_NAME][domain]
        effects[domain]={
          "paid_block0_minus_CB59_gold_NLL":
               paid["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
          "paid_block0_minus_CB59_teacher_KL":
               paid["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
          "UNPAID_FP32_block0_minus_CB59_gold_NLL":
               oracle["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
          "UNPAID_FP32_block0_minus_CB59_KL":
               oracle["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
          "paid_block0_teacher_argmax_agreement":
               paid["teacher_next_token_argmax_agreement_fraction"],
          "unpaid_FP32_oracle_does_NOT_qualify_0p7bpp":True,
        }
    gate=all(effects[d]["paid_block0_minus_CB59_gold_NLL"]<0
             and effects[d]["paid_block0_minus_CB59_teacher_KL"]<=0
             for d in DOMAINS)
    return {
      "all_original_source_NLL_KL_PPL":aggregated,
      "new_VALID_frozen_effects":effects,
      "physically_paid_block0_improves_true_NLL_and_teacher_KL_in_both_domains":gate,
      "only_original_CB59_and_paid_block0_are_physically_eligible":list(PHYSICAL_ARMS),
      "unpaid_FP32_oracle_never_qualifies_rate_or_speed":True,
      "even_a_PASS_does_not_mean_coherent_generation_or_QA":True,
    }
