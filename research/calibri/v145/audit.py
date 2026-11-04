"""Predeclared untouched data confirmation; never optimize from these results."""
from v142.scoring import aggregate
from .protocol import ARMS,SOURCE_DOMAINS


def assemble(evidence):
    if tuple(evidence)!=ARMS:
        raise ValueError("missing one of three locked preselected conditions")
    result={}
    for domain in SOURCE_DOMAINS:
        if any(len(evidence[arm][domain])!=8 for arm in ARMS):
            raise ValueError("not eight predeclared 64-token windows per arm")
        base=aggregate(evidence[ARMS[0]][domain])
        chosen=aggregate(evidence[ARMS[1]][domain])
        paid=aggregate(evidence[ARMS[2]][domain])
        if not base["tokens_scored"]==chosen["tokens_scored"]==paid["tokens_scored"]==504:
            raise ValueError("different aligned next token counts")
        result[domain]={
            "original_CB59":base,
            "v144_selected_0_input_0_5_head":chosen,
            "v140_paid_original_input_and_head":paid,
            "v144_selected_minus_CB59_NLL":
              chosen["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
            "v144_selected_minus_CB59_teacher_KL":
              chosen["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
            "v140_full_minus_CB59_NLL":
              paid["mean_TRUE_next_token_NLL"]-base["mean_TRUE_next_token_NLL"],
            "v140_full_minus_CB59_teacher_KL":
              paid["teacher_to_candidate_KL"]-base["teacher_to_candidate_KL"],
        }
    selected_both=all(
        result[d]["v144_selected_minus_CB59_NLL"]<0 and
        result[d]["v144_selected_minus_CB59_teacher_KL"]<=0
        for d in SOURCE_DOMAINS)
    full_both=all(
        result[d]["v140_full_minus_CB59_NLL"]<0 and
        result[d]["v140_full_minus_CB59_teacher_KL"]<=0
        for d in SOURCE_DOMAINS)
    return {
        "new_train_split_text_heldout":result,
        "v144_preselected_passes_both_shifted_domains":selected_both,
        "full_v140_passes_both_shifted_domains":full_both,
        "no_any_gain_reselection_or_training":True,
        "semantic_response_ability_is_NOT_inferred_from_NLL":True,
        "even_pass_is_NOT_independent_v89_TEST":True,
    }
