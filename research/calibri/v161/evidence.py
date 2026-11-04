"""Require v159 upstream-cause clue and actual NEGATIVE v160 input source.

No test samples accessed here; only prior already completed result artifacts.
"""
import json
from pathlib import Path
from .protocol import CB153_SHA,ORIGINAL_V140_SHA,check


def verify_old(v159,v160):
    check()
    a=json.loads(Path(v159).read_text())
    b=json.loads(Path(v160).read_text())
    if a.get("stage")!="v159_ORIGINAL_30BLOCK_REAL_ACTIVATION_DRIFT_MAP":
        raise ValueError("not real original v159 activation trace")
    if b.get("stage")!="v160_ORIGINAL_PAID_TIED_EMBEDDING_VS_OUTPUT_CAUSAL_VALID":
        raise ValueError("not actual completed tied embedding/head v160")
    if a.get("full_physical_original_source_CB153_SHA256")!=CB153_SHA:
        raise ValueError("v159 pinned original source changed")
    if b.get("original_frozen_complete_CB140_source_SHA256")!=ORIGINAL_V140_SHA:
        raise ValueError("v160 fully paid source changed")
    domains=a["teacher_relative_RMSE_COSINE_and_stage_delta"][
        "existing_original_TRAIN_FIT_two_domain_activation_trace"]
    earlier=b["causal_input_vs_head_quality"][
        "paired_per_domain_causal_input_head_effects"]
    if b["causal_input_vs_head_quality"][
        "paid_input_ONLY_improves_NLL_and_KL_on_both_new_domains"]:
        raise ValueError("v160 input paid correction did not fail as expected")
    for domain in ("wiki_fit","news_fit"):
        item=domains[domain]
        inc=item["full_CB59_teacher_vs_packed_drift_by_source_stage"][
            "block_00_output"]["delta_relative_RMSE_vs_preceding_trace"]
        if not inc>0.2:
            raise ValueError("first block not actually a strong upstream error site")
    for domain in ("wiki_fresh_embedding_causal","news_fresh_embedding_causal"):
        comp=earlier[domain]["paid_input_only_10"]
        if not (comp["vs_CB59_TRUE_next_token_NLL"]>0 and
                comp["vs_CB59_FP32_teacher_KL"]>0):
            raise ValueError("actual original embedding repair result altered")
    return {
      "v159_first_block_activation_drift_in_two_original_FIT_domains":True,
      "v160_physical_input_only_correction_failed_true_NLL_and_KL_both_valid_domains":True,
      "v160_stronger_head_ONLY_not_semantic_solution":True,
      "block0_oracle_unpaid_FP32_must_not_count_as_packed":True,
      "prior_negative_results_not_hidden":True,
    }
