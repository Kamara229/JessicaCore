"""Require true completed v165 top-K feasibility before hidden-state pilot."""
import json
from pathlib import Path
from .protocol import check,PAID_V161_BYTES


def check_v165(actual,physical):
    check()
    value=json.loads(Path(actual).read_text())
    if value.get("stage")!="v165_NO_TRAINING_GREEN_TUNNEL_TOPK_RESCUE_UPPER_BOUND":
        raise ValueError("missing actual complete Green Tunnel v165 topK diagnosis")
    if value.get("actual_frozen_CB161_sha256")!=physical.sha256:
        raise ValueError("v165 source does not match complete original physical CB161")
    if physical.size!=PAID_V161_BYTES or physical.layer!=0 or len(physical.patches)!=7:
        raise ValueError("original compressed source not physically complete")
    if not value.get("no_new_weights_or_external_tunnel_trained"):
        raise ValueError("v165 did not actually measure a no-training topK bound")
    prior=value["GREEN_TUNNEL_TOPK_oracle_upper_bounds_not_measured_Green_Tunnel"]
    paid=prior["fully_paid_layer0_CB161"]
    limits={}
    for domain in ("wiki_fit","news_fit"):
        upper=paid[domain]["topK_support_information_theoretic_reranker_ceiling_not_performance"]
        small=upper["128"]["teacher_TOP1_recovery_upper_bound_fraction"]
        big=upper["2048"]["teacher_TOP1_recovery_upper_bound_fraction"]
        if not (0<=small<.20 and small<=big<.70):
            raise ValueError("v165 observed topK limitation not supported by actual source")
        limits[domain]={"perfect_teacher_top128_recovery_ceiling":small,
                        "perfect_teacher_top2048_recovery_ceiling":big,
                        "even_teacher_oracle_within_topK_fails_for_remaining_tokens":True}
    return {
        "full_paid_CB161_SHA256":physical.sha256,
        "full_physical_bytes":physical.size,
        "actual_unpaid_v165_output_only_GT_topK_limits":limits,
        "reason_to_test_earlier_hidden_activation_points":True,
        "prior_model_generative_semantics_not_recovered":True,
        "no_new_TEST_OR_QA_or_green_model_claim":True,
    }
