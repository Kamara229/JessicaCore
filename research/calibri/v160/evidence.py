"""Require real completed v159 layer0/embedding drift before testing remedy."""
import json
from pathlib import Path
from .protocol import CB153_SHA,FIT_DOMAINS,check


def verify_v159(path):
    check()
    result=json.loads(Path(path).read_text())
    if result.get("stage")!="v159_ORIGINAL_30BLOCK_REAL_ACTIVATION_DRIFT_MAP":
        raise ValueError("not actual v159 original activation trace")
    if result.get("full_physical_original_source_CB153_SHA256")!=CB153_SHA:
        raise ValueError("source of previously measured drift changed")
    domains=result["teacher_relative_RMSE_COSINE_and_stage_delta"][
        "existing_original_TRAIN_FIT_two_domain_activation_trace"]
    if tuple(domains)!=FIT_DOMAINS:
        raise ValueError("missing v159 original two old TRAIN FIT domains")
    values={}
    for key,entry in domains.items():
        emb=entry["input_tied_embedding_original_CB59_RMSE"]
        block0=entry["full_CB59_teacher_vs_packed_drift_by_source_stage"][
            "block_00_output"]["delta_relative_RMSE_vs_preceding_trace"]
        if not (.6<emb<.9 and block0>0.1):
            raise ValueError("actual v159 emb/block0 original finding differs")
        values[key]={"original_embedding_rRMSE":emb,
                     "block0_additional_correlation":block0}
    return {
        "v159_original_CB59_embedding_and_first_layer_drift":values,
        "prior_actual_v159_only_OLD_TRAIN_FIT":True,
        "new_v160_validation_unseen_when_gains_locked":True,
        "not_a_rationale_to_omit_negative_0p7bit_model_performance":True,
    }
