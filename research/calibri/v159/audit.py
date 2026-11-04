"""Measure upstream drift by depth, not only final supervised output.

The per-block change in teacher-relative RMSE is CORRELATIVE, not proof of
causal error assigned to that block; upstream errors propagate through all
subsequent pretrained operators. Distinguish input codebook corruption,
early block drift, layer6 intervention, and the final normalized state.
"""
from v158.audit import similarity
from .protocol import TRACE_KEYS,FIT_DOMAINS,MODEL_ARMS,LAYERS


def by_layer(teacher,student):
    if tuple(teacher)!=TRACE_KEYS or tuple(student)!=TRACE_KEYS:
        raise ValueError("30 original transformer block trace points required")
    result={}
    previous=None
    for depth,key in enumerate(TRACE_KEYS):
        quality=similarity(teacher[key],student[key])
        if quality["relative_RMSE"]<0:
            raise ValueError("invalid teacher source relative error")
        quality["position"]=depth
        quality["delta_relative_RMSE_vs_preceding_trace"]=(
            quality["relative_RMSE"]-previous if previous is not None else None)
        result[key]=quality
        previous=quality["relative_RMSE"]
    return result


def compare_domain(teacher,baseline,paid):
    ordinary=by_layer(teacher,baseline)
    changed=by_layer(teacher,paid)
    diffs={}
    for stage in TRACE_KEYS:
        diffs[stage]=(
            changed[stage]["relative_RMSE"]-ordinary[stage]["relative_RMSE"])
    base_emb=ordinary["input_embedding"]["relative_RMSE"]
    base_block6=ordinary["block_06_output"]["relative_RMSE"]
    base_final=ordinary["final_norm"]["relative_RMSE"]
    paid_block6=changed["block_06_output"]["relative_RMSE"]
    paid_final=changed["final_norm"]["relative_RMSE"]
    layer_growth=[(
        i,
        ordinary[f"block_{i:02d}_output"]["delta_relative_RMSE_vs_preceding_trace"])
        for i in range(LAYERS)]
    largest=sorted(layer_growth,key=lambda pair:-pair[1])
    return {
        "full_CB59_teacher_vs_packed_drift_by_source_stage":ordinary,
        "full_CB153_paid_vproj_teacher_vs_packed_drift_by_source_stage":changed,
        "paid_v_proj_minus_native_CB59_teacher_RMSE_by_stage":diffs,
        "input_tied_embedding_original_CB59_RMSE":base_emb,
        "original_block6_post_output_RMSE":base_block6,
        "paid_vproj_block6_post_output_RMSE":paid_block6,
        "native_CB59_final_norm_RMSE":base_final,
        "paid_vproj_final_norm_RMSE":paid_final,
        "paid_vproj_changes_drift_at_block6":diffs["block_06_output"],
        "paid_vproj_changes_drift_after_final_norm":diffs["final_norm"],
        "largest_FIVE_teacher_normalized_rRMSE_increments_correlative_ONLY":[
            {"block":int(k),"incremental_rRMSE":float(d)} for k,d in largest[:5]],
        "first_stage_above_teacher_relative_RMSE_thresholds":{
            str(level):next((key for key in TRACE_KEYS
                             if ordinary[key]["relative_RMSE"]>=level),None)
            for level in (.5,1.,1.5,2.)},
        "cannot_attribute_cause_to_one_layer_from_post_activation_RMSE_alone":True,
        "no_language_quality_or_release_success_from_this_probe":True,
    }


def summarize(domains):
    if tuple(domains)!=FIT_DOMAINS:
        raise ValueError("must measure both original TRAIN FIT source domains")
    return {
        "existing_original_TRAIN_FIT_two_domain_activation_trace":domains,
        "common_heaviest_5_RMSE_increments":{
            domain:results["largest_FIVE_teacher_normalized_rRMSE_increments_correlative_ONLY"]
            for domain,results in domains.items()},
        "this_is_diagnostic_not_new_TEST_or_source_VALID":True,
        "original_F32_teacher_oracle_not_physical_0p7bpp_candidate":True,
        "no_parameters_retrained_or_refitted":True,
    }
