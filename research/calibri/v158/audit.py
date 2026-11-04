"""Report true upstream-vs-weight error scales, not aggregate NLL proxy."""
from math import sqrt
import torch
from .protocol import FIT_DOMAINS,TOKENS_EACH


def _shape(x,y):
    if x.shape!=y.shape or x.ndim!=2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("paired model source vectors must be same shape finite")


def similarity(reference,observed):
    a=reference.to(torch.float64)
    b=observed.to(torch.float64)
    _shape(a,b)
    refnorm=float((a*a).sum())
    diffnorm=float(((a-b)**2).sum())
    cos=torch.nn.functional.cosine_similarity(a,b,dim=-1,eps=1e-12)
    return {
        "relative_RMSE":sqrt(diffnorm/max(refnorm,1e-18)),
        "absolute_RMSE_per_coordinate":sqrt(diffnorm/a.numel()),
        "mean_cosine":float(cos.mean()),
        "positive_cosine_fraction":float((cos>0).to(torch.float32).mean()),
        "student_rms_over_teacher_rms":sqrt(
            float((b*b).sum())/max(refnorm,1e-18)),
    }


def decompose(teacher_x,compressed_x,teacher_y,fp32_of_compressed,
              cb59_on_teacher,cb59_on_compressed,paid_on_teacher,
              paid_on_compressed):
    if teacher_x.shape!=compressed_x.shape or len(teacher_x)!=TOKENS_EACH:
        raise ValueError("incorrect fixed trained source FIT windows")
    if teacher_y.shape!=fp32_of_compressed.shape or teacher_y.shape!=cb59_on_teacher.shape:
        raise ValueError("incompatible operator output shapes")
    return {
       "upstream_layer6_vproj_input_drift":similarity(teacher_x,compressed_x),
       "upstream_input_ONLY_FP32_weight_response_drift":
           similarity(teacher_y,fp32_of_compressed),
       "original_CB59_weight_ONLY_error_at_FP32_teacher_input":
           similarity(teacher_y,cb59_on_teacher),
       "paid_1bit_weight_ONLY_error_at_FP32_teacher_input":
           similarity(teacher_y,paid_on_teacher),
       "original_CB59_weight_ONLY_error_at_native_student_input":
           similarity(fp32_of_compressed,cb59_on_compressed),
       "paid_1bit_weight_ONLY_error_at_native_student_input":
           similarity(fp32_of_compressed,paid_on_compressed),
       "native_CB59_COMBINED_input_and_weight_error_against_FP32":
           similarity(teacher_y,cb59_on_compressed),
       "paid_1bit_COMBINED_input_and_weight_error_against_FP32":
           similarity(teacher_y,paid_on_compressed),
       "whole_original_layer6_effect_also_depends_on_downstream_layers":True,
    }


def summary(per_domain):
    if tuple(per_domain)!=FIT_DOMAINS:
        raise ValueError("missing original source TRAIN FIT domain")
    result={}
    for domain in FIT_DOMAINS:
        item=per_domain[domain]
        def error(key):
            return item[key]["relative_RMSE"]
        teacher_weight5=error("original_CB59_weight_ONLY_error_at_FP32_teacher_input")
        teacher_weight6=error("paid_1bit_weight_ONLY_error_at_FP32_teacher_input")
        live_weight5=error("original_CB59_weight_ONLY_error_at_native_student_input")
        live_weight6=error("paid_1bit_weight_ONLY_error_at_native_student_input")
        end5=error("native_CB59_COMBINED_input_and_weight_error_against_FP32")
        end6=error("paid_1bit_COMBINED_input_and_weight_error_against_FP32")
        result[domain]={
            "paid_1bit_reduces_original_weight_error_on_teacher_inputs":teacher_weight6<teacher_weight5,
            "paid_1bit_reduces_original_weight_error_on_student_inputs":live_weight6<live_weight5,
            "paid_1bit_reduces_combined_actual_original_operator_error":end6<end5,
            "delta_combined_relative_RMSE_vs_native_CB59":end6-end5,
            "ratio_upstream_drift_vs_original_5bit_weight_only_error":
                error("upstream_input_ONLY_FP32_weight_response_drift")/
                max(live_weight5,1e-9),
            "upstream_drift_dominates_original_weight_only_error":
                error("upstream_input_ONLY_FP32_weight_response_drift")>live_weight5,
        }
    return {
        "original_TRAlN_FIT_only_domain_findings":result,
        "not_a_new_source_validation_or_sciq_boolq_semantic_result":True,
        "no_additional_model_bits_learned_or_runtime_changed":True,
    }
