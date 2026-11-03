"""Immutable evidence ingestion and reproducible error triage.
Automatically diagnoses; never infers a quality PASS from CI success.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path

TRAIN_SPLITS=("TRAIN","TRAIN_VALID")

def sha256(raw):return hashlib.sha256(raw).hexdigest()

def ingest(path:Path):
    raw=path.read_bytes()
    report=json.loads(raw)
    if report.get("experiment")!="CALIBRI_v91_frozen_original_sources_root_cause_decomposition":
        raise AssertionError("unexpected CALIBRI diagnostic artifact/version")
    if report.get("official_independent_TEST_never_loaded_or_tuned") is not True:
        raise AssertionError("official TEST provenance cannot enter automated selection")
    splits=report["public_train_dataset_sources"]
    if set(splits)!=set(TRAIN_SPLITS) or any(v["split"]!="train" for v in splits.values()):
        raise AssertionError("auto doctor requires disjoint public TRAIN-only data")
    first=[splits[k]["record_range"] for k in TRAIN_SPLITS]
    if not (first[0][1]<=first[1][0] or first[1][1]<=first[0][0]):
        raise AssertionError("TRAIN and TRAIN_VALID overlap")
    return report,{"report_SHA256":sha256(raw),"data_sources":splits,
                   "source_SHA256":report["physical_sources"]["CB59_SHA"],
                   "correction_SHA256":report["physical_sources"]["CB88_SHA"],
                   "original_independent_TEST_not_read":True}

def triage(report):
    head=report["four_frozen_head_ablations"]
    original=report["original_FP32_teacher_BoolQ_same_TRAIN_slices"]
    incidents=[]
    for split in TRAIN_SPLITS:
        summary=head[split]["summary"]
        teacher=original[split]
        t_correct=sum(row["correct"] for row in teacher)
        if t_correct<.75*len(teacher):
            incidents.append({"key":"teacher_ceiling:"+split,"severity":"high",
                "observed":t_correct/len(teacher),"hypothesis":"Small base teacher may be weak on this constrained QA prompt",
                "recommended_action":"validate_prompt_and_model_ceiling_on_TRAIN_only"})
        tensor=summary["component_effects"]["tensor_yes_minus_no"]
        bias=summary["component_effects"]["bias_yes_minus_no"]
        if abs(tensor)>1 and abs(tensor)>3*max(abs(bias),1e-6):
            incidents.append({"key":"tensor_logit_shift:"+split,"severity":"critical",
                "observed_tensor_yes_minus_no":tensor,"observed_bias_yes_minus_no":bias,
                "recommended_action":"full_output_operator_rewrite_with_MARGIN_guard",
                "diagnostic":"frozen tensor-only shifts Yes/No far more than bias-only"})
    greedy=report["four_frozen_greedy_diagnostic_modes"]
    if greedy["CB59"]["mean_distinct_ratio"]<0.25:
        incidents.append({"key":"upstream_greedy_collapse","severity":"critical",
            "baseline_distinct":greedy["CB59"]["mean_distinct_ratio"],
            "corrected_distinct":greedy["CB88"]["mean_distinct_ratio"],
            "recommended_action":"trace_quantized_attention_values_and_KV_cache"})
    causal=report["single_module_FP32_oracle_rescue_not_paid_source"]
    for x in causal["ranked_single_module_rescues"]:
        if x["mean_KL_gain_from_exact_FP32_one_module"]>=0.4 and x["module"].startswith("model.layers."):
            incidents.append({"key":"block_causal_candidate:"+x["module"],"severity":"high",
                "causal_teacher_KL_gain_2_TRAIN_probes":x["mean_KL_gain_from_exact_FP32_one_module"],
                "local_relative_RMS":x["local_same_input_relative_RMS"],
                "recommended_action":"inplace_whole_block_codebook_rewrite_with_zero_extra_storage",
                "caveat":"Two truncated TRAIN contexts: priority hypothesis, not confirmed production importance"})
    ranks={"critical":0,"high":1,"medium":2}
    incidents.sort(key=lambda item:(ranks[item["severity"]],item["key"]))
    return incidents
