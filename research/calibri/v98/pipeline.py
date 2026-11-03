"""Frozen v98 read-only layer-residual and logit-entropy diagnosis.

Diagnostic on entirely new synthetic prefix prompts; NOT TEST, tuning,
candidate creation, model training or original-weights repair.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v94.generation import generate_one
from v96.grade import string_breakdown
from .hooks import compare_same_context
from .spec import PROBES,MAX_CONTEXT_TOKENS

def run(source_path:Path,lock_path:Path,out:Path):
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    out.mkdir(parents=True,exist_ok=True)
    locked=json.loads(lock_path.read_text())
    if locked.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04":
        raise AssertionError("original physical references are not frozen")
    raw=source_path.read_bytes()
    baseline=locked["frozen_reference_sources"]["CB59"]
    sha=hashlib.sha256(raw).hexdigest()
    bpw=8*len(raw)/EXPECTED_UNIQUE
    if (len(raw)!=baseline["complete_archive_bytes"] or sha!=baseline["sha256"]
        or bpw>0.700 or MODEL_SHA!=locked["original_model_revision"]):
        raise AssertionError("frozen source size, SHA, or model revision changed")
    model,info,_=make_model("baseline",SimpleNamespace(original=source_path,folder=out))
    if not info.get("zero_dense_pretrained_2D_hot_weights"):
        raise AssertionError("not true native compact-source forward")
    teacher=original_teacher()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    records=[]
    for domain,prompt in PROBES:
        encoded=tok.encode(prompt,add_special_tokens=False)
        if not (1<len(encoded)<=MAX_CONTEXT_TOKENS):
            raise AssertionError("missing, empty or unexpectedly long probe prefix")
        ids=torch.tensor(encoded,dtype=torch.long)[None,:]
        trace=compare_same_context(teacher,model,ids,tok)
        reference=generate_one(teacher,tok,prompt)
        packed=generate_one(model,tok,prompt)
        record={
            "domain":domain,
            "original_prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            "original_token_prefix_sha256":hashlib.sha256(
                ids.detach().numpy().astype("<i8").tobytes()).hexdigest(),
            "token_prefix_length":len(encoded),
            "same_input_activation_and_logit_trace":trace,
            "true_autoregressive_FP32":{
                "text":reference["text"],
                "raw_generated_ids":reference["ids"],
                "lexical":string_breakdown(reference["text"],reference["ids"])},
            "true_autoregressive_CB59":{
                "text":packed["text"],
                "raw_generated_ids":packed["ids"],
                "lexical":string_breakdown(packed["text"],packed["ids"])},
        }
        records.append(record)
        worst=trace["stage_drift_summary"]["top_positive_drift_jumps"][:3]
        print(json.dumps({
            "stage":"v98_same_prefix_drift_and_actual_generation",
            "domain":domain,
            "teacher_top1_probability":trace["teacher_next_token"]["top1_probability"],
            "native_top1_probability":trace["native_next_token"]["top1_probability"],
            "teacher_entropy":trace["teacher_next_token"]["entropy_nats"],
            "native_entropy":trace["native_next_token"]["entropy_nats"],
            "top10_overlap":trace["top10_vocab_overlap_count"],
            "worst_positive_jump_stages":worst,
            "teacher_repetition_alarm":record["true_autoregressive_FP32"]["lexical"]["repeat_over_75pct"],
            "native_repetition_alarm":record["true_autoregressive_CB59"]["lexical"]["repeat_over_75pct"]},ensure_ascii=False),flush=True)
    if not records or any(len(r["same_input_activation_and_logit_trace"]["stages"])<30 for r in records):
        raise AssertionError("did not record all frozen transformer layers")
    stages=[x["stage"] for x in records[0]["same_input_activation_and_logit_trace"]["stages"]]
    if any([x["stage"] for x in r["same_input_activation_and_logit_trace"]["stages"]]!=stages for r in records):
        raise AssertionError("some diagnostic prompts missed a transformer layer")
    ranking=[]
    for stage in stages:
        rows=[next(x for x in r["same_input_activation_and_logit_trace"]["stages"]
            if x["stage"]==stage) for r in records]
        ranking.append({
            "stage":stage,
            "mean_relative_RMS_drift":sum(x["metrics"]["relative_rms_error"] for x in rows)/len(rows),
            "mean_teacher_student_cosine":sum(x["metrics"]["cosine_similarity"] for x in rows)/len(rows),
            "mean_incremental_drift_vs_previous_stage":
              None if rows[0]["delta_from_previous_stage"] is None else
              sum(x["delta_from_previous_stage"] for x in rows)/len(rows)})
    positive=sorted((x for x in ranking if x["mean_incremental_drift_vs_previous_stage"] is not None),
                 key=lambda x:x["mean_incremental_drift_vs_previous_stage"],reverse=True)
    result={
        "experiment":"CALIBRI_v98_original_CB59_frozen_layer_residual_drift_and_logit_collapse",
        "pretrained_revision":MODEL_SHA,
        "physical_unchanged_CB59_SHA256":sha,
        "physical_unchanged_archive_bytes":len(raw),
        "physical_unchanged_all_in_bpp":bpw,
        "contexts":"four new synthetic diagnostic prompts, not a labeled benchmark or official TEST",
        "teacher_forced_same_prefix_traces_vs_true_free_generation":records,
        "mean_across_prompts_by_residual_stage":ranking,
        "largest_positive_mean_drift_increments":positive[:8],
        "diagnostic_only_observational_not_proof_of_block_causality":True,
        "avoid_confusing_entropy_and_benchmark_semantic_quality":True,
        "did_not_train_select_change_quantization_or_any_weights":True,
        "previous_official_v89_TEST_never_opened":True,
        "live_Jessica_main_Render_Supabase_APK_unchanged":True,
        "limitations":"Layer residual drift uses forced same input prompts, not native generated "
          "sequence. It separates drift from self-reinforcement but does not identify "
          "causal defects alone; lexical repetition metrics are not semantic quality. "
          "Any future repair must be physically paid and match native CPU speed, "
          "correctness and coherent real text."}
    dest=out/"v98_frozen_layer_residual_and_logits_forced_vs_generated.json"
    dest.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v98_completed_frozen_full_layer_localization",
        "top_mean_positive_rms_jump_stages":positive[:5],
        "archive_bytes":len(raw),
        "previous_TEST_untouched":True,
        "new_physical_weights_or_bytes":0},ensure_ascii=False),flush=True)
    return result
