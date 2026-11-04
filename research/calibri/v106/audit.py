"""Re-score ONLY a pinned historical v96 artifact; no live model or TEST.

Report whether two complementary cheap morphology indicators flag the
obvious repeated-subword collapse that v96's >75% single-token alarm
misses, without claiming the new threshold generalizes to new models.
"""
from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path
from v96.answer_review import first_response
from .signals import generation_signals

SOURCE_SHA="917c786af2bb54c8ae1f9b88029230c54e4761373fe791f99263939da4ac2ac5"
SOURCE_RUN=41739479888
NAMES=("CB59","v92","original_FP32")

def audit(raw:bytes):
    digest=sha256(raw).hexdigest()
    if digest!=SOURCE_SHA:
        raise AssertionError("unexpected original frozen v96 generated-output SHA")
    source=json.loads(raw)
    if (source.get("experiment")!="CALIBRI_v96_original_answer_and_semantic_content_canary_audit"
        or source.get("previous_official_TEST_never_loaded") is not True):
        raise AssertionError("v96 report provenance broken")
    models=source["full_original_models_genuine_greedy_decoding"]
    if set(models)!=set(NAMES):
        raise AssertionError("original three paired pretrained models absent")
    full={}
    summary={}
    for name in NAMES:
        data=models[name]
        records=data["questions"]+data["open_ended"]
        if len(data["questions"])!=8 or len(data["open_ended"])!=4 or len(records)!=12:
            raise AssertionError("original v96 12 identical tasks missing")
        scores=[]
        for entry in records:
            ids=entry["generated_token_ids"]
            if len(ids)!=32:
                raise AssertionError("original generated sequence truncated")
            info=generation_signals(entry["actual_generated_text"],ids)
            scores.append({
                "original_prompt_SHA":entry["prompt_SHA"],
                "task_type":entry["category"],
                "first_visible_response_excerpt":first_response(entry["actual_generated_text"])[:160],
                "old_v96_single_token_repetition_alarm":bool(
                    entry["lexical_quality_observations"]["repeat_over_75pct"]),
                "signals":info,
            })
        full[name]=scores
        summary[name]={
            "original_generation_count":len(scores),
            "original_old_single_token_alarm_count":sum(
               int(x["old_v96_single_token_repetition_alarm"]) for x in scores),
            "new_exploratory_two_signal_alarm_count":sum(
               int(x["signals"]["exploratory_fivegram_AND_zlib_anomaly_alarm"])
               for x in scores),
            "fake_role_line_continuations_count":sum(
               int(x["signals"]["simulated_role_line_present"]) for x in scores),
            "mean_fivegram_duplicated_coverage":sum(
               x["signals"]["five_character_ngram_duplicated_window_fraction"]
               for x in scores)/len(scores),
            "original_answer_correctness_not_inferred_from_this_alarm":True
        }
    paired=[tuple(r["original_prompt_SHA"] for r in full[n]) for n in NAMES]
    if len(set(paired))!=1:
        raise AssertionError("not identical original v96 prompts")
    if (summary["CB59"]["original_old_single_token_alarm_count"]!=1
        or summary["v92"]["original_old_single_token_alarm_count"]!=1):
        raise AssertionError("historical baseline alarm counts unexpectedly changed")
    return {
      "experiment":"CALIBRI_v106_v96_locked_response_anomaly_signal_regression",
      "original_source_file_SHA256":digest,
      "original_v96_finished_Actions_run":SOURCE_RUN,
      "summaries":summary,
      "complete_original_response_feature_records":full,
      "v96_original_FP32_0_of_8_strict_prefix_grader_false_negative":True,
      "original_v96_factual_first_response_interpretive_review_FP32_4_of_8":True,
      "original_native_CB59_and_v92_factual_correct_0_of_8":True,
      "thresholds_historically_inspected_posthoc_NOT_training_or_unseen_quality":True,
      "new_detector_must_be_independently_validated_before_Jessica_autonomy":True,
      "no_model_weights_or_candidates_loaded_or_selected":True,
      "original_v89_independent_TEST_unopened":True,
      "Jessica_live_main_Render_Supabase_APK_unmodified":True,
      "scientific_status":"original compressed model generation quality FAIL"
    }

def run(source:Path,out:Path):
    result=audit(source.read_bytes())
    out.mkdir(parents=True,exist_ok=True)
    file=out/"v106_v96_original_generation_repetition_detector_regression.json"
    file.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"event":"v106_frozen_v96_generation_signal_regression_completed",
      "old_vs_new":{n:(x["original_old_single_token_alarm_count"],
                        x["new_exploratory_two_signal_alarm_count"])
                 for n,x in result["summaries"].items()},
      "no_new_TEST_or_training":True},ensure_ascii=False),flush=True)
    return result
