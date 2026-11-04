"""Apply v106 preregistered morphological alarm UNCHANGED on v103/v104.

Manual *source-specific* labels: 6 original pretrained FP32 generations
are syntactically fluent (though can be factually wrong/role contaminated);
12 packed-native generated replies are obvious repeated nonsense. These
labels come from historical visible outputs, not independent blinded expert
annotations. Do not confuse fluency detection with correctness.
"""
from __future__ import annotations
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path
from v106.signals import generation_signals
from .contract import SOURCE_FILES

def verify(raw,name):
    spec=SOURCE_FILES[name]
    digest=sha256(raw).hexdigest()
    if digest!=spec["sha256"]:
        raise AssertionError("historical generated-text artifact hash mismatch: "+name)
    data=json.loads(raw)
    if data.get("independent_official_v89_TEST_unopened") is not True and name=="v103":
        raise AssertionError("v103 did not certify its original locked TEST")
    if data.get("no_official_independent_TEST_opened") is not True and name=="v104":
        raise AssertionError("v104 did not certify its original locked TEST")
    if name=="v103":
        cases=data["genuine_original_model_chat_template_texts_and_labels"]
        if len(cases)!=spec["expected_prompts"]:
            raise AssertionError("unexpected original 4 v103 contexts")
        rows=[]
        for prompt in cases:
            model_outputs=prompt["texts"]
            for model,result in model_outputs.items():
                rows.append({"run":name,"domain":prompt["domain"],
                    "prompt_SHA256":prompt["chat_template_prompt_sha256"],
                    "model":model,"text":result["actual_decoded_text"],
                    "tokens":result["raw_generation_ids"],
                    "expected_label":prompt["gold_original_answer"]})
    else:
        cases=data["real_CPU_KV_cache_32_token_original_chat_template"]
        if len(cases)!=spec["expected_prompts"]:
            raise AssertionError("unexpected original 2 v104 contexts")
        rows=[]
        for prompt in cases:
            model_outputs=prompt["original_chat_template_KV_cached_full_model_texts"]
            for model,result in model_outputs.items():
                rows.append({"run":name,"domain":prompt["domain"],
                    "prompt_SHA256":prompt["chat_prompt_sha256"],
                    "model":model,"text":result["actual_text"],
                    "tokens":result["generated_ids"],
                    "expected_label":prompt["gold"]})
    counts=Counter(x["model"] for x in rows)
    if (set(counts)!={spec["original_FP32"],*spec["compressed_models"]}
         or any(v!=spec["expected_prompts"] for v in counts.values())):
        raise AssertionError("dropped pretrained/candidate comparison response")
    for x in rows:
        if len(x["tokens"])!=32 or not isinstance(x["text"],str):
            raise AssertionError("historical original fixed 32-token response altered")
        x["historical_human_visible_gibberish"]=x["model"]!=spec["original_FP32"]
        x["historical_label_provenance"]="source-specific posthoc visual reading, NOT blinded"
        x["diagnostic"]=generation_signals(x["text"],x["tokens"])
        x["alarm"]=x["diagnostic"]["exploratory_fivegram_AND_zlib_anomaly_alarm"]
    return rows

def run(v103:Path,v104:Path,out:Path):
    all_rows=verify(v103.read_bytes(),"v103")+verify(v104.read_bytes(),"v104")
    confusion=Counter()
    by_model=defaultdict(lambda:Counter())
    for x in all_rows:
        label=x["historical_human_visible_gibberish"]
        alarm=x["alarm"]
        key="TP" if alarm and label else ("FP" if alarm else ("FN" if label else "TN"))
        confusion[key]+=1
        by_model[x["model"]][key]+=1
    if (len(all_rows)!=18 or
        sum(x["historical_human_visible_gibberish"] for x in all_rows)!=12 or
        sum(not x["historical_human_visible_gibberish"] for x in all_rows)!=6):
        raise AssertionError("unbalanced or incomplete cross-run source")
    if len({(r["run"],r["model"],r["prompt_SHA256"]) for r in all_rows})!=len(all_rows):
        raise AssertionError("duplicate original generation outputs")
    tp,tn,fp,fn=(confusion[k] for k in ("TP","TN","FP","FN"))
    result={
        "experiment":"CALIBRI_v107_fixed_v106_text_detector_cross_run_observational_evaluation",
        "original_v96_detector_fitting_or_posthoc_review_run":41739479888,
        "independent_historical_source_output_runs":[41747347901,41748235592],
        "source_file_SHA256_pins":{k:v["sha256"] for k,v in SOURCE_FILES.items()},
        "existing_v106_thresholds_NOT_modified_or_tuned_from_v103_v104":True,
        "source_results_count":len(all_rows),
        "reviewed_visible_text_repeated_gibberish":12,
        "reviewed_visible_text_fluent_BUT_sometimes_wrong_FP32":6,
        "confusion_vs_source_specific_visual_fluency_label":dict(confusion),
        "sensitivity_on_this_specific_small_12_output_observational_sample":
             tp/max(tp+fn,1),
        "specificity_on_this_specific_small_6_output_observational_sample":
             tn/max(tn+fp,1),
        "per_model_confusion":{k:dict(v) for k,v in sorted(by_model.items())},
        "all_paired_original_answers":all_rows,
        "language_fluency_NOT_mathematical_or_factual_accuracy":True,
        "not_independent_blinded_annotation_nor_prospective_VALID":True,
        "separate_validated_coherence_detector_not_yet_achieved":True,
        "false_pos_or_neg_on_small_observational_sample_NOT_universal_rate":True,
        "generation_not_rerun_weights_not_modified_or_loaded":True,
        "official_independent_v89_TEST_unopened":True,
        "no_Live_Jessica_main_Render_Supabase_APK_changes":True}
    out.mkdir(parents=True,exist_ok=True)
    p=out/"v107_frozen_v103_v104_cross_run_anomaly_out_of_v96.json"
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v107_cross_run_evaluation_done",
       "visible_source_label_vs_fixed_detector":{"TP":tp,"FP":fp,"TN":tn,"FN":fn},
       "source_specific_not_independently_blinded":True,
       "original_TEST_untouched":True},ensure_ascii=False),flush=True)
    return result
