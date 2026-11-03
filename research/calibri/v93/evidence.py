"""Read only directly observable behavior from TRAIN-only v92 outcome evidence.

No full-vocabulary KL, teacher/student distribution, NLL or FP32 similarity.
Model output quality is judged by tasks and readable generation, not mimicry.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path

EXPERIMENT="CALIBRI_v92_automated_detect_triage_rewrite_validate_RESEARCH_ONLY"

def read(path:Path):
    raw=path.read_bytes()
    data=json.loads(raw)
    if data.get("experiment")!=EXPERIMENT or data.get("original_TEST_not_used_for_tuning") is not True:
        raise AssertionError("wrong evidence or frozen independent TEST provenance")
    sources=data["public_TRAIN_sources"]
    if set(sources)!=set(("TRAIN","TRAIN_VALID")):
        raise AssertionError("missing original train/valid provenance")
    for value in sources.values():
        if value.get("dataset")!="google/boolq" or value.get("split")!="train":
            raise AssertionError("not permitted: non-TRAIN split")
    a,b=(sources[k]["record_range"] for k in ("TRAIN","TRAIN_VALID"))
    if not (a[1]<=b[0] or b[1]<=a[0]):
        raise AssertionError("overlapping independent train/valid")
    physical=data["physical_candidate"]
    if physical.get("native_ABI_unchanged") is not True or physical.get("all_other_modules_byte_unchanged") is not True:
        raise AssertionError("unverified whole-block native rewrite")
    return data,{"evidence_SHA256":hashlib.sha256(raw).hexdigest(),
                "frozen_TEST_not_accessed":True,
                "data_split":"two disjoint subsets of original BoolQ train"}

def extract(data):
    paired=data["paired_new_train_VALID_quality"]
    orig=paired["original"];new=paired["candidate"]
    n=orig["n"]
    if n!=new["n"] or n!=len(orig["items"]) or n!=len(new["items"]) or n<=0:
        raise AssertionError("invalid paired real answer counts")
    clean={}
    for label,rows in (("original",orig["items"]),("candidate",new["items"])):
        clean[label]=[]
        for r in rows:
            if r["gold"] not in (0,1) or r["pred"] not in (0,1):
                raise AssertionError("unrecognized original ground-truth answer")
            if r["correct"]!=int(r["gold"]==r["pred"]):
                raise AssertionError("inconsistent externally scored correctness")
            clean[label].append({k:r[k] for k in ("record_index","prompt_SHA","gold","pred","correct")})
    if [(x["record_index"],x["prompt_SHA"],x["gold"]) for x in clean["original"]] != [
        (x["record_index"],x["prompt_SHA"],x["gold"]) for x in clean["candidate"]]:
        raise AssertionError("different input prompts or labels")
    if sum(x["correct"] for x in clean["original"])!=orig["correct"] or sum(x["correct"] for x in clean["candidate"])!=new["correct"]:
        raise AssertionError("reported correctness differs from paired answers")
    greedy=data["greedy_on_same_fresh_diagnostic_prompts"]
    baseline=greedy["CB59"];candidate=greedy["rewritten"]
    for v in (baseline,candidate):
        if v.get("genuine_KV_cache") is not True or not v.get("prompts"):
            raise AssertionError("not real comparable greedy decoding")
        if any(p["distinct_ratio"]<0 or p["distinct_ratio"]>1 for p in v["prompts"]):
            raise AssertionError("invalid token diversity")
    if [p["diagnostic_prompt_SHA"] for p in baseline["prompts"]] != [
        p["diagnostic_prompt_SHA"] for p in candidate["prompts"]]:
        raise AssertionError("unequal greedy prompts")
    return {"answer_rows":clean,
            "generation":{name:{key:float(greedy[src][key]) for key in
                ("mean_distinct_ratio","mean_most_frequent_share")}
                for name,src in (("baseline","CB59"),("candidate","rewritten"))},
            "generation_prompts":len(baseline["prompts"]),
            "archive_bytes":data["physical_candidate"]["same_complete_source_length"],
            "native_block_speed_ratio":float(data["paired_native_block_microbenchmark"]["candidate_speed_ratio"]),
            "native_speed_is_full_model":not data["paired_native_block_microbenchmark"]["microbench_not_full_model_speed"],
            "new_source_SHA":data["physical_candidate"]["candidate_SHA256"]}
