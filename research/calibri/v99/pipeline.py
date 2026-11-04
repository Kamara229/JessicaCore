"""v99 diagnostic causal rescue of accumulated original CB59 hidden-state drift.

Exact FP32 activations are supplied at one decoder boundary, while all
pretrained native CB59 trained matrices remain frozen and unchanged.
A gain here DOES NOT produce or validate a compressed LLM: the oracle
requires expensive FULL ORIGINAL FP32 inference on every input prefix.
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
from v99.spec import PROBES,STAGES,GENERATE_ON_CATEGORIES
from v99.patch import paired_one_token_logits,native_logits
from v99.behavior import compare_logits,generate

def run(original:Path,locked:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    lock=json.loads(locked.read_text())
    frozen=lock["frozen_reference_sources"]["CB59"]
    raw=original.read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    bpw=8*len(raw)/EXPECTED_UNIQUE
    if (lock.get("contract_version")!="CALIBRI_BASELINE_LOCK_2026-11-04"
        or sha!=frozen["sha256"]
        or len(raw)!=frozen["complete_archive_bytes"]
        or MODEL_SHA!=lock["original_model_revision"]
        or bpw>.700):
        raise AssertionError("original native physical source is not locked")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    compressed,info,_=make_model("baseline",SimpleNamespace(original=original,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("actual native packed inference required")
    teacher=original_teacher()
    observations=[]
    with torch.inference_mode():
        for category,prompt in PROBES:
            ids=torch.tensor(tok.encode(prompt,add_special_tokens=False),dtype=torch.long)[None,:]
            if not (1<ids.numel()<=96):
                raise AssertionError("invalid same-token prefix")
            ref=native_logits(teacher,ids)
            base=native_logits(compressed,ids)
            reference=compare_logits(ref,base,tok)
            variants={}
            for stage in STAGES:
                exact,student=paired_one_token_logits(teacher,compressed,ids,stage)
                if not bool(torch.allclose(exact,ref,atol=1e-5,rtol=1e-5)):
                    raise AssertionError("FP32 teacher changed between comparable probes")
                variants[stage]={"full_vocab_same_prefix":compare_logits(ref,student,tok)}
                if category in GENERATE_ON_CATEGORIES:
                    variants[stage]["real_12_token_greedy_with_unpaid_repeated_FP32_oracle"]=generate(
                        compressed,tok,prompt,stage=stage,teacher=teacher)
                print(json.dumps({
                    "stage":"v99_diagnostic_FP32_activation_one_stage_only",
                    "probe":category,"oracle_stage":stage,
                    "frozen_CB59_top10_overlap":reference["reference_top10_overlap_count"],
                    "patched_CB59_top10_overlap":variants[stage]["full_vocab_same_prefix"]["reference_top10_overlap_count"],
                    "next_token_matches_original_FP32":variants[stage]["full_vocab_same_prefix"]["FP32_top1_matched"],
                    "physical_source_bytes_unchanged":len(raw)},ensure_ascii=False),flush=True)
            original_generated={}
            if category in GENERATE_ON_CATEGORIES:
                original_generated={
                    "original_FP32":generate(teacher,tok,prompt),
                    "frozen_native_CB59":generate(compressed,tok,prompt)}
            observations.append({"prompt_domain":category,
                "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
                "same_input_token_length":int(ids.numel()),
                "original_vs_CB59_next_token_reference":reference,
                "activation_only_oracles":variants,
                "actual_greedy_unassisted_baselines":original_generated})
    if len(observations)!=len(PROBES):raise AssertionError("missing paired probe")
    ranking={}
    for stage in STAGES:
        records=[r["activation_only_oracles"][stage]["full_vocab_same_prefix"] for r in observations]
        ranking[stage]={
            "mean_top10_token_overlap_with_original_FP32":
                sum(x["reference_top10_overlap_count"] for x in records)/len(records),
            "FP32_top1_matches_out_of_four":
                sum(int(x["FP32_top1_matched"]) for x in records),
            "coherent_natural_text_manually_reviewed":False,
            "requires_unpaid_FP32_teacher_on_EVERY_autoregressive_step":True,
            "physically_deployable":False}
    output={"experiment":"CALIBRI_v99_original_FP32_temporarily_injected_layer_boundary_causal_rescue",
        "physical_CB59_source_sha256_unchanged":sha,
        "full_original_packed_source_bytes":len(raw),
        "all_in_bpp":bpw,
        "original_pretrained_revision":MODEL_SHA,
        "new_synthetic_contexts_no_official_TEST":observations,
        "observer_top10_ranking_is_not_semantic_quality_or_model_selection":ranking,
        "full_original_FP32_activations_unpaid_and_needed_per_decoding_step":True,
        "new_compressed_weights_created_or_optimized":False,
        "unchanged_native_model_source_and_ALL_quantized_modules":True,
        "old_independent_TEST_not_opened":True,
        "live_Jessica_main_Render_Supabase_APK_untouched":True,
        "interpretation":"Single upstream boundary patch is a causal upper-bound test for "
            "where downstream native operator degradation makes model recoverability "
            "fail. NEVER report a patched variant as a deployable compact model "
            "or real compressed runtime speed. Top10/token distinctness not semantic meaning."}
    dest=out/"v99_FP32_hidden_boundary_oracles_vs_CB59_actual_greedy.json"
    dest.write_text(json.dumps(output,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v99_completed_non_deployable_causal_boundary_probes",
        "top10_overlaps":{k:v["mean_top10_token_overlap_with_original_FP32"]
                          for k,v in ranking.items()},
        "physical_unchanged_bytes":len(raw),
        "previous_official_TEST_untouched":True},ensure_ascii=False),flush=True)
    return output
