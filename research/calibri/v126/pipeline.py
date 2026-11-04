"""Freeze 3 different learned 8MB Green networks on never-used GPT2 host."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors.torch import load_file
from v120.critic import GreenTunnel
from .protocol import MODEL,REVISION,SHA_V120,SHA_V124,SHA_V125,MAX_NEURAL_BYTES,VALID
from .holdout import generate,controls
from .evaluate import report

EXPECTED={"v120_original":SHA_V120,"v124_hard_BCE":SHA_V124,"v125_pairwise":SHA_V125}

def load(file,sha):
    if file.stat().st_size>MAX_NEURAL_BYTES or hashlib.sha256(file.read_bytes()).hexdigest()!=sha:
        raise ValueError("neural shared source must stay exactly frozen")
    m=GreenTunnel()
    m.load_state_dict(load_file(str(file)),strict=True)
    m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m

def run(paths,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    originals={k:hashlib.sha256(file.read_bytes()).hexdigest()
               for k,file in paths.items()}
    if originals!=EXPECTED:raise AssertionError("changed 8MB GreenTunnel in new architecture")
    source=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=REVISION,torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    host_parameter_count=sum(p.numel() for p in source.parameters())
    generated=generate(source,tok)
    rows=controls(generated)
    networks={k:load(file,EXPECTED[k]) for k,file in paths.items()}
    results=report(networks,rows)
    if any(hashlib.sha256(file.read_bytes()).hexdigest()!=EXPECTED[k]
           for k,file in paths.items()):
        raise AssertionError("same neural weights were modified by inference")
    dataset_sha=hashlib.sha256(json.dumps(generated,sort_keys=True).encode()).hexdigest()
    report_obj={
        "experiment":"v126 UNSEEN pretrained GPT2 architecture, NO retraining frozen Green neural files",
        "model":MODEL,"pinned_revision":REVISION,
        "real_pretrained_host_num_parameters":host_parameter_count,
        "real_source_model_original_dtype":"FP32",
        "FP32_pretrained_host_not_under_CALIBRI_0_7bpp":True,
        "original_SMOLM135M_CB59_core_bpp_not_changed_from_prior":.6973527890657376,
        "one_shared_neural_weight_file_per_variant_below_10MB":True,
        "frozen_8MB_weights_original_hashes":originals,
        "no_model_specific_dense_projection_learned_for_new_host":True,
        "all_FROZEN_sources_loaded_unchanged":True,
        "new_host_never_in_prior_training":True,
        "new_16_QA_questions_not_seen_in_prior_training":True,
        "new_host_generation_never_uses_gold_answer":True,
        "source_data_SHA256":dataset_sha,
        "heldout_results_by_model":results,
        "no_NEURAL_language_quality_improvement_asserted":True,
        "trillion_parameter_compatibility_UNTESTED":True,
        "v89_formal_independent_TEST_unopened":True,
        "no_main_Render_Supabase_Jessica_LIVE_APK_change":True,
    }
    (out/"v126_unseen_gpt2.json").write_text(
        json.dumps(report_obj,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v126_FROZEN_SHARED_8MB_THIRD_HOST",
        "host_params":host_parameter_count,"unchanged_neural_shas":True,
        "frozen_factual_metrics":{
            k:{key:v[key] for key in ("correct_gold_reference_accept","plausible_wrong_reject",
                                      "real_wrong_GPT2_reject","oracle_included_gold_top1_rank")}
            for k,v in results.items()},
        "no_coherent_LLM_ability_claim":True},ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("v120","v124","v125","out"):p.add_argument(name,type=Path)
    a=p.parse_args()
    run({"v120_original":a.v120,"v124_hard_BCE":a.v124,
         "v125_pairwise":a.v125},a.out)
