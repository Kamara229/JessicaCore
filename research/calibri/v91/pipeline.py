"""Frozen CB59/CB88 v91 root-cause battery, strictly on public TRAIN slices.
No TEST selection, no weights saved, no deployed Jessica changes.
"""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v88.physical import unpack
from v90.provenance import verify_sources
from v90.qa import build_cases
from .heads import HeadComponents,run_head_ablations
from .greedy import greedy_ablations
from .layers import layerwise_probes,MAX_TOKENS
from .influence import causal_one_block_rescue

def run(original_path:Path,cb88_path:Path,out:Path):
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    raw=original_path.read_bytes()
    cb88=cb88_path.read_bytes()
    sources=verify_sources(raw,cb88)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases,origins,answers=build_cases(tok)
    student,info,_=make_model("baseline",SimpleNamespace(original=original_path,folder=out))
    if not info["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("dense trained source accidentally in runtime")
    _,A,B,bias=unpack(cb88)
    heads=run_head_ablations(student,cases,answers,HeadComponents(A,B,bias))
    greedy=greedy_ablations(student,tok,A,B,bias)
    teacher,local,drift,diagnostics=layerwise_probes(student,cases)
    influence=causal_one_block_rescue(student,teacher,local,diagnostics,answers)
    report={
        "experiment":"CALIBRI_v91_frozen_original_sources_root_cause_decomposition",
        "model":MODEL,"original_pretrained_revision":MODEL_SHA,
        "physical_sources":sources,"public_train_dataset_sources":origins,
        "four_frozen_head_ablations":heads,
        "four_frozen_greedy_diagnostic_modes":greedy,
        "all_original_dense_projection_same_input_relative_errors":local,
        "accumulated_hidden_state_drift_by_transformer_layer":drift,
        "single_module_FP32_oracle_rescue_not_paid_source":influence,
        "deep_probe_splits":[{"split":x["split"],"prompt_SHA":x["prompt_SHA"],
              "truncated_original_train_tokens":x["truncated_tokens"]} for x in diagnostics],
        "no_optimizers_no_training_no_saved_new_model_weights":True,
        "official_independent_TEST_never_loaded_or_tuned":True,
        "no_claim_scientific_PASS":True,
        "Jessica_main_Render_Supabase_APK_untouched":True,
        "limitations":"Only 32+32 public BoolQ train-split examples. Layer local/causal "
           "probes use one truncated 48-token TRAIN context per split. Exact single-layer "
           "FP32 swap is causal oracle diagnostics only and would exceed the compressed "
           "bit budget if deployed; all original compressed weights remain frozen."}
    path=out/"v91_root_cause_head_bias_tensor_block_errors_and_greedy.json"
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    abstract={"head":{split:{k:v["summary"][k] for k in
                     ("CB59","bias_only","tensor_only","CB88")} for split,v in heads.items()},
        "greedy_distinct":{k:v["mean_distinct_ratio"] for k,v in greedy.items()},
        "highest_local_error_blocks":sorted(
            [{"name":k,"rms":v["relative_RMS_error"]} for k,v in local.items()],
             key=lambda x:x["rms"],reverse=True)[:8],
        "best_single_module_oracle_KL_gain":influence["ranked_single_module_rescues"][:5],
        "all_TEST_untouched":True,"artifact":str(path)}
    # Log only scalar summary for CI. Full per-case findings in uploaded artifact.
    abstract["best_single_module_oracle_KL_gain"]=[
        {"name":x["module"],"kl_gain":x["mean_KL_gain_from_exact_FP32_one_module"]}
        for x in abstract["best_single_module_oracle_KL_gain"]]
    print(json.dumps(abstract,ensure_ascii=False),flush=True)
    return report
