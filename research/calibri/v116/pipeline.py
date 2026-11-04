"""Real frozen-checkpoint compact-storage vs generated hot-GEMV speed audit.

NO retraining. NO imitation of FP32 weights/logits. Primary invariant is
identical user-visible responses before/after transforming *the same*
compact trained functional operators into a temporary native dense kernel.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path
import torch
from safetensors.torch import load_file

from v114.protocol import COLORS,OBJECTS,example,MODES
from v114.operators import TinyLanguageModel
from v114.evaluate import paired_speed
from v115.protocol import SPLITS,SEEDS,partition
from .runtime import (
    FusedBlock,fuse_compact,hot_generated_weight_bytes,
    hot_generated_all_buffers_bytes)
from .protocol import (
    ORIGINAL_JSON_SHA256,BASELINE_V115_ACTIONS_RUN,
    BASELINE_ARTIFACT,MAX_ABS_LOGIT_DIFF,CPU_MIN_RATIO,REPEATS)

def immutable_source(path):
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=ORIGINAL_JSON_SHA256:
        raise AssertionError("v115 immutable evidence JSON changed")
    return json.loads(blob)

def pair_baselines(source,root):
    # Every individual 18-file SHA must match immutable original evidence.
    original=source["all_18_runs"]
    expected={(r["split"],r["seed"],r["mode"]):r for r in original}
    if len(expected)!=18:
        raise AssertionError("missing distinct original trained sources")
    loaded={}
    for salt in SPLITS:
        for seed in SEEDS:
            for mode in MODES:
                info=expected[(salt,seed,mode)]
                name=f"{salt}_{seed}_{mode}.safetensors"
                path=root/name
                if path.stat().st_size!=info["checkpoint_bytes"]:
                    raise AssertionError("changed physical original weights size")
                if hashlib.sha256(path.read_bytes()).hexdigest()!=info["checkpoint_sha256"]:
                    raise AssertionError("changed original learned weights SHA")
                model=TinyLanguageModel(mode)
                model.load_state_dict(load_file(str(path)),strict=True)
                model.eval()
                loaded[(salt,seed,mode)]=model
    return loaded

@torch.inference_mode()
def verify_same_answers(original,fused):
    all_cases=[example(color,obj) for color in COLORS for obj in OBJECTS]
    if len(all_cases)!=36:
        raise AssertionError("expected complete tiny synthetic domain")
    for item in all_cases:
        context=torch.tensor(item["context"],dtype=torch.long)
        if original.generate(context)!=fused.generate(context):
            raise AssertionError("fused inference changed an existing trained answer")
    # Logit equality on actual prefixes and independent random token sequences.
    largest=0.
    generator=torch.Generator().manual_seed(116)
    for item in all_cases[:4]:
        context=torch.tensor([item["context"]],dtype=torch.long)
        for previous in (
            torch.tensor([[2]],dtype=torch.long),
            torch.tensor([[2,4,19,5,15,6]],dtype=torch.long),
            torch.randint(0,len(COLORS)+len(OBJECTS)+11,(1,6),generator=generator),
        ):
            dense=original(context,previous)
            scratch=fused(context,previous)
            largest=max(largest,float(torch.max(torch.abs(dense-scratch))))
    if largest>MAX_ABS_LOGIT_DIFF:
        raise AssertionError("generated CPU GEMV changed observable logits")
    return largest,len(all_cases)

def measure(root,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    source=immutable_source(root/"v115_replication.json")
    trained=pair_baselines(source,root)
    report={
        "experiment":"CALIBRI v116 structural on-disk compact vs hot generated block GEMV",
        "source_v115_run":BASELINE_V115_ACTIONS_RUN,
        "source_v115_artifact":BASELINE_ARTIFACT,
        "exact_source_v115_JSON_sha256":ORIGINAL_JSON_SHA256,
        "original_all_18_checkpoints_verified":True,
        "new_trained_weights":False,
        "full_physical_compact_archive_is_immutable":True,
        "generated_operator_blocks_only_in_hot_RAM":True,
        "this_toy_caches_all_3_blocks_without_eviction":True,
        "does_NOT_measure_real_135M_model_070bpp":True,
        "no_KL_NLL_selection":True,
        "no_original_v89_TEST_access":True,
        "no_production_changes":True,
        "cases":[],
    }
    for salt in SPLITS:
        for seed in SEEDS:
            dense=trained[(salt,seed,"dense")]
            compact=trained[(salt,seed,"shared_adapter")]
            fused=fuse_compact(compact)
            maxdelta,ncontexts=verify_same_answers(compact,fused)
            ratio=paired_speed({
                "dense":dense,"compact":compact,"generated_fused":fused
            },repeat=REPEATS)
            physical={
                r["mode"]:r["checkpoint_bytes"]
                for r in source["all_18_runs"]
                if r["split"]==salt and r["seed"]==seed
            }
            if physical["shared_adapter"]>=physical["dense"]:
                raise AssertionError("structural model did not save physical bytes")
            new_buffers=hot_generated_weight_bytes(fused)
            all_buffers=hot_generated_all_buffers_bytes(fused)
            if new_buffers<=0 or all_buffers<new_buffers:
                raise AssertionError("hot physical operator budget missing")
            result={
                "split":salt,"seed":seed,
                "all_36_context_answer_token_sequences_identical":True,
                "max_abs_original_vs_generated_logits":maxdelta,
                "reused_model_whole_source_bytes":physical["shared_adapter"],
                "dense_baseline_whole_source_bytes":physical["dense"],
                "structural_archive_fraction_of_dense":(
                    physical["shared_adapter"]/physical["dense"]),
                "generated_dense_2D_operator_RAM_bytes":new_buffers,
                "all_generated_block_buffer_bytes":all_buffers,
                "cpu_original_compact_ratio":(
                    ratio["compact"]["examples_per_second"]/
                    ratio["dense"]["examples_per_second"]),
                "cpu_generated_fused_ratio":(
                    ratio["generated_fused"]["examples_per_second"]/
                    ratio["dense"]["examples_per_second"]),
                "cpu_fused_meets_predeclared_speed_floor":(
                    ratio["generated_fused"]["examples_per_second"]>=
                    CPU_MIN_RATIO*ratio["dense"]["examples_per_second"]),
                "paired_cpu_raw":ratio,
            }
            report["cases"].append(result)
            print(json.dumps({
                "stage":"v116_frozen","split":salt,"seed":seed,
                "source_bytes":physical["shared_adapter"],
                "generated_hot_weight_bytes":new_buffers,
                "answer_parity":True,
                "max_logit_delta":maxdelta,
                "compact_speed_ratio":round(result["cpu_original_compact_ratio"],4),
                "fused_speed_ratio":round(result["cpu_generated_fused_ratio"],4),
            }),flush=True)
    fused_rates=[row["cpu_generated_fused_ratio"] for row in report["cases"]]
    raw_rates=[row["cpu_original_compact_ratio"] for row in report["cases"]]
    report["summary"]={
        "trained_checkpoint_count":18,
        "mixed_compact_source_count":6,
        "all_36_generated_sequences_parity_per_replica":True,
        "all_216_generated_sequences_identical":True,
        "median_cpu_structural_original_vs_dense":statistics.median(raw_rates),
        "median_cpu_generated_fused_vs_dense":statistics.median(fused_rates),
        "minimum_cpu_generated_fused_vs_dense":min(fused_rates),
        "count_fused_speed_pass":sum(row["cpu_fused_meets_predeclared_speed_floor"]
                                     for row in report["cases"]),
        "replicates":len(report["cases"]),
        "no_permanent_dense_source_checkpoint":True,
        "not_a_complete_pretrained_LLM":True,
    }
    (out/"v116_fused_cpu.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v116_completed","summary":report["summary"]}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    measure(args.source,args.out)
