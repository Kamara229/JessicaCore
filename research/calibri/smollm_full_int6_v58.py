#!/usr/bin/env python3
"""CALIBRI v58: *new stronger instruct-tuned* open weights real physical INT6.

Model HuggingFaceTB/SmolLM2-135M-Instruct, Apache-2.0. First CALIBRI
experiment outside Pythia-14M. Source weights are bfloat16 on HF; for
fair CPU FP32 teacher and signed INT6 six-bit source the trained BF16
numbers are loaded into model FP32. Tied token embedding/output weights
are stored exactly ONCE with CB58 alias manifest (CB46 signed INT6 for
all unique 2D floating source weights, exact 1D). Do not silently count
or store alias weight twice; source parameter count equals model
named_parameters() unique trained parameters.

One fixed source-only SSE-balanced 12%-per-tensor FP16 restoration sidecar
up to 6.50 real physical bits per *unique pretrained parameter*, with
15480 bytes reserved. This is original source weight restoration, not
gradient/task-trained correction. A single preregistered TRAIN-only gate
scores first-use SciQ/BoolQ 16 training QA items each. After choice
is immutable, first-use SciQ TEST and BoolQ VALIDATION 32 each for
FP32 teacher/plain INT6/physical corrected INT6. Real same-host warm
full 128-token prefill+32 autoregressive KV-cache steps, 2+7 times.

At-rest INT6 only; entire model expands into dense FP32 hot RAM once.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,AutoConfig
from huggingface_hub import HfApi
from quality import load_token_ids,DATASET
from shared_checkpoint_v58 import (
    pack_shared,unpack_shared,cold_load_shared,smoke as shared_smoke)
from deterministic_budget_v49 import (
    candidates,choose,smoke as deterministic_smoke)
from sparse_recovery_v47 import (
    encode_sparse_patch,parse_patch,apply_patch)
from practical_int6_v46 import timing
from task_margin_v57 import (
    row_from_task,evaluate_real_task_margins,tasks_sha)
from datasets import load_dataset
from gradient_recovery_experiment_v47 import physical_write

MODEL58="HuggingFaceTB/SmolLM2-135M-Instruct"
TRAIN=(
    ("SciQ","allenai/sciq",None,"train[864:880]"),
    ("BoolQ","super_glue","boolq","train[608:624]"),
)
HELDOUT=(
    ("SciQ","allenai/sciq",None,"test[352:384]"),
    ("BoolQ","super_glue","boolq","validation[480:512]"),
)
RESERVED_BYTES=15480
MAX_PHYS_BPW=6.5
WINDOW=128
PROMPT=128
DECODE=32


def candidate_train_gate(train):
    """First-use actual task 16-case binary and margin TRAIN-only gate."""
    if set(train)!={"base_int6_pre","source_sse_balanced_repair"}:
        raise AssertionError("unregistered train candidates")
    families=("SciQ","BoolQ")
    base=train["base_int6_pre"]
    patch=train["source_sse_balanced_repair"]
    improvements={}
    count_delta={}
    for name in families:
        if base[name]["n"]!=16 or patch[name]["n"]!=16:
            raise AssertionError("TRAIN task count changed")
        improvements[name]=(
            patch[name]["mean_gold_logprob_margin"]-
            base[name]["mean_gold_logprob_margin"])
        count_delta[name]=patch[name]["correct"]-base[name]["correct"]
    eligible=(sum(count_delta.values())>=0 and
              min(count_delta.values())>=0 and
              min(improvements.values())>0. and
              math.fsum(improvements.values())/2>=0.001)
    chosen=("source_sse_balanced_repair" if eligible else "base_int6_pre")
    return {"chosen":chosen,
            "immutable_rule":"Both TRAIN families: no lost correct answers (16 each), positive mean gold-vs-wrong margin, macro margin gain>=0.001. Else plain INT6 PRE.",
            "training_exact_correct_deltas":count_delta,
            "training_family_margin_deltas":improvements,
            "eligible_for_expensive_repair":eligible,
            "TRAIN_only_full_task_metrics":train}



def load_two_tasks(definitions,revisions,expected):
    """This is intentionally a TWO-task pilot, not v57 four-task suite.

    Reuses exactly the same pretrained zero-shot answer text
    normalization, but changes only the task set scope. The earlier
    four-task loader rejected this valid frozen pre-TEST pilot scope.
    """
    tasks={}
    for name,repo,cfg,split in definitions:
        ds=(load_dataset(repo,cfg,revision=revisions[name],split=split)
            if cfg else load_dataset(repo,revision=revisions[name],split=split))
        tasks[name]=[row_from_task(name,row) for row in ds]
        if len(tasks[name])!=expected:
            raise AssertionError("frozen exact TRAIN/TEST task length mismatch")
    if set(tasks)!={"SciQ","BoolQ"}:
        raise AssertionError("unexpected genuine instruction QA pilot task source")
    return tasks

def load_disk_model(cfg,base_path,source_hash,patch_file,method):
    model=AutoModelForCausalLM.from_config(cfg,torch_dtype=torch.float32)
    evidence=cold_load_shared(model,base_path)
    if method=="source_sse_balanced_repair":
        evidence["corrected"]=apply_patch(model,patch_file,source_hash)
    elif method!="base_int6_pre":
        raise AssertionError("unsupported blind model choice")
    if model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr():
        raise AssertionError("recovered weights no longer tied at runtime")
    model.eval()
    return model,evidence


def audit_task_metrics(metrics,expected):
    for method,scores in metrics.items():
        for name in scores:
            r=scores[name]
            if r["n"]!=expected or r["correct"]!=sum(r["per_item_correct"]):
                raise AssertionError("incorrect real answer count evidence "+method+"/"+name)


def main(output):
    if not (os.getenv("ATEN_CPU_CAPABILITY")=="avx2" and
            os.getenv("DNNL_MAX_CPU_ISA")=="AVX2" and
            os.getenv("ONEDNN_MAX_CPU_ISA")=="AVX2" and
            str(torch._C._get_cpu_capability()).upper()=="AVX2"):
        raise AssertionError("pin both independent hosts to AVX2")
    torch.manual_seed(58)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    shared_smoke()
    deterministic_smoke()
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    api=HfApi()
    rev=api.model_info(MODEL58).sha
    wiki_rev=api.dataset_info(DATASET).sha
    dataset_revs={name:api.dataset_info(repo).sha for name,repo,_,_ in TRAIN}
    tok=AutoTokenizer.from_pretrained(MODEL58,revision=rev)
    cfg=AutoConfig.from_pretrained(MODEL58,revision=rev)
    if not cfg.tie_word_embeddings or cfg.model_type!="llama":
        raise AssertionError("source instruction tuned Llama model alias expected")
    original=AutoModelForCausalLM.from_pretrained(
        MODEL58,revision=rev,torch_dtype=torch.float32)
    original.eval()
    state=original.state_dict()
    unique_count=sum(p.numel() for p in original.parameters())
    base,meta,unique,_=pack_shared(state)
    if unique_count!=meta["unique_source_parameter_count"]:
        raise AssertionError("tied embedding incorrectly counted in bpw denominator")
    if not meta["alias_derived_tensors"]:
        raise AssertionError("source model ties embedding and output weights; alias missing")
    base_file=path.parent/"CALIBRI_v58_SmolLM2_full_INT6.cb58"
    base_evidence=physical_write(base_file,base)
    if len(base)<10_000_000:
        raise AssertionError("this is not a full pretrained 135M checkpoint")
    runtime,verified_meta=unpack_shared(base,expected_keys=state.keys())
    decoded={name:runtime[name] for name in unique}
    max_bytes=math.floor(MAX_PHYS_BPW*unique_count/8)
    spare=max_bytes-len(base)-RESERVED_BYTES
    if spare<200000:
        raise AssertionError("real complete model 6.5bpw patch budget exhausted")
    scores,geometries=candidates(unique,decoded)
    proposed=choose(scores,spare)["sse_balanced_12pct"]
    selected_rows=proposed["selection"]
    payload=encode_sparse_patch(unique,decoded,selected_rows,
                                base_evidence["sha256"])
    patch_file=path.parent/"CALIBRI_v58_SmolLM2_source_repair.cr47"
    patch_evidence=physical_write(patch_file,payload)
    shapes={k:tuple(v.shape) for k,v in unique.items() if v.ndim==2}
    reconstructed=parse_patch(patch_file.read_bytes(),
                              base_evidence["sha256"],shapes)
    if (len(payload)!=proposed["actual_patch_bytes_expected"] or
        len(reconstructed)!=len(selected_rows) or
        len(base)+len(payload)+RESERVED_BYTES>max_bytes):
        raise AssertionError("physical full model correction exceeds 6.50bpw")
    damaged=bytearray(payload)
    damaged[-1]^=1
    try:parse_patch(damaged,base_evidence["sha256"],shapes)
    except ValueError:pass
    else:raise AssertionError("physical correction SHA/CRC failed")

    # Freeze full physical source and correction BEFORE loading data used
    # for accept/reject. TRAIN-label choices and output scoring are
    # deliberately kept independent from stored row precision allocation.
    train_tasks=load_two_tasks(TRAIN,dataset_revs,16)
    train_digests=tasks_sha(train_tasks,TRAIN)
    training={}
    for method in ("base_int6_pre","source_sse_balanced_repair"):
        q,evidence=load_disk_model(cfg,base_file,base_evidence["sha256"],
                                   patch_file,method)
        training[method]=evaluate_real_task_margins(
            q,tok,train_tasks,method+"_TRAIN")
        del q
    audit_task_metrics(training,16)
    gate=candidate_train_gate(training)
    chosen=gate["chosen"]
    # This is the *first* point at which untouched TEST questions and
    # their correctness labels can be read.
    test=load_two_tasks(HELDOUT,dataset_revs,32)
    test_digests=tasks_sha(test,HELDOUT)
    teacher_metrics=evaluate_real_task_margins(
        original,tok,test,"teacher_HELDOUT")
    tests={}
    for method in ("base_int6_pre","source_sse_balanced_repair"):
        q,_=load_disk_model(cfg,base_file,base_evidence["sha256"],patch_file,method)
        tests[method]=evaluate_real_task_margins(
            q,tok,test,method+"_HELDOUT")
        del q
    audit_task_metrics({**tests,"teacher":teacher_metrics},32)

    wiki_tokens=load_token_ids(tok,"train",wiki_rev,160)
    prefill=wiki_tokens[:128]
    follow=wiki_tokens[128:160]
    reference_timing=timing(original,prefill,follow)
    del original
    selected,evidence=load_disk_model(cfg,base_file,base_evidence["sha256"],
                                      patch_file,chosen)
    chosen_timing=timing(selected,prefill,follow)
    del selected
    warm=chosen_timing["total_prefill_and_decode32_ms"]["median"]/(
         reference_timing["total_prefill_and_decode32_ms"]["median"])

    real_selected_bytes=len(base)+(len(payload) if
        chosen=="source_sse_balanced_repair" else 0)
    report={
        "experiment":"CALIBRI_v58_open_instruction_model_135m_first_real_INT6",
        "new_model":MODEL58,"original_model_revision":rev,
        "trained_instruction_model":True,"hf_license":"apache-2.0",
        "architecture":cfg.model_type,"unique_original_parameter_count":unique_count,
        "original_source_dtype":"HF BF16 converted to FP32 teacher for paired CPU",
        "source_tied_weight_aliases":meta["alias_derived_tensors"],
        "entire_physical_checkpoint":base_evidence,
        "full_CB58_source_format":"alias-manifest + entire CB46 six-bit unique-trained-state archive, all CRC/SHA",
        "source_only_physical_FP16_sparse_repair":{
            **patch_evidence,"selected_original_rows":len(selected_rows),
            "per_matrix_row_counts":proposed["module_counts"],
            "max_patch_bytes":spare,"full_model_plus_patch_bytes":len(base)+len(payload),
            "full_model_plus_patch_bpw":8*(len(base)+len(payload))/unique_count},
        "reserved_other_post_bytes":RESERVED_BYTES,
        "full_entire_model_MAX_6_50bpw_bytes":max_bytes,
        "selected_method":chosen,
        "selected_actual_stored_bytes":real_selected_bytes,
        "selected_actual_stored_bpw":8*real_selected_bytes/unique_count,
        "training_dataset_revision":dataset_revs,
        "source_WikiText_train_revision":wiki_rev,
        "TRAIN_real_2x16_task_text_SHA256":train_digests,
        "one_shot_TRAIN_task_gate":gate,
        "first_read_after_gate_HELDOUT_2x32_SHA256":test_digests,
        "heldout_teacher_real_task_accuracy":teacher_metrics,
        "heldout_all_INT6_methods_task_accuracy":tests,
        "actual_full_model_speed":{
            "input_prompt_tokens":128,"KV_cache_autoregressive_decode_steps":32,
            "warmups":2,"repeats":7,
            "original_uncompressed_teacher":reference_timing,
            "from_physical_files_chosen":chosen_timing,
            "chosen_to_original_warm_total_ratio":warm,
            "user_20pct_max_slowdown_pass":warm<=1.20,
            "fresh_checkpoint_and_optional_sparse_restore":evidence,
        },
        "limits":[
           "This is actual stronger 135M instruction-tuned open weights, but it is still NOT production Jessica.",
           "INT6 compressed on DISK and expanded to full FP32 hot active RAM once at cold startup; no native packed INT6 compute, no RAM savings.",
           "True effective bpw counts only UNIQUE tied pretrained tensor parameters and actual physical alias metadata, preventing false source-rate denominator.",
           "The quality gate uses only two tiny 16-case TRAIN task families; independent 32-case first-use HELDOUT per task is underpowered, no statistical superiority claim.",
           "All physical correction rows are chosen by source-only deterministic quantization error, without test/label data.",
           "Source model license must be honored by any user who later distributes derivative weights; CI only offline public open weights.",
           "Research-only draft PR; main, live Jessica, Render, Supabase and APK untouched."
        ]}
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({
        "v58":"MEASURED_REAL_OPEN_INSTRUCTION_MODEL",
        "source_model":MODEL58,
        "model_SHA":rev,
        "unique_parameters":unique_count,
        "tied_alias_names":meta["alias_derived_tensors"],
        "full_real_CB58_BYTES":len(base),
        "CB58_SHA":base_evidence["sha256"],
        "sparse_residual_BYTES":len(payload),
        "patch_SHA":patch_evidence["sha256"],
        "source_plus_patch_bpw":8*(len(base)+len(payload))/unique_count,
        "TRAIN_only_choice":chosen,
        "TRAIN_correct":{m:{n:x["correct"] for n,x in v.items()}
                         for m,v in training.items()},
        "HELDOUT_correct":{
            "teacher":{n:x["correct"] for n,x in teacher_metrics.items()},
            **{m:{n:x["correct"] for n,x in v.items()} for m,v in tests.items()}},
        "actual_warm_selected_over_teacher":warm,
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",default="results/calibri_v58.json")
    args=ap.parse_args()
    main(args.output)
