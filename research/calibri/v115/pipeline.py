"""Preregistered multiple-run, label-first comparison on small models.

This replicates task ability, NOT broad reasoning or spontaneous chat quality.
No model selected via validation, and no imitation of pretrained weights.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file

from v114.protocol import MODES,TOKENS,batch
from v114.operators import TinyLanguageModel
from v114.evaluate import assess,paired_speed
from .protocol import SPLITS,SEEDS,TRAINING_STEPS,BATCH,LR,partition

def train(mode,examples,seed):
    torch.manual_seed(seed)
    random.seed(seed)
    rng=random.Random(seed)
    model=TinyLanguageModel(mode)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    model.train()
    losses=[]
    for step in range(TRAINING_STEPS):
        context,prev,answers=batch(examples,BATCH,rng)
        logits=model(context,prev)
        loss=F.cross_entropy(logits.reshape(-1,len(TOKENS)),
                             answers.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step%100==0 or step==TRAINING_STEPS-1:
            losses.append((step,round(float(loss.detach()),5)))
    return model,losses

def summarize(rows):
    results={}
    for mode in MODES:
        cases=[r for r in rows if r["mode"]==mode]
        correct=[r["heldout"]["exact_correct"] for r in cases]
        speeds=[r["cpu_examples_per_second"] for r in cases]
        results[mode]={
            "replicate_count":len(cases),
            "total_heldout_answers":8*len(cases),
            "correct_answers":sum(correct),
            "macro_correct_rate":sum(correct)/(8*len(cases)),
            "min_correct_per_replicate":min(correct),
            "max_correct_per_replicate":max(correct),
            "median_cpu_examples_per_second":statistics.median(speeds),
        }
    paired=[]
    for salt in SPLITS:
        for seed in SEEDS:
            matched={r["mode"]:r for r in rows if r["split"]==salt and r["seed"]==seed}
            base=matched["dense"]
            for mode in MODES[1:]:
                item=matched[mode]
                paired.append({
                    "split":salt,"seed":seed,"mode":mode,
                    "delta_exact_answers":(
                        item["heldout"]["exact_correct"]-
                        base["heldout"]["exact_correct"]),
                    "byte_ratio_to_dense":item["checkpoint_bytes"]/base["checkpoint_bytes"],
                    "cpu_speed_ratio_to_dense":item["cpu_examples_per_second"]/
                        base["cpu_examples_per_second"],
                    "passes_90pct_baseline_and_candidate_and_cpu":(
                        base["heldout"]["exact_correct"]>=8 and
                        item["heldout"]["exact_correct"]>=base["heldout"]["exact_correct"] and
                        item["cpu_examples_per_second"]>=
                          .9*base["cpu_examples_per_second"]),
                })
    return {"models":results,"paired":paired}

def run(out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for salt in SPLITS:
        fitting,valid=partition(salt)
        for seed in SEEDS:
            active={}
            for mode in MODES:
                started=time.perf_counter()
                model,losses=train(mode,fitting,seed)
                exam=assess(model,valid)
                fit_exam=assess(model,fitting)
                file=out/f"{salt}_{seed}_{mode}.safetensors"
                save_file({k:v.detach().cpu().contiguous()
                           for k,v in model.state_dict().items()},str(file))
                saved=TinyLanguageModel(mode)
                saved.load_state_dict(load_file(str(file)),strict=True)
                if assess(saved,valid)["rows"]!=exam["rows"]:
                    raise AssertionError("model archive altered original responses")
                entry={
                    "split":salt,"seed":seed,"mode":mode,
                    "checkpoint_bytes":file.stat().st_size,
                    "checkpoint_sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
                    "trainable_parameters":sum(p.numel() for p in model.parameters()),
                    "fit_time_seconds":time.perf_counter()-started,
                    "train_progress":losses,"train":fit_exam,"heldout":exam,
                }
                active[mode]=(model,entry)
                rows.append(entry)
            speed=paired_speed({mode:pair[0] for mode,pair in active.items()},repeat=2)
            for mode,(_,entry) in active.items():
                entry["cpu_examples_per_second"]=speed[mode]["examples_per_second"]
                print(json.dumps({
                    "stage":"v115_replicate","split":salt,"seed":seed,"mode":mode,
                    "correct_heldout":entry["heldout"]["exact_correct"],
                    "correct_fit":entry["train"]["exact_correct"],
                    "grammar":entry["heldout"]["grammar_rate"],
                    "checkpoint_bytes":entry["checkpoint_bytes"],
                    "cpu_examples_per_second":round(entry["cpu_examples_per_second"],3),
                }),flush=True)
    comparison=summarize(rows)
    report={
        "study":"CALIBRI v115 quality-first multiple-split multiple-seed small-model check",
        "objective":"new model should give correct grammatical answers; no teacher KL/NLL",
        "model_limit":"narrow synthetic single-fact color QA only; not real open-domain speech",
        "predeclared_splits":SPLITS,"predeclared_seeds":SEEDS,
        "training_steps":TRAINING_STEPS,
        "all_18_runs":rows,"summary":comparison,
        "note":"both splits contain pairs from same 36-pair synthetic domain examined in v114; cannot assert statistically independent unseen domain generalization",
        "no_full_pretrained_llm_070bps_claim":True,
        "no_official_v89_test_read":True,
        "no_live_modifications":True,
    }
    (out/"v115_replication.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":"v115_finished",
        "summary":comparison["models"],
        "paired":comparison["paired"],
    }),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    run(p.parse_args().out)
