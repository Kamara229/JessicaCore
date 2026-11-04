"""Experiment launcher: optimize actual answer ground truth, not hidden teacher state."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import save_file,load_file

from .protocol import TRAIN,VALID,TOKENS,MODES,SEED,STEPS,BATCH,LR,batch
from .operators import TinyLanguageModel
from .evaluate import assess,paired_speed

def fit(mode):
    random.seed(SEED)
    torch.manual_seed(SEED)
    generator=random.Random(SEED)
    model=TinyLanguageModel(mode)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    progress=[]
    model.train()
    for step in range(STEPS):
        context,previous,gold=batch(TRAIN,BATCH,generator)
        logits=model(context,previous)
        loss=F.cross_entropy(
            logits.reshape(-1,len(TOKENS)),gold.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%100==0 or step==STEPS-1:
            progress.append({
                "step":step,
                "supervised_answer_train_loss":round(float(loss.detach()),5)})
    return model,progress

def run(out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    report={
        "experiment":"CALIBRI v114 functionality-first tiny structural operators",
        "selection_goal":"ground-truth responses and grammatical first-turn, no original logits",
        "scope":"Toy color/object grounded QA; no open-domain LLM claim",
        "source":{
            "seed":SEED,"training_steps":STEPS,"batch":BATCH,
            "learning_rate":LR,"train_pairs":len(TRAIN),
            "validation_unseen_pairs":len(VALID),
            "heldout_cases_disjoint":not bool(set(TRAIN)&set(VALID)),
        },
        "models":{},"comparisons":{},
        "does_not_compress_existing_pretrained_weights":True,
        "entire_model_saved_without_excluding_head_or_embedding":True,
        "no_old_v89_test_opened":True,
        "no_production_change":True,
    }
    models={}
    for mode in MODES:
        start=time.perf_counter()
        model,losses=fit(mode)
        trained=assess(model,TRAIN)
        heldout=assess(model,VALID)
        path=out/(mode+".safetensors")
        save_file({
            key:v.detach().cpu().contiguous()
            for key,v in model.state_dict().items()
        },str(path))
        contents=path.read_bytes()
        restored=TinyLanguageModel(mode)
        restored.load_state_dict(load_file(str(path)),strict=True)
        # Full checkpoint is sufficient for identical inference.
        if assess(restored,VALID)["rows"]!=heldout["rows"]:
            raise AssertionError("serialized checkpoint did not reproduce model answers")
        report["models"][mode]={
            "whole_checkpoint_bytes":len(contents),
            "whole_checkpoint_sha256":hashlib.sha256(contents).hexdigest(),
            "all_trainable_parameters":sum(p.numel() for p in model.parameters()),
            "training_seconds":time.perf_counter()-start,
            "fit_losses":losses,
            "train_answers":trained,
            "heldout_answers":heldout,
        }
        models[mode]=model
        print(json.dumps({
            "stage":"v114_trained","mode":mode,
            "entire_checkpoint_bytes":len(contents),
            "parameters":report["models"][mode]["all_trainable_parameters"],
            "train_correct":trained["correct_rate"],
            "heldout_correct":heldout["correct_rate"],
            "heldout_grammar":heldout["grammar_rate"],
            "heldout_eos":heldout["eos_rate"],
        }),flush=True)
    speed=paired_speed(models)
    for mode in MODES:
        report["models"][mode]["cpu_autoregressive"]=speed[mode]
    reference=report["models"]["dense"]
    for mode in MODES[1:]:
        candidate=report["models"][mode]
        report["comparisons"][mode]={
            "whole_checkpoint_byte_ratio":(
                candidate["whole_checkpoint_bytes"]/
                reference["whole_checkpoint_bytes"]),
            "correctness_delta_vs_dense":(
                candidate["heldout_answers"]["correct_rate"]-
                reference["heldout_answers"]["correct_rate"]),
            "grammar_delta_vs_dense":(
                candidate["heldout_answers"]["grammar_rate"]-
                reference["heldout_answers"]["grammar_rate"]),
            "cpu_examples_per_second_ratio_vs_dense":(
                speed[mode]["examples_per_second"]/
                speed["dense"]["examples_per_second"]),
            "functional_quality_pass":(
                reference["heldout_answers"]["correct_rate"]>=.9
                and candidate["heldout_answers"]["correct_rate"]>=
                    reference["heldout_answers"]["correct_rate"]
                and candidate["heldout_answers"]["grammar_rate"]>=
                    reference["heldout_answers"]["grammar_rate"]),
            "real_cpu_speed_pass_at_0_9_dense":(
                speed[mode]["examples_per_second"]>=
                .9*speed["dense"]["examples_per_second"]),
        }
    report["caveats"]={
        "QA_is_only_8_heldout_color_object_combinations":True,
        "baseline_must_pass_90pct_quality_gate_before_claim":True,
        "not_0_7_bits_per_pretrained_parameter":True,
        "not_semantic_general_language":True,
        "no_KL_or_NLL_targets_in_training_or_selection":True,
    }
    (out/"v114_behavior.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({
        "stage":"v114_done",
        "baseline_correctness":reference["heldout_answers"]["correct_rate"],
        "comparisons":report["comparisons"],
    }),flush=True)

if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("out",type=Path)
    run(arg.parse_args().out)
