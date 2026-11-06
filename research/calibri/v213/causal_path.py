"""CALIBRI v213: TRAIN-only causal localization of v206 output collapse.

Seven read-only teacher-hidden injection points + a crossed 2x2 diagnostic
of tied FP32/packed embedding-head and final language state. Original FP32
weights and mixed outputs here exist *ONLY* for diagnostic reference; none
are saved as an inference model or claimed to fit the 0.7 bpp requirement.

No public DEV, VALID or TEST may enter this script; no weight update occurs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v84.sources import original_teacher
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v176.operators import clear
from v212.run import SOURCE_V206_SHA256

# Frozen, balanced two opposing-prompt pairs from each of four TRAIN domains.
CAUSAL_TRAIN_PAIR_IDS = (0,1,6,7,12,13,18,19)
PROBE_COUNT = 16


def summarize_mode(rows, name):
    arr = [r["variants"][name] for r in rows]
    return {
        "correct_TRAIN_first_tokens":sum(int(v["correct_gold"]) for v in arr),
        "agree_FP32_first_tokens":sum(int(v["agree_fp32"]) for v in arr),
        "distinct_first_tokens":len({v["chosen"] for v in arr}),
        "predicted_first_token_counts":dict(Counter(v["chosen"] for v in arr)),
        "prompts":len(arr),
    }


@torch.no_grad()
def probe(v206_source, out):
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    source_bytes=Path(v206_source).read_bytes()
    if hashlib.sha256(source_bytes).hexdigest()!=SOURCE_V206_SHA256:
        raise AssertionError("Causal probe must start from authentic sealed v206")
    torch.manual_seed(213)
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_SHA)
    _, all_pairs, _, origin = training_inputs(tokenizer)
    if len(all_pairs)!=24 or any(all_pairs[i]["split"]!="TRAIN" for i in CAUSAL_TRAIN_PAIR_IDS):
        raise AssertionError("Frozen TRAIN protocol changed or leaked VALID/DEV")
    teacher = original_teacher().cpu().eval()
    student, sources = build_trainable_from_paid_source(CB59(source_bytes))
    student.eval()
    if teacher.config.num_hidden_layers!=len(student.model.layers):
        raise AssertionError("FP32 and packed model architecture mismatched")
    total_layers = len(student.model.layers)
    checkpoints = sorted(set([0, total_layers//6, total_layers//3, total_layers//2,
                              (2*total_layers)//3, (5*total_layers)//6, total_layers-1]))
    fp32_head = teacher.get_output_embeddings().weight.detach()
    seen=[]
    for pair_id in CAUSAL_TRAIN_PAIR_IDS:
        pair = all_pairs[pair_id]
        for side in ("left","right"):
            sample=pair[side]
            ids=sample["prompt_ids"][None].long()
            gold=int(sample["first_expected_id"])
            if not 0<=gold<fp32_head.shape[0]:
                raise AssertionError("Invalid frozen TRAIN target")
            original_state={}
            hooks=[]
            def save_teacher_norm(_module,_input,output):
                original_state["norm"]=output.detach().clone()
            hooks.append(teacher.model.norm.register_forward_hook(save_teacher_norm))
            for index in checkpoints:
                def save_teacher_layer(_module,_input,output,idx=index):
                    x=output[0] if isinstance(output,tuple) else output
                    original_state[idx]=x.detach().clone()
                hooks.append(teacher.model.layers[index].register_forward_hook(save_teacher_layer))
            try:
                teacher_logits=teacher(input_ids=ids,use_cache=False).logits[0,-1]
            finally:
                for h in hooks:h.remove()
            if set(original_state)!=(set(checkpoints)|{"norm"}):
                raise AssertionError("Teacher hidden layers not captured")
            teacher_choice=int(teacher_logits.argmax())
            compact_state={}
            def save_compact_norm(_module,_input,output):
                compact_state["norm"]=output.detach().clone()
            clear(sources)
            hook=student.model.norm.register_forward_hook(save_compact_norm)
            try:
                compact_logits=student(input_ids=ids,use_cache=False).logits[0,-1]
            finally:
                hook.remove()
            if "norm" not in compact_state:
                raise AssertionError("Missing compressed final hidden state")
            variants={}
            def rec(key,logits):
                if not torch.isfinite(logits).all():
                    raise AssertionError("Nonfinite TRAIN first-token logits")
                choice=int(logits.argmax())
                variants[key]={
                    "chosen":choice,
                    "correct_gold":choice==gold,
                    "agree_fp32":choice==teacher_choice,
                    "gold_margin_vs_top1":round(float(logits[gold]-logits.max()),5),
                }
            rec("FP32",teacher_logits)
            rec("CB206",compact_logits)
            # Dense FP32 head on compressed hidden: isolates output-map damage.
            rec("FP32_HEAD_with_CB206_HIDDEN",
                F.linear(compact_state["norm"][0,-1],fp32_head))
            # Compressed head on teacher hidden: isolates hidden-state damage.
            rec("CB206_HEAD_with_FP32_HIDDEN",
                student.lm_head(original_state["norm"])[0,-1])
            clear(sources)
            for index in checkpoints:
                hidden=original_state[index]
                def replace_output(_module,_input,output,teacher_state=hidden):
                    if isinstance(output,tuple):
                        return (teacher_state,)+output[1:]
                    return teacher_state
                handler=student.model.layers[index].register_forward_hook(replace_output)
                try:
                    clear(sources)
                    swapped_logits=student(input_ids=ids,use_cache=False).logits[0,-1]
                finally:
                    handler.remove()
                    clear(sources)
                rec(f"FP32_LAYER_{index}_HIDDEN_INJECTION",swapped_logits)
            seen.append({
                "TRAIN_pair_id":pair_id,"domain":pair["domain"],"side":side,
                "expected_first_token":gold,"FP32_first_token":teacher_choice,
                "variants":variants
            })
            print(json.dumps({"stage":"v213_causal_probe",
                "done":len(seen),"total":PROBE_COUNT,
                "domain":pair["domain"],"side":side,
                "FP32":variants["FP32"]["chosen"],
                "v206":variants["CB206"]["chosen"],
                "fp32_head_on_compact":variants["FP32_HEAD_with_CB206_HIDDEN"]["chosen"],
                "compact_head_on_fp32":variants["CB206_HEAD_with_FP32_HIDDEN"]["chosen"]
            }),flush=True)
    if len(seen)!=PROBE_COUNT:
        raise AssertionError("TRAIN-only causal sample incomplete")
    all_variants=sorted(seen[0]["variants"])
    report={
        "version":"CALIBRI_v213_read_only_causal_qualification",
        "source_verified_v206_sha256":SOURCE_V206_SHA256,
        "training_pair_indices":list(CAUSAL_TRAIN_PAIR_IDS),
        "fp32_layer_probe_indices":checkpoints,
        "total_teacher_layers":total_layers,
        "TRAIN_case_count":len(seen),
        "summary_by_variant":{name:summarize_mode(seen,name) for name in all_variants},
        "detailed_TRAIN_cases":seen,
        "dense_FP32_head_or_hidden_are_DIAGNOSTIC_ONLY_NOT_COMPACT":True,
        "NO_CHECKPOINT_TRAINING_OR_SELECTION_BY_DEV_VALID_TEST":True,
        "NO_RETRAINING_OR_CHANGES_TO_SEALED_v206":True,
        "no_LIVE_or_Jessica_runs":True
    }
    (out/"v213_TRAIN_CAUSAL_head_vs_hidden_layer_interventions.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v213_causal_completed",
        "source":SOURCE_V206_SHA256,
        "TRAIN_case_count":len(seen),
        "summary":report["summary_by_variant"]},ensure_ascii=False),flush=True)
    return report


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("sealed_v206",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    probe(args.sealed_v206,args.output)
