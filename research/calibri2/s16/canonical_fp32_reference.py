"""S16 frozen pretrained FP32 135M *reference-only* on S15 complex QA.

User correctly challenged easy Yes/No tests and previously requested knowing
what the original 135M FP32 model can actually answer. The physical paid 36M
CALIBRI from-scratch model is NOT trained from these weights or this test,
and never imports/derives the FP32 reference. This read-only benchmark
downloads the canonical frozen instruction model for evaluation only.

Exactly the same composite 192 Yes/No-plus-color twin suite, 16 open prompts,
same tokenizer, same greedy max-tokens and answer parsing as S15. This is
a scale/weight/provenance DIFFERENT baseline, not iso-architecture parity,
and not a QA-selected checkpoint. Store full actual answer token sequences
and negative results; never store/copy original FP32 pretrained weights to
research or LIVE; do not let predictions influence CALIBRI training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s3.fresh_heldout import generate
from calibri2.s15.compositional_qa import (
    make_cases, OPEN_QUALITY, ANSWER_CAP, OPEN_CAP,
    parse_two_slots, compare_rows,
)
from v174.run import save


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(2116)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        attn_implementation="eager").cpu().eval()
    if model.config.num_hidden_layers==12 and model.config.hidden_size==320:
        raise AssertionError("FP32 reference unexpectedly replaced by 36M research model")
    nparams=sum(v.numel() for v in model.parameters())
    if not 100_000_000<nparams<150_000_000:
        raise AssertionError("Wrong canonical FP32 reference model")
    if any(v.dtype!=torch.float32 for v in model.parameters()):
        raise AssertionError("FP32 baseline contains non-FP32 weights")
    cases=make_cases()
    sha=hashlib.sha256(json.dumps([cases,OPEN_QUALITY],
                                   sort_keys=True).encode()).hexdigest()
    rows=[]
    for i,case in enumerate(cases):
        gold=tok.encode(case["gold"],add_special_tokens=False)+[tok.eos_token_id]
        answer=generate(model,{},tok,case["question"],ANSWER_CAP)
        actual=tok.decode(answer,skip_special_tokens=True)
        isvalid=bool(re.fullmatch(
          r"\s*(Yes|No)\s*[,;:\-]?\s*(red|blue|green|yellow)\s*[.!]?\s*",
          actual,re.I))
        flag,color=parse_two_slots(actual)
        rows.append({
            **case,
            "expected_ids_plus_EOS":gold,
            "actual_ids":answer,
            "actual_text":actual,
            "EOS_emitted":bool(answer) and answer[-1]==tok.eos_token_id,
            "strict_full_EOS":answer==gold,
            "valid_two_slot_format":isvalid,
            "flag_correct":flag==case["expected_flag"],
            "color_correct":color==case["expected_color"],
            "both_slots_valid_and_correct":(
                 isvalid and flag==case["expected_flag"]
                 and color==case["expected_color"]),
        })
        if (i+1)%48==0:
            print(json.dumps({"event":"S16_135M_FP32_REFERENCE_MULTISLOT_PROGRESS",
              "checked":i+1,
              "strict":sum(x["strict_full_EOS"] for x in rows),
              "two_semantic":sum(x["both_slots_valid_and_correct"] for x in rows)
            }),flush=True)
    free=[]
    for q in OPEN_QUALITY:
        a=generate(model,{},tok,q,OPEN_CAP)
        free.append({"question":q,
          "actual_tokens":a,
          "actual_answer":tok.decode(a,skip_special_tokens=True),
          "EOS_emitted":bool(a) and a[-1]==tok.eos_token_id,
          "requires_manual_human_coherence_evaluation":True})
    scores=compare_rows(rows)
    result={
      "stage":"S16_CANONICAL_135M_PRETRAINED_FP32_REFERENCE_READONLY_ON_S15",
      "model":MODEL,"model_frozen_revision_sha":MODEL_SHA,
      "original_FP32_unique_params":nparams,
      "all_parameters_FP32":True,
      "identical_S15_new_192_and_16_open_qa_suite_SHA256":sha,
      "questions":len(rows),"open_questions":len(free),
      "comparison_not_equal_architecture_or_size":True,
      "not_36M_model_FP32_same_architecture":True,
      "only_evaluation_no_training_teacher_loss_or_logits_transfer":True,
      "no_pretrained_FP32_weights_exported":True,
      "CALIBRI_from_scratch_training_untouched":True,
      **scores,
      "all_192_real_generated_answers":rows,
      "all_16_real_compositional_open_answers":free,
      "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    save(args.output/"ACTUAL_CALIBRI2_S16_ORIGINAL_FP32_REFERENCE_192_AND_16.json",result)
    print(json.dumps({k:v for k,v in result.items()
        if k not in ("all_192_real_generated_answers",
                     "all_16_real_compositional_open_answers",
                     "all_96_counterfactual_twin_comparisons",
                     "most_common_mismatches")}),flush=True)


if __name__=="__main__":
    main()
