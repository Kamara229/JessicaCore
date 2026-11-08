"""Independent FP32 PUBLIC PRETRAINED 135M reference, NOT the 36M compressed source.

This contextualizes whether questions on the precommitted S26 QA are
solvable by a pretrained instruction-following model with the same
tokenizer/template. It CANNOT prove 36M compression parity or serve as
a teacher. Frozen S26 QA (SHA pinned below) existed before S26 training.
No training, gradients, alternate checkpoints, model selection, or LIVE.
"""
from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from calibri2.s26.frozen_new_qa import manifest, cases, OPEN_QUESTIONS
from v174.run import save

EXPECTED_QA_SHA="e8e3e75c5fedf845aef512a104723a4d04fab3980ecec7f4deb3575ae020ac3f"
NEW_DOMAIN="reserved_novel_color_noun"
VALID_TWO=re.compile(r"^\s*(red|blue|green|yellow)\s+"
    r"(cup|book|bag|lamp|chair|hat|plate|scarf|clock|spoon|pencil|bottle|fork|glove|shirt|key)\s*[.!]?\s*$",
    re.I)

@torch.inference_mode()
def greedy(model, tok, question, max_new_tokens):
    chat=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    prefix=tok.encode(chat,add_special_tokens=False)
    if not 1<=len(prefix)<160:
        raise AssertionError("Malformed reference chat prompt")
    context=torch.tensor([prefix],dtype=torch.long)
    generated=[]
    for _ in range(max_new_tokens):
        logits=model(input_ids=context,use_cache=False).logits[0,-1]
        if not torch.isfinite(logits).all():
            raise AssertionError("Nonfinite FP32 pretrained output")
        nxt=int(logits.argmax())
        generated.append(nxt)
        if nxt==tok.eos_token_id:break
        context=torch.cat((context,context.new_tensor([[nxt]])),dim=1)
    return generated

def totals(rows):
    stats=defaultdict(Counter)
    common=Counter()
    for row in rows:
        dom=stats[row["domain"]]
        dom["n"]+=1
        dom["strict_full_EOS"]+=int(row["strict_full_EOS"])
        dom["correct_first_token"]+=int(row["correct_first_token"])
        dom["EOS_emitted"]+=int(row["EOS_emitted"])
        if row["domain"]==NEW_DOMAIN:
            m=VALID_TWO.fullmatch(row["actual_text"])
            gold=row["gold"].split()
            correct_c=bool(m and m.group(1).lower()==gold[0])
            correct_n=bool(m and m.group(2).lower()==gold[1])
            dom["valid_two_word"]+=int(bool(m))
            dom["correct_color_with_valid_format"]+=int(correct_c)
            dom["correct_noun_with_valid_format"]+=int(correct_n)
            dom["both_semantic_words_correct"]+=int(correct_c and correct_n)
            common[row["actual_text"].strip()]+=1
    return dict(stats),common.most_common(16)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("output",type=Path)
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    frozen=manifest()
    assert frozen["qa_suite_sha256"]==EXPECTED_QA_SHA
    assert len(cases())==144 and len(OPEN_QUESTIONS)==16
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=MODEL_SHA,torch_dtype=torch.float32,
        attn_implementation="eager").cpu().eval()
    params=sum(p.numel() for p in model.parameters())
    if params!=EXPECTED_UNIQUE or any(p.dtype!=torch.float32 for p in model.parameters()):
        raise AssertionError("Wrong actual real pretrained FP32 135M checkpoint")

    rows=[]
    for idx,case in enumerate(cases()):
        gold=tok.encode(case["gold"],add_special_tokens=False)+[tok.eos_token_id]
        if len(gold)>14:raise AssertionError("Gold answer exceeds reference token cap")
        generated=greedy(model,tok,case["question"],14)
        row={
            **case,
            "expected_token_ids_full_EOS":gold,
            "actual_token_ids":generated,
            "actual_text":tok.decode(generated,skip_special_tokens=True),
            "strict_full_EOS":generated==gold,
            "correct_first_token":bool(generated) and generated[0]==gold[0],
            "EOS_emitted":bool(generated) and generated[-1]==tok.eos_token_id
        }
        rows.append(row)
        if (idx+1)%48==0:
            print(json.dumps({
                "event":"S26_PUBLIC_135M_PRETRAINED_FP32_REFERENCE_PROGRESS",
                "checked":idx+1,
                "strict_full_EOS_correct":sum(x["strict_full_EOS"] for x in rows)
            }),flush=True)
    open_rows=[]
    for q in OPEN_QUESTIONS:
        result=greedy(model,tok,q,32)
        open_rows.append({
            "question":q,
            "actual_answer":tok.decode(result,skip_special_tokens=True),
            "actual_tokens":result,
            "EOS":bool(result) and result[-1]==tok.eos_token_id,
            "human_coherence_review_required":True
        })
    domain,common=totals(rows)
    report={
        "stage":"S26_EXTERNAL_PRETRAINED_135M_FP32_REFERENCE_NOT_MATCHED_36M",
        "checkpoint_model":MODEL,
        "checkpoint_revision":MODEL_SHA,
        "actual_model_params":params,
        "actual_weight_dtype":"float32",
        "S26_precommitted_frozen_qa_SHA256":EXPECTED_QA_SHA,
        "same_tokenizer_and_chat_template_as_compressed_36M":True,
        "same_architecture_or_trained_initial_weights_as_36M":False,
        "is_not_an_FP32_teacher_for_S26":True,
        "not_a_matched_compression_quality_retention_experiment":True,
        "only_precommitted_single_pretrained_checkpoint_evaluated":True,
        "no_QA_used_for_training_or_model_selection":True,
        "total_cases":len(rows),
        "total_correct_full_EOS":sum(x["strict_full_EOS"] for x in rows),
        "per_domain":domain,
        "common_real_answers":common,
        "every_144_actual_answer_and_gold":rows,
        "every_16_open_actual_answer":open_rows
    }
    save(a.output/"ACTUAL_S26_INDEPENDENT_PUBLIC_135M_FP32_REFERENCE.json",report)
    print(json.dumps({key:value for key,value in report.items() if key not in
        ("every_144_actual_answer_and_gold","every_16_open_actual_answer","common_real_answers")}),flush=True)

if __name__=="__main__":
    main()
