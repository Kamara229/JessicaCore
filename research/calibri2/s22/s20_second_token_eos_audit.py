"""CALIBRI2 S22: isolate why S20 reduced early EOS but gained 0/96 new full two-word answers.

S20 physical 36M training showed 0→8 correct TRAIN two-word answers but
remained 0/96 on frozen unseen two-word bindings. S22 tests continuation.
S22 is a separate READ-ONLY causal *diagnostic* (not proof of cause):
given frozen S20 TRAIN-only color+noun paired examples, test:
  (A) full free greedy answer;
  (B) next-token prediction after the CORRECT GOLD first token is supplied;
  (C) EOS prediction after all correct GOLD answer tokens are supplied.
Compare physical S7 vs sealed S20 at *identical TRAIN inputs*.

If forced continuation fails as well as free continuation, the model
does not merely have an incorrect first-token search path; its learned
multiword continuation remains weak. If forced next token works but
free continuation fails, inspect premature-EOS or incorrect first token.
No original 135M FP32, validation, independent QA, TRAIN optimizer or
production edits, no inferred correct response claims from CE alone.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_SHA,SOURCE_BYTES,SOURCE_PARAMS,MODEL_SPEC
from calibri2.s20.multiword_curriculum import make_training_pairs
from calibri2.s3.fresh_heldout import generate
from v174.run import save

S20_SHA="4b62bcec5e40c2b7b73d3be38e4c365d1cf9a62614420497490795534477e93d"
TAKE=16


def load(path,expected):
    blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=expected or len(blob)!=SOURCE_BYTES:
        raise AssertionError("Expected SHA-verified physical 36M CB59")
    b=CB59(blob)
    if b.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("S22 unexpected original 36M architecture")
    if 8*len(blob)/SOURCE_PARAMS>.7:
        raise AssertionError("S22 source exceeded actual packed 0.7 bpp")
    model,sources=create_small_paid_model(b,build_config(MODEL_SPEC))
    return model.eval(),sources


@torch.inference_mode()
def conditional_next(model,sources,prefix,gold_prefix,target,eos):
    context=torch.cat((prefix,gold_prefix)).view(1,-1)
    clear(sources)
    logits=model(input_ids=context,use_cache=False).logits[0,-1]
    top=int(torch.argmax(logits))
    target_logit=float(logits[int(target)])
    eos_logit=float(logits[int(eos)])
    rank=int(torch.count_nonzero(logits>logits[int(target)]))+1
    clear(sources)
    return {"top_id":top,
            "target_id":int(target),
            "target_top1":top==int(target),
            "target_rank":rank,
            "eos_id":int(eos),
            "eos_top1":top==int(eos),
            "eos_logit_minus_gold":eos_logit-target_logit}


def evaluate(path,sha,pairs,tok,label):
    model,sources=load(path,sha)
    eos=int(tok.eos_token_id)
    evidence=[]
    for pair in pairs:
        for side in ("left","right"):
            sample=pair[side]
            prefix=sample["prompt_ids"]
            gold=tok.encode(sample["answer"],add_special_tokens=False)
            if len(gold)<2:
                raise AssertionError("S22 must evaluate at least TWO gold answer tokens")
            first=conditional_next(model,sources,prefix,
                                   torch.empty(0,dtype=torch.long),gold[0],eos)
            second=conditional_next(model,sources,prefix,
                                    torch.tensor(gold[:1],dtype=torch.long),
                                    gold[1],eos)
            end=conditional_next(model,sources,prefix,
                                    torch.tensor(gold,dtype=torch.long),
                                    eos,eos)
            free=generate(model,sources,tok,sample["question"],12)
            evidence.append({
              "pair":pair["pair_id"],"side":side,
              "question":sample["question"],
              "gold_answer":sample["answer"],
              "gold_ids_plus_EOS":gold+[eos],
              "free_greedy_ids":free,
              "free_greedy_answer":tok.decode(free,skip_special_tokens=True),
              "free_exact_full_plus_EOS":free==gold+[eos],
              "free_first_gold_token_correct":bool(free) and free[0]==gold[0],
              "free_one_answer_token_then_EOS":len(free)==2 and free[-1]==eos,
              "forced_first_token":first,
              "forced_second_token":second,
              "forced_final_EOS":end,
            })
    return {
      "model_label":label,"model_sha":sha,
      "records":evidence,
      "example_count":len(evidence),
      "free_full_two_words":sum(x["free_exact_full_plus_EOS"] for x in evidence),
      "free_first_token_correct":sum(x["free_first_gold_token_correct"] for x in evidence),
      "free_single_token_then_EOS":sum(x["free_one_answer_token_then_EOS"] for x in evidence),
      "teacher_forced_second_gold_token_top1":sum(x["forced_second_token"]["target_top1"] for x in evidence),
      "teacher_forced_second_EOS_top1":sum(x["forced_second_token"]["eos_top1"] for x in evidence),
      "teacher_forced_final_correct_EOS_top1":sum(x["forced_final_EOS"]["target_top1"] for x in evidence),
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("s7",type=Path)
    p.add_argument("s20",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    available=make_training_pairs(tok)
    pairs=available[:TAKE//2]+available[-TAKE//2:]
    if len(pairs)!=TAKE:
        raise AssertionError("Incorrect fixed old TRAIN 2word source sample count")
    before=evaluate(args.s7,SOURCE_SHA,pairs,tok,"S7_READONLY")
    after=evaluate(args.s20,S20_SHA,pairs,tok,"S20_READONLY")
    alignment={}
    for a,b in zip(before["records"],after["records"]):
        if (a["question"]!=b["question"] or a["gold_ids_plus_EOS"]!=b["gold_ids_plus_EOS"]):
            raise AssertionError("Conditional old TRAIN question/answer changed")
    report={
      "stage":"S22_READONLY_EOS_AFTER_FIRST_WORD_DIAGNOSTIC",
      "SOURCE_S7_SHA":SOURCE_SHA,
      "SOURCE_S20_SHA":S20_SHA,
      "source_all_physical_bytes":SOURCE_BYTES,
      "source_physical_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
      "historical_TRAIN_only":True,
      "total_paired_samples":TAKE*2,
      "S7_before":before,
      "S20_after":after,
      "same_gold_first_token_forced_diagnostic":True,
      "no_135m_FP32_teacher_or_comparison":True,
      "no_QA_VALID_TEST_used":True,
      "no_training_or_model_mutations":True,
      "all_answers_actual_greedy_or_logged_teacher_conditionals":True,
    }
    save(args.output/"ACTUAL_CALIBRI2_S22_TRAIN_2WORD_EOS_DIAGNOSTIC.json",report)
    print(json.dumps({
      "event":"S22_SECOND_TOKEN_EOS_CAUSE_READONLY_FINISHED",
      "num_samples":TAKE*2,
      "S7":{
        k:v for k,v in before.items()
        if k not in ("records","model_sha","model_label")
      },
      "S20":{
        k:v for k,v in after.items()
        if k not in ("records","model_sha","model_label")
      },
      "first_8_actual_response_examples":[{
         "question":a["question"],
         "gold":a["gold_answer"],
         "S7_actual":a["free_greedy_answer"],
         "S20_actual":b["free_greedy_answer"],
         "S7_after_FORCED_first_word":a["forced_second_token"],
         "S20_after_FORCED_first_word":b["forced_second_token"],
       } for a,b in list(zip(before["records"],after["records"]))[:8]],
    }),flush=True)


if __name__=="__main__":
    main()
