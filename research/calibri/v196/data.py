"""Green Tunnel: public TRAIN only, full 49k output-token teacher references.

The mechanism optimizes the ENTIRE output vocabulary at each chosen
TRAIN token position. This does NOT literally mean every language word
is present in the small text sample. It provides nonselected words
with gradients via the teacher's full distribution.
"""
import hashlib
import json
import torch

from v176.data import prepare
from v186.pairs import build_pairs
from v169.sources import all_cases
from v193.questions import cases as new_dev_cases
from . import protocol as P


def training_inputs(tokenizer):
    samples,origin=prepare(tokenizer)
    if len(samples)!=64:raise AssertionError("Public corpus changed")
    train=[samples[i] for i in P.PUBLIC_TRAIN_INDICES]
    forbidden={x["question"] for x in all_cases()}
    forbidden|={x["question"] for x in new_dev_cases()}
    for row in train:
        if row.get("question") in forbidden:
            raise AssertionError("Green Tunnel leaked public QA answers to TRAIN")
    valid=build_pairs(tokenizer,"VALID",forbidden)
    train_pairs=build_pairs(tokenizer,"TRAIN",forbidden)
    if len(train_pairs)!=P.TRAIN_PAIRS or len(valid)!=P.VALID_PAIRS:
        raise AssertionError("Train/held-out VALID pair drift")
    train_questions={x.get("question") for x in train if x.get("question")}
    if train_questions & {p[side]["question"] for p in valid for side in ("left","right")}:
        raise AssertionError("VALID overlap with public Green Tunnel instruction inputs")
    signed=[]
    for index,row in zip(P.PUBLIC_TRAIN_INDICES,train):
        ids=row["ids"]
        signed.append({"original_public_corpus_index":index,"kind":row["kind"],
             "n_input_tokens":len(ids),
             "complete_input_sha256":hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
             "answer":row.get("answer") if row["kind"]=="supervised_chat" else None,
             "question":row.get("question") if row["kind"]=="supervised_chat" else None})
    return train,train_pairs,valid,{"public_source":origin,
        "selected_training_records":signed,"actual_total_samples":len(train),
        "full_original_source_teacher_NOT_loaded_for_input_selection":True,
        "disjoint_prior_public_DEV_and_VALID":True}


def positions_and_targets(sample):
    ids=sample["ids"]
    if len(ids)<4:raise AssertionError("Invalid selected public text")
    if sample["answer_start"] is not None:
        start=sample["answer_start"]-1
        positions=torch.arange(start,len(ids)-1,dtype=torch.long)
        if len(positions)>12:positions=positions[:12]
    else:
        positions=torch.tensor(sorted(set(
            min(len(ids)-2,k) for k in (8,24,40,55)
        )),dtype=torch.long)
    if not len(positions) or positions.max().item()+1>=len(ids):
        raise AssertionError("No valid next-token training positions")
    targets=ids[positions+1].clone().detach()
    if sample["answer_start"] is not None:
        expected=ids[sample["answer_start"]:]
        if targets[0]!=expected[0]:
            raise AssertionError("Supervised full-answer TRAIN target mismatch")
    return positions,targets


@torch.no_grad()
def full_vocabulary_references(original_fp32_teacher,samples,temperature):
    teacher=original_fp32_teacher.cpu().eval()
    frozen=[]
    for i,sample in enumerate(samples):
        ids=sample["ids"].detach().cpu().clone()[None]
        positions,targets=positions_and_targets(sample)
        logits=teacher(input_ids=ids,use_cache=False).logits[0,positions,:]
        if logits.ndim!=2 or logits.shape[1]!=P.VOCAB_SIZE or not torch.isfinite(logits).all():
            raise AssertionError("Original 135M teacher full vocabulary wrong")
        distribution=torch.softmax(logits.float()/temperature,dim=-1).to(torch.float16)
        if distribution.shape!=(len(positions),P.VOCAB_SIZE):
            raise AssertionError("Dropped original vocabulary words")
        frozen.append({"ids":ids,
            "positions":positions,
            "gold_actual_NEXT_tokens":targets,
            "full_49152_token_teacher_probs_fp16":distribution.cpu().clone(),
            "kind":sample["kind"],"teacher_full_VOCAB":P.VOCAB_SIZE,
            "public_source_index":P.PUBLIC_TRAIN_INDICES[i]})
        if (i+1)%5==0:
            print(json.dumps({"stage":"green_tunnel_full_vocab_teacher_TRAIN_only",
                "records":i+1,"full_vocab":P.VOCAB_SIZE}),flush=True)
    if len(frozen)!=len(P.PUBLIC_TRAIN_INDICES):
        raise AssertionError("Some pretrained teacher training positions missing")
    return frozen
