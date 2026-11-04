"""True fixed-answer-label conditional likelihood: not a lexical proxy.

For each prompt, score the SAME next-token label set under all models, never
generate free text or force a correct answer into the prompt. Some tokenizers
may not have a single token for every label: fail closed, do not fake scores.
"""
import torch
from torch.nn.functional import log_softmax,softmax

from v108.generation import build_chat
from .protocol import ARMS,DOMAINS,COUNT


def label_tokens(tokenizer,labels):
    ids=[]
    for option in labels:
        encoded=tokenizer.encode(" "+option,add_special_tokens=False)
        if len(encoded)!=1:
            raise ValueError(f"label {option} is not a single tokenizer token")
        ids.append(int(encoded[0]))
    if len(set(ids))!=len(ids):raise ValueError("answer labels share token id")
    return tuple(ids)


def encode_prompt(tokenizer,question):
    text=build_chat(tokenizer,question)
    ids=tokenizer.encode(text,add_special_tokens=False)
    if not 4<len(ids)<1024:raise ValueError("QA prompt context too long or empty")
    return torch.tensor([ids],dtype=torch.long)


@torch.inference_mode()
def measure_one(model,ids,answer_token_ids,gold):
    output=model(input_ids=ids,use_cache=False).logits
    if output.ndim!=3 or output.shape[0]!=1 or output.shape[1]!=ids.shape[1]:
        raise ValueError("incompatible last-token logits")
    last=output[0,-1].to(torch.float64)
    if not torch.isfinite(last).all():raise ValueError("non-finite answer logits")
    candidates=last[list(answer_token_ids)]
    normalized=softmax(candidates,dim=-1)
    prediction=int(candidates.argmax())
    other=candidates.clone()
    other[gold]=-torch.inf
    return {
        "selected_index":prediction,"is_correct":prediction==gold,
        "confidence_of_correct_label_conditional":float(normalized[gold]),
        "gold_margin_logit_vs_other":float(candidates[gold]-other.max()),
        "answer_label_distribution":[float(p) for p in normalized],
    }


def aggregate(records):
    if len(records)!=COUNT:raise ValueError("missing fixed number of QA tasks")
    accuracy=sum(r["is_correct"] for r in records)/len(records)
    return {
        "questions":len(records),
        "correct":int(sum(r["is_correct"] for r in records)),
        "accuracy":accuracy,
        "mean_gold_label_conditional_probability":
            sum(r["confidence_of_correct_label_conditional"] for r in records)/len(records),
        "mean_gold_choice_margin":sum(r["gold_margin_logit_vs_other"] for r in records)/len(records),
    }


def summary(results):
    if tuple(results)!=ARMS:raise ValueError("mismatched predeclared model arms")
    evaluation={}
    for corpus in DOMAINS:
        if any(len(results[arm][corpus])!=COUNT for arm in ARMS):
            raise ValueError("missing heldout QA questions")
        measures={arm:aggregate(results[arm][corpus]) for arm in ARMS}
        teacher=results["FP32"][corpus]
        for arm in ARMS:
            measures[arm]["teacher_forced_choice_prediction_agreements"]=sum(
                r["selected_index"]==t["selected_index"]
                for r,t in zip(results[arm][corpus],teacher))
        measures["CB140_half_vs_CB59_accuracy_delta"]=(
            measures["CB140_head_half"]["accuracy"]-measures["CB59"]["accuracy"])
        measures["CB140_full_vs_CB59_accuracy_delta"]=(
            measures["CB140_full"]["accuracy"]-measures["CB59"]["accuracy"])
        evaluation[corpus]=measures
    return {
        "validation":evaluation,
        "CB140_half_no_regression_in_both":all(
            evaluation[c]["CB140_half_vs_CB59_accuracy_delta"]>=0 for c in DOMAINS),
        "CB140_full_no_regression_in_both":all(
            evaluation[c]["CB140_full_vs_CB59_accuracy_delta"]>=0 for c in DOMAINS),
        "interpretation":"Fixed next-label likelihood on tiny 8+8 validation only. Not free response accuracy, independent locked v89 TEST, or an adequate statistical QA PASS.",
    }
