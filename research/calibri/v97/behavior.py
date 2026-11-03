"""Original public labeled BoolQ and actual free-language generation.

Grade BoolQ against real public ground truth. Synthetic answer-reference
in first-sentence is ONLY diagnostic evidence, not certified accuracy.
All generated text is preserved for later human semantic review.
"""
from __future__ import annotations
from hashlib import sha256
from v94.qa import source_rows,encode_case,score_cases
from v94.answers import summary
from v94.generation import generate_one
from v96.answer_review import leading_reference_signal
from v96.grade import string_breakdown
from .spec import BOOLQ_RANGE,ANSWER_PROMPTS,FREE_PROMPTS

def original_boolq(tok):
    rows,source=source_rows("google/boolq",BOOLQ_RANGE)
    yes=tok.encode(" Yes",add_special_tokens=False)
    no=tok.encode(" No",add_special_tokens=False)
    if len(yes)!=1 or len(no)!=1:raise AssertionError("actual original tokenizer incompatible")
    cases=[]
    for index,row in enumerate(rows):
        snippet=tok.decode(tok.encode(str(row["passage"]),add_special_tokens=False)[:256],
                           skip_special_tokens=True)
        prompt=f"Context: {snippet}\nQuestion: {row['question']}\nAnswer:"
        cases.append(encode_case(tok,prompt,BOOLQ_RANGE[0]+index,
                                 0 if row["answer"] else 1))
    return cases,source,[yes[0],no[0]]

def evaluate_model(model,tok,cases,answer_ids):
    counted=score_cases(model,cases,answer_ids)
    questions=[]
    for domain,prompt,refs in ANSWER_PROMPTS:
        output=generate_one(model,tok,prompt)
        signal=leading_reference_signal(output["text"],refs)
        questions.append({"category":domain,"prompt_SHA":output["prompt_SHA"],
            "actual_text":output["text"],"raw_ids":output["ids"],
            "first_answer_reference_signal":signal,
            "lexical":string_breakdown(output["text"],output["ids"])})
    free=[]
    for domain,prompt in FREE_PROMPTS:
        output=generate_one(model,tok,prompt)
        free.append({"category":domain,"prompt_SHA":output["prompt_SHA"],
            "actual_text":output["text"],"raw_ids":output["ids"],
            "lexical":string_breakdown(output["text"],output["ids"]),
            "semantic_coherence_human_review_required":True})
    responses=questions+free
    return {"boolq":{"summary":summary(counted,2),"items":counted},
        "fixed_answers":questions,"free_text":free,
        "repeated_over_75pct":sum(x["lexical"]["repeat_over_75pct"] for x in responses),
        "mean_distinct_token_fraction":sum(x["lexical"]["unique_token_fraction"] for x in responses)/len(responses),
        "first_sentence_answer_reference_present":sum(
             x["first_answer_reference_signal"]["reference_present_in_first_answer_sentence"] for x in questions),
        "free_generation_semantic_quality_unverified":True,
        "actual_original_public_TRAIN_ground_truth":True}

def paired_checks(results):
    base=results["CB59_NATIVE_FROZEN"]
    sha=[x["prompt_SHA"] for x in base["fixed_answers"]+base["free_text"]]
    labels=[(x["prompt_SHA"],x["gold"]) for x in base["boolq"]["items"]]
    for name,report in results.items():
        if [x["prompt_SHA"] for x in report["fixed_answers"]+report["free_text"]]!=sha:
            raise AssertionError("unmatched generated prompt "+name)
        if [(x["prompt_SHA"],x["gold"]) for x in report["boolq"]["items"]]!=labels:
            raise AssertionError("different original public task labels "+name)
    return True
