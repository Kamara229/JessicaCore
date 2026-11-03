"""Original label correctness + meaningful generation diagnostics only.

No approximation-error or teacher-distribution score is used for candidate
ranking. Repeated-token ratio only a guard; human semantic review required.
"""
from __future__ import annotations
from v94.qa import score_cases
from v94.answers import summary
from v94.generation import generate_one
from v95.spec import TRAIN_GENERATION,VALID_GENERATION

def eval_answers(model,datasets,token_ids,split):
    out={}
    for domain in ("BoolQ","SciQ"):
        samples=datasets[domain][split]
        rows=score_cases(model,samples,token_ids[domain])
        out[domain]={"score":summary(rows,len(token_ids[domain])),
                     "items":rows}
    return out

def eval_greedy(model,tok,prompts):
    data=[]
    for category,prompt in prompts:
        one=generate_one(model,tok,prompt)
        ids=one["ids"]
        most=max(ids.count(token) for token in set(ids))/len(ids)
        data.append({"category":category,"prompt_SHA":one["prompt_SHA"],
                     "token_ids":ids,"decoded_response":one["text"],
                     "distinct_ratio":len(set(ids))/len(ids),
                     "most_frequent_share":most,
                     "elapsed_seconds":one["elapsed_seconds"]})
    return {"prompts":data,"n":len(data),
        "mean_distinct_ratio":sum(x["distinct_ratio"] for x in data)/len(data),
        "mean_most_frequent_share":sum(x["most_frequent_share"] for x in data)/len(data),
        "tokens_per_second":sum(len(x["token_ids"]) for x in data)/
                            max(sum(x["elapsed_seconds"] for x in data),1e-9),
        "human_review_of_semantic_quality_still_required":True,
        "full_CPU_model_including_prompt_prefill_and_greedy_decode":True}

def fit_rank(records):
    # Semantic quality is not yet proven; correctness dominates novelty.
    # Balanced accuracy addresses degenerate all-No BoolQ or majority SciQ.
    def score(one):
        correct=sum(one["QA"][domain]["score"]["correct"] for domain in ("BoolQ","SciQ"))
        balanced=sum(one["QA"][domain]["score"]["balanced_accuracy"]
                     for domain in ("BoolQ","SciQ"))
        novelty=one["greedy"]["mean_distinct_ratio"]
        repetition=one["greedy"]["mean_most_frequent_share"]
        return (correct,round(balanced,6),-round(repetition,6),round(novelty,6))
    ordered=sorted(records,key=lambda k:(score(records[k]),k),reverse=True)
    return ordered,{"score_order":["ground_truth_correct_answers",
            "class_balanced_accuracy","repetition_low","distinct_tokens_high"],
        "score_by_name":{key:score(value) for key,value in records.items()},
        "fixed_on_public_TRAIN_only":True,
        "no_FP32_distribution_or_KL_NLL_in_ranking":True}
