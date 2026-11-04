"""Behavioral generation quality and REAL paired full-model autoregressive CPU timings.

Distinct token ratio is a repetition alarm, not a proof of semantic quality.
Keep exact returned text for independent human review; no model evaluator bias.
"""
from __future__ import annotations
import hashlib,time
import torch
from v90.diagnostics import greedy_summary

NEW_PROMPTS=(
    ("arithmetic","A box contains 3 red balls and 2 blue balls. How many balls are there? Answer:"),
    ("arithmetic","What is 7 plus 5? Give just the answer:"),
    ("arithmetic","There are 12 apples. Four were eaten. Apples left:"),
    ("arithmetic","What is 8 times 3? Answer:"),
    ("factual","Question: Which planet is known as the red planet? Answer:"),
    ("factual","Question: What is the chemical formula for water? Answer:"),
    ("factual","Question: What is the capital of Japan? Answer:"),
    ("factual","Question: What do bees collect from flowers? Answer:"),
    ("reasoning","All birds in this room can fly. A sparrow is in this room. Therefore:"),
    ("reasoning","A train travels 60 km in one hour. In two hours at the same speed it travels:"),
    ("reasoning","If Monday is before Tuesday and Tuesday before Wednesday, Monday is:"),
    ("reasoning","A metal key sinks in water. Explain one physical reason:"),
    ("instruction","Write a short instruction for washing hands safely:"),
    ("instruction","Give two ways to save electricity at home:"),
    ("creative","Write two sentences about a friendly robot and a lost dog:"),
    ("creative","Finish a short story: The little fox found a mysterious blue box."),
)
GENERATION_TOKENS=32

@torch.inference_mode()
def generate_one(model,tok,prompt,*,stop_at_eos=False):
    ids=tok.encode(prompt,add_special_tokens=False)
    if not 1<len(ids)<160:raise AssertionError("bad original TRAIN-only diagnostic prompt")
    eos=tok.eos_token_id
    if stop_at_eos and eos is None:
        raise ValueError("chat generation requires the tokenizer's end-of-turn ID")
    x=torch.tensor(ids,dtype=torch.long)[None,:]
    start=time.perf_counter()
    out=model(input_ids=x,use_cache=True)
    cache=out.past_key_values
    token=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
    emitted=[int(token.item())]
    first_token_time=time.perf_counter()
    for _ in range(GENERATION_TOKENS-1):
        if stop_at_eos and emitted[-1]==eos:
            break
        out=model(input_ids=token,past_key_values=cache,use_cache=True)
        cache=out.past_key_values
        token=out.logits[:,-1,:].argmax(dim=-1,keepdim=True)
        emitted.append(int(token.item()))
    finish=time.perf_counter()
    return {"ids":emitted,"text":tok.decode(emitted,skip_special_tokens=True),
            "elapsed_seconds":finish-start,"prefill_seconds":first_token_time-start,
            "decode_seconds":finish-first_token_time,
            "prompt_SHA":hashlib.sha256(prompt.encode()).hexdigest(),
            "full_model_token_steps":len(emitted),
            "eos_generated":eos in emitted if eos is not None else False}

def paired_greedy(student,tok,parent,leaf,original,new):
    baseline=[];candidate=[]
    try:
        for i,(category,prompt) in enumerate(NEW_PROMPTS):
            # Alternate timing order to reduce build/cache/CPU thermal bias.
            order=("baseline","candidate") if i%2==0 else ("candidate","baseline")
            scored={}
            for kind in order:
                setattr(parent,leaf,original if kind=="baseline" else new)
                scored[kind]=generate_one(student,tok,prompt)
            for group,name in ((baseline,"baseline"),(candidate,"candidate")):
                record=scored[name]
                group.append({"category":category,"prompt_SHA":record["prompt_SHA"],
                              "generated_ids":record["ids"],"generated_text":record["text"],
                              "elapsed_seconds":record["elapsed_seconds"],
                              "eos_generated":record["eos_generated"]})
    finally:setattr(parent,leaf,original)
    if any(a["prompt_SHA"]!=b["prompt_SHA"] for a,b in zip(baseline,candidate)):
        raise AssertionError("not matched generation prompts")
    def summarize(rows):
        summ=greedy_summary([x["generated_ids"] for x in rows])
        summ["examples"]=[{**summary,**record} for summary,record in zip(summ["prompts"],rows)]
        del summ["prompts"]
        summ["seconds_total"]=sum(x["elapsed_seconds"] for x in rows)
        summ["tokens_generated_total"]=GENERATION_TOKENS*len(rows)
        summ["tokens_per_second"]=summ["tokens_generated_total"]/max(summ["seconds_total"],1e-9)
        summ["category_mean_distinct"]={
          category:sum(x["distinct_ratio"] for x in summ["examples"] if x["category"]==category)/
          sum(x["category"]==category for x in summ["examples"])
          for category in sorted(set(x["category"] for x in summ["examples"]))}
        summ["text_quality_human_review_required"]=True
        return summ
    a,b=summarize(baseline),summarize(candidate)
    return {"original":a,"candidate":b,
            "paired_full_model_greedy_decode_speed_ratio":
                b["tokens_per_second"]/max(a["tokens_per_second"],1e-9),
            "same_original_PROMPTS":True,"genuine_KV_cache":True,
            "full_model_autoregressive_including_prefill":True,
            "not_proof_of_semantic_quality":True}
