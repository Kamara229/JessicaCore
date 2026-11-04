"""Information-theoretic *headroom* for a 10MB reranker on actual CB59.

Never send human-labeled correct answer to the candidate generator.
ONLY score first-token accessibility of true target (top-K oracle diagnostic).
Reranker cannot recover an absent candidate or corrupt world model.
"""
import torch
from v108.generation import build_chat
from v94.generation import generate_one
from .protocol import VALID_A,VALID_B,OPEN_B,CANDIDATE_TOP_K
from .bytes_adapter import pairs

@torch.inference_mode()
def first_correct_token_rank(model,tok):
    results=[]
    for question,gold in VALID_A+VALID_B:
        tokens=tok.encode(build_chat(tok,question),add_special_tokens=False)
        logits=model(input_ids=torch.tensor([tokens]),use_cache=False).logits[0,-1].float()
        candidates=set()
        for text in (gold," "+gold):
            ids=tok.encode(text,add_special_tokens=False)
            if ids:candidates.add(ids[0])
        if not candidates:
            raise AssertionError("gold validation not representable with host tokenizer")
        ranks=[1+int((logits>logits[id]).sum()) for id in candidates]
        best=min(ranks)
        results.append({"question":question,"true_answer":gold,
                        "correct_FIRST_subtoken_best_rank":best,
                        "contains_TRUE_first_token_in_topK":{
                            str(k):bool(best<=k) for k in CANDIDATE_TOP_K},
                        "oracle_test_only_not_injected":True})
    return results

@torch.inference_mode()
def real_candidates(model,tok,critic):
    actual=[]
    for question,gold in VALID_A[:6]+VALID_B[:6]:
        result=generate_one(model,tok,build_chat(tok,question),stop_at_eos=True)
        actual.append({"question":question,"true_gold_for_GRADING_ONLY":gold,
                       "generated":result["text"],"token_ids":result["ids"],
                       "did_not_inject_gold_into_generator":True})
    for question in OPEN_B:
        result=generate_one(model,tok,build_chat(tok,question),stop_at_eos=True)
        actual.append({"question":question,"true_gold_for_GRADING_ONLY":None,
                       "generated":result["text"],"token_ids":result["ids"],
                       "did_not_inject_gold_into_generator":True})
    critic.eval()
    scores=[]
    for start in range(0,len(actual),16):
        items=actual[start:start+16]
        x=pairs([(row["question"],row["generated"]) for row in items])
        scores.extend(torch.sigmoid(critic(x)).tolist())
    for row,p in zip(actual,scores):
        row["critic_probability"]=float(p)
        row["choose_to_pass_original_answer"]=bool(p>=.5)
        row["critique_fallback"]="decline_or_requery" if p<.5 else "pass_through"
        row["new_answer_was_NOT_invented"]=True
    return actual

def headroom_summary(records):
    return {str(k):{
        "true_first_answer_subtoken_represented":sum(
            r["correct_FIRST_subtoken_best_rank"]<=k for r in records),
        "out_of":len(records)
    } for k in CANDIDATE_TOP_K}
