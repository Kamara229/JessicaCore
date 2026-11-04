"""Correct misleading 'first gold token' metric across variable tokenizers."""
import re
import torch
from .protocol import TOP_K

def content(tok,ids):
    k=0
    while k<len(ids) and not tok.decode([ids[k]],skip_special_tokens=True).strip():
        k+=1
    if k>=len(ids):raise ValueError("reference lacks a real lexical subtoken")
    return ids[:k],ids[k]

def choices(tok,gold):
    raw=tok.encode(gold,add_special_tokens=False)
    spaced=tok.encode(" "+gold,add_special_tokens=False)
    if not raw or not spaced:raise ValueError("nonempty answer required")
    return [(seq,)+content(tok,seq) for seq in (raw,spaced)]

def rank(logits,idx):
    return 1+int((logits>logits[idx]).sum().item())

@torch.inference_mode()
def calculate(model,tok,original_input,gold):
    parsed=choices(tok,gold)
    # Historical bug preserved AS A FLAGGED OLD SCORE, not used for PASS.
    initial=model(input_ids=original_input,use_cache=False).logits[0,-1].float()
    legacy_best=min(rank(initial,seq[0]) for seq,prefix,token in parsed)
    # Answer-bearing token should itself represent actual text, never ' '.
    lexical_at_raw=min(rank(initial,lexical) for seq,prefix,lexical in parsed)
    # Teacher-forced whitespace prefix only: this is NOT unprompted inference.
    conditional=[]
    for seq,sprefix,lexical in parsed:
        x=torch.tensor([original_input[0].tolist()+sprefix])
        logits=(initial if not sprefix else
                model(input_ids=x,use_cache=False).logits[0,-1].float())
        conditional.append(rank(logits,lexical))
    best_conditional=min(conditional)
    nonsemantic_prefix_any=any(bool(sprefix) for seq,sprefix,lexical in parsed)
    return {
        "historical_LEADING_space_possible_first_gold_rank_INVALID_for_semantics":legacy_best,
        "first_ACTUAL_LEXICAL_token_rank_unconditional_PROXY":lexical_at_raw,
        "first_LEXICAL_token_rank_after_FORCED_whitespace_conditioning":best_conditional,
        "some_gold_encoded_as_leading_whitespace_ONLY_token":nonsemantic_prefix_any,
        "candidate_first_word_token_ids":[lexical for seq,sprefix,lexical in parsed],
        "candidate_leading_nonsemantic_tokens":[list(pre) for seq,pre,lexical in parsed],
        "old_wrong_target_IDs":[seq[0] for seq,pre,lexical in parsed],
        "corrected_lexical_target_IDs":[lexical for seq,pre,lexical in parsed],
        "historical_old_topK_inflated":{
            str(k):legacy_best<=k for k in TOP_K},
        "lexical_proxy_new_topK":{
            str(k):lexical_at_raw<=k for k in TOP_K},
        "lexical_teacher_forced_prefix_topK":{
            str(k):best_conditional<=k for k in TOP_K},
        "lexical_diagnostic_is_NOT_whole_correct_answer":True}
