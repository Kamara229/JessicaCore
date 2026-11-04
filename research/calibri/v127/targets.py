"""Content-aware lexical probe; never count a bare whitespace as correct."""
import torch
from v108.generation import build_chat
from v94.generation import generate_one

def prompt_input(tok,q):
    rendered=build_chat(tok,q)
    ids=tok.encode(rendered,add_special_tokens=False)
    return torch.tensor([ids],dtype=torch.long),rendered

def reference(tok,gold):
    """Pre-register all answer tokens + EOS and first actual lexical subtoken."""
    ids=tok.encode(" "+gold,add_special_tokens=False)
    if not ids or tok.eos_token_id is None:
        raise ValueError("reference must contain content and end of assistant turn")
    n=0
    while n<len(ids) and not tok.decode([ids[n]],skip_special_tokens=True).strip():
        n+=1
    if n>=len(ids):
        raise ValueError("answer is whitespace-only, no lexical content to test")
    lexical=ids[n]
    if not tok.decode([lexical],skip_special_tokens=True).strip():
        raise AssertionError("invalid whitespace-only lexical target")
    return {"full_answer_tokens_with_EOS":ids+[tok.eos_token_id],
            "whitespace_prefix_ids":ids[:n],
            "first_NON_WHITESPACE_answer_subtoken":lexical,
            "first_lexical_offset":n,
            "full_content_answer_tokens_without_EOS":ids}

def teacher_forced_logits(model,source,ref):
    goal=ref["full_answer_tokens_with_EOS"]
    original=source[0].tolist()
    seq=torch.tensor([original+goal[:-1]],dtype=torch.long)
    logits=model(input_ids=seq,use_cache=False).logits[
        0,len(original)-1:len(original)-1+len(goal),:].float()
    if logits.shape[0]!=len(goal):
        raise AssertionError("full answer target alignment wrong")
    return logits,torch.tensor(goal,dtype=torch.long)

def score(model,source,ref):
    """Rank of first *lexical content* subtoken, conditioned on fixed whitespace.

    This is a DIAGNOSTIC teacher-forced prefix and can NOT claim the
    model autonomously produced this prefix.
    """
    with torch.no_grad():
        prefix=source[0].tolist()+ref["whitespace_prefix_ids"]
        logits=model(input_ids=torch.tensor([prefix]),use_cache=False).logits[0,-1].float()
        target=ref["first_NON_WHITESPACE_answer_subtoken"]
        rank=1+int((logits>logits[target]).sum().item())
        return {"first_NON_WHITESPACE_subtoken_rank_CONDITIONAL":rank,
                "first_NON_WHITESPACE_subtoken_top1_CONDITIONAL":bool(
                    logits.argmax().item()==target),
                "first_NON_WHITESPACE_subtoken_probability_CONDITIONAL":float(
                    torch.softmax(logits,dim=-1)[target]),
                "requires_UNGENERATED_whitespace_prefix":bool(
                    ref["whitespace_prefix_ids"]),
                "conditioning_prefix_token_ids":ref["whitespace_prefix_ids"],
                "actual_content_target_token_id":target,
                "actual_predicted_token_id":int(logits.argmax())}

def answer(model,tok,prompt):
    with torch.no_grad():
        item=generate_one(model,tok,prompt,stop_at_eos=True)
    return {"text":item["text"][:450],
            "token_ids":item["ids"],
            "reached_eos":bool(item["eos_generated"]),
            "elapsed_seconds":item["elapsed_seconds"],
            "full_decoding_steps":item["full_model_token_steps"]}
