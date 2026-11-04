"""Third previously unseen model answers; direct generated false/true labels."""
import re,torch
from .protocol import VALID,MAX_NEW_TOKENS

def text_key(x):
    return re.sub(r"[^\w]+"," ",x).strip().casefold()

@torch.inference_mode()
def generate(model,tok):
    model.eval()
    candidates={}
    for q,gold in VALID:
        ids=tok(q,return_tensors="pt")["input_ids"]
        outputs=model.generate(ids,max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
        continuation=outputs[0,ids.shape[1]:]
        candidates[q]={"answer":tok.decode(continuation,skip_special_tokens=True),
                       "token_ids":continuation.tolist(),"correct_reference":gold}
    return candidates

def controls(data):
    rows=[]
    names=[a for q,a in VALID[:8]]
    for i,(q,truth) in enumerate(VALID):
        if i<8:
            wrong=names[(i+1)%8]
        else:
            wrong=str(int(truth)+1)
        candidate=data[q]["answer"]
        label=int(text_key(candidate)==text_key(truth))
        rows.extend(((q,truth,1,"independent_gold_ORACLE_only"),
                     (q,wrong,0,"plausible_wrong"),
                     (q,candidate,label,"UNSEEN_GPT2_REAL_OUTPUT")))
    if len(rows)!=48:
        raise AssertionError("incorrect unseen host validation grouping")
    return rows
