"""Z107 TRAIN-only same-prefix lexical discrimination, preregistered before training.
Use the original Z103 cache and matched terminal CE in both arms.
No new parameters, tokenizer changes, inference constraints or FP32 teacher.
"""
import json
from itertools import combinations
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z102_terminal_balance import objective, ARMS as CE_ARMS
from research.calibri_nra3.s119q2i8z87_short_word_curriculum import fresh,native,FAMILIES
from research.calibri_nra3.s119q2i8z88_short_word_train import NATIVE_SHA,FRESH_SHA
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import digest
ARMS=("MATCHED_CE","MATCHED_CE_PLUS_CONTEXT")
MARGIN=.5
WEIGHT=.25
EOS=2
def plan_pairs(rows):
    """Return pair indices into packed short logits, using only shared gold prefix.
    Every pair contains two distinct words from the same task family.
    Exclude EOS comparisons: lexical discrimination is not stopping calibration.
    """
    assert len(rows)==48 and len({r["id"] for r in rows})==48
    offsets=[];n=0
    for r in rows:
        offsets.append(n);n+=len(r["gold"])
        assert r["gold"][-1]==EOS and EOS not in r["gold"][:-1]
    out=[]
    for i,j in combinations(range(len(rows)),2):
        a,b=rows[i],rows[j]
        if a["family"]!=b["family"]:continue
        assert a["word"]!=b["word"]
        p=0
        while p<min(len(a["gold"]),len(b["gold"])) and a["gold"][p]==b["gold"][p]:p+=1
        assert p<min(len(a["gold"]),len(b["gold"]))
        x,y=a["gold"][p],b["gold"][p]
        if EOS in (x,y):continue
        out.append((offsets[i]+p,offsets[j]+p,x,y))
    assert n==204 and 0<len(out)<=264
    return out
def pair_penalty(logits,pairs):
    assert logits.shape==(204,49152) and pairs.ndim==2 and pairs.shape[1]==4 and len(pairs)>0
    i,j,x,y=pairs.unbind(1)
    # Both distributions are evaluated under exactly the same preceding answer tokens.
    # Loss requires the right lexical branch for each different prompt.
    a=logits[i,x]-logits[i,y]
    b=logits[j,y]-logits[j,x]
    return .5*(F.softplus(MARGIN-a).mean()+F.softplus(MARGIN-b).mean())
def short_objective(logits,gold,arm,pairs):
    assert arm in ARMS
    ce=objective(logits,gold,CE_ARMS[1])
    if arm==ARMS[0]:return ce
    return ce+WEIGHT*pair_penalty(logits,pairs)
def plan(tok):
    n=native(tok)
    assert n["native_plan_SHA256"]==NATIVE_SHA and n["new_SHA256"]==FRESH_SHA
    rows=[dict(id=r["id"],family=r["family"],word=r["word"],gold=r["target_ids"]) for r in n["per_case"]]
    pairs=plan_pairs(rows)
    return {"stage":"Z107_CONTEXT_DISCRIMINATION_PREREG","source_native_SHA256":NATIVE_SHA,"source_TRAIN_SHA256":FRESH_SHA,
            "pair_plan_SHA256":digest(pairs),"pairs":len(pairs),"same_family_all_distinct_word_pairs":True,"shared_gold_prefix":True,"exclude_EOS":True,
            "arms":ARMS,"margin":MARGIN,"aux_weight_within_short_loss":WEIGHT,"short_replay_loss_weights":[.5,.5],"updates_per_arm":1764,
            "cached_positions_per_arm":563724,"parent":"exact revived Z73","control":"exact Z103 matched CE","new_parameters":0,
            "training_gate":"physical control reproduces Z103 within prior tolerance; no promotion from TRAIN","no_DEV_scoring":True,"no_FP32":True,
            "no_model_forwards":True,"no_inference_mask":True,"output_gate":0}
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    print("Z107_PREREG "+json.dumps(plan(AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)),sort_keys=True),flush=True)
