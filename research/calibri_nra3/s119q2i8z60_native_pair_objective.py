"""Z60 tokenizer-only prereg and differentiable GPT paired native-token loss.
No model weights, FP32 predictions, DEV labels or training runs.
Full sequence CE remains primary; optional matched contrast at first
non-EOS divergence under the same reference-answer prefix.
"""
import json
from collections import defaultdict,Counter
import torch
from torch.nn import functional as F
from research.calibri_nra3 import s119q2i8z59_gpt_compositional_curriculum as z59
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import digest
VOCAB=49152
EOS=2
PAIR_WEIGHT=.10
OBJECTIVES=("FULL_CE","FULL_CE_PLUS_GPT_PAIR")
def require(condition,msg):
    if not condition:raise ValueError(msg)
def pair_targets(a,b,*,vocab=VOCAB,eos=EOS):
    require(len(a)>=2 and len(b)>=2 and a[-1]==b[-1]==eos and
            eos not in a[:-1] and eos not in b[:-1] and
            all(type(x) is int and 0<=x<vocab for x in a+b),"full native targets with terminal EOS")
    for i,(x,y) in enumerate(zip(a,b)):
        if x!=y:
            if eos in (x,y):return None
            return (i,x,y)
    raise ValueError("different authored pair answers must have distinct token sequences")
def token_loss(logits,gold,position,*,objective,contrast=None,vocab=VOCAB,eos=EOS):
    require(objective in OBJECTIVES and logits.shape==(1,vocab) and
            bool(torch.isfinite(logits).all()) and
            0<=position<len(gold) and gold[-1]==eos,
            "full native loss, declared objective and causal position")
    target=gold[position]
    ce=F.cross_entropy(logits,torch.tensor([target],device=logits.device))/len(gold)
    extra=logits.new_zeros(())
    if contrast is not None:
        at,correct,other=contrast
        require(0<=at<len(gold)-1 and gold[at]==correct and
                0<=other<vocab and correct!=other and eos not in (correct,other),
                "contrast only at valid non-EOS divergence")
        if objective==OBJECTIVES[1] and position==at:
            extra=PAIR_WEIGHT*F.softplus(logits[0,other]-logits[0,correct])
    return ce+extra,ce,extra
def native_plan(tok):
    from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer
    manifest=z59.verify()
    raw=z59.curriculum()
    require(tok.vocab_size==VOCAB and tok.eos_token_id==EOS,"same native tokenizer")
    encoded={r["id"]:verified_native_answer(tok,r) for r in raw}
    pairs=defaultdict(list)
    for r in raw:
        if "pair_id" in r:pairs[r["pair_id"]].append(r)
    require(len(encoded)==98 and len(pairs)==28,"all preregistered train examples")
    contrasts={};excluded=[];positions=Counter()
    for k,p in sorted(pairs.items()):
        require(len(p)==2 and all(r["split"]=="TRAIN" for r in p),"two TRAIN members")
        a,b=(encoded[r["id"]]["answer_native_with_eos"] for r in p)
        d=pair_targets(a,b)
        if d is None:
            excluded.append(k)
            continue
        i,x,y=d
        require(a[:i]==b[:i],"identical preceding answer tokens")
        contrasts[p[0]["id"]]=(i,x,y)
        contrasts[p[1]["id"]]=(i,y,x)
        positions[i+1]+=1
    require(len(contrasts)>=40 and len(contrasts)+2*len(excluded)==56,
            "at least 20 valid native contrast pairs, report all exclusions")
    plan=[]
    for r in raw:
        enc=encoded[r["id"]]
        require(len(enc["prefix"])+len(enc["answer_native_with_eos"])<160,"native length cap")
        plan.append({"id":r["id"],"kind":r["kind"],
                     "prefix":enc["prefix"],"gold":enc["answer_native_with_eos"],
                     "contrast":contrasts.get(r["id"])})
    return {"stage":"Z60_GPT_NATIVE_PAIR_OBJECTIVE_PREREG",
            "curriculum_SHA256":manifest["curriculum_SHA256"],
            "native_plan_SHA256":digest(plan),"train_rows":98,
            "native_target_positions":sum(len(r["gold"]) for r in plan),
            "contrast_pairs":len(contrasts)//2,"contrast_rows":len(contrasts),
            "excluded_EOS_first_difference_pairs":excluded,
            "native_divergence_position_counts":dict(positions),
            "objectives":OBJECTIVES,"pair_weight":PAIR_WEIGHT,
            "new_parameters":0,"extra_model_forwards":0,
            "FP32_model_calls":0,"actual_model_forwards":0,"optimizer_steps":0,
            "candidate_accepted":False,"output_gate":0}
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    print("Z60_NATIVE_PLAN",json.dumps(native_plan(tok),ensure_ascii=False,sort_keys=True),flush=True)
