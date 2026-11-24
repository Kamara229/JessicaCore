"""Z102 precommit a matched objective-only EOS exposure ablation.
Keep 50/50 short/replay group weighting and all full vocabulary logits.
Only supervised terminal-position aggregate weight changes; no generation mask.
Physical training must be separately authorized by completed research outcome.
"""
import json
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119a_hard_four_goals import check
from research.calibri_nra3.s119q2i8z88_short_word_train import prepare,NATIVE_SHA,FRESH_SHA
EOS=2
TERMINAL_SHARE=146/1822
ARMS=("BASE_GROUP_MEAN_CE","MATCHED_TERMINAL_SHARE_CE")
def weights(gold,arm):
    check(arm in ARMS and gold.ndim==1 and len(gold)>1,"valid matched arm and targets")
    terminal=gold==EOS
    n=int(terminal.sum());m=len(gold)-n
    check(n>0 and m>0,"both terminal and lexical positions represented")
    if arm==ARMS[0]:return torch.full((len(gold),),1/len(gold),device=gold.device,dtype=torch.float64)
    return torch.where(terminal,TERMINAL_SHARE/n,(1-TERMINAL_SHARE)/m).double().to(gold.device)
def objective(logits,gold,arm):
    check(logits.ndim==2 and logits.shape[0]==len(gold) and logits.shape[1]==49152,"full original native vocabulary")
    if arm==ARMS[0]:return F.cross_entropy(logits,gold)
    per=F.cross_entropy(logits,gold,reduction="none")
    return (per*weights(gold,arm).to(per.dtype)).sum()
def plan(tok):
    rows,_,_,_=prepare(tok)
    short=[r for r in rows if r["id"].startswith("S119Q2I8Z87_")]
    old=[r for r in rows if not r["id"].startswith("S119Q2I8Z87_")]
    check(len(short)==48 and len(old)==98,"exact Z97/Z100 rows")
    def count(rs):
        ids=[g for r in rs for g in r["answer_native_with_eos"]]
        check(all(r["answer_native_with_eos"][-1]==EOS and r["answer_native_with_eos"].count(EOS)==1 for r in rs),"exact single terminal per answer")
        return {"rows":len(rs),"positions":len(ids),"EOS":ids.count(EOS),"natural_share":ids.count(EOS)/len(ids)}
    s=count(short);groups=[count(old[i::14]) for i in range(14)]
    check(s["positions"]==204 and sum(g["positions"] for g in groups)==1618,"exact native positions")
    base_share=.5*s["natural_share"]+.5*sum(g["natural_share"] for g in groups)/14
    check(base_share>TERMINAL_SHARE,"actual prior overweighting under group CE")
    return {"stage":"Z102_TERMINAL_BALANCE_PREREG","arms":ARMS,"native_SHA256":NATIVE_SHA,"curriculum_SHA256":FRESH_SHA,
        "short":s,"replay_groups":groups,"baseline_mean_terminal_loss_weight":base_share,
        "treatment_terminal_loss_weight":TERMINAL_SHARE,"both_groups_loss_weights":[.5,.5],
        "same_data_and_optimizer_required":True,"hypothesis_not_causal_result":True,
        "no_model_forwards":True,"no_model_training":True,"no_FP32":True,"no_DEV_scoring":True,
        "no_inference_mask":True,"output_gate":0}
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    print("Z102_PREREG "+json.dumps(plan(AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)),sort_keys=True),flush=True)
