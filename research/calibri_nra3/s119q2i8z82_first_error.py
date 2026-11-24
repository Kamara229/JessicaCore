"""Read-only first-divergence forensics on completed Z76 TRAIN42.
No decoder or FP32 calls; no evaluation corpus or optimization.
"""
import argparse,json
from collections import Counter
from pathlib import Path
from research.calibri_nra3.s119q2i8z62_utf8_train_forensics import status,classify,COMPLETE,INCOMPLETE
from research.calibri_nra3.s119q2i8z57_fullanswer_eval import TRAIN_SHA,CHILD_SHA,GT_SHA
from research.calibri_nra3.s119q2i8z75_joint_after_head import PINS
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import curriculum,digest
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import verified_native_answer
ARMS=("Z55_PARENT","HEAD_CONTROL","HEAD_REVIVE_ZERO_UNITS")
def require(ok,msg):
    if not ok:raise ValueError(msg)
def prefix_match(a,b):
    i=0
    for x,y in zip(a,b):
        if x!=y:break
        i+=1
    return i
def analyze(tok,ids,gold):
    p=prefix_match(ids,gold)
    out={"prefix_match":p,"first_divergence":p+1 if p<min(len(ids),len(gold)) else None,
         "first_invalid_position":None,"first_nonspace_gold_position":None}
    for i in range(len(gold)):
        s=tok.decode(gold[:i+1],skip_special_tokens=True)
        if s.strip() and "�" not in s:
            out["first_nonspace_gold_position"]=i+1
            break
    if out["first_divergence"] is not None:
        before,after,label=classify(tok,gold[:p],ids[p])
        out.update(first_error_label=label,first_error_prefix_state=before,
                   first_error_gold=gold[p],first_error_chosen=ids[p],
                   first_error_raw=str(tok.convert_ids_to_tokens(ids[p])))
    for i in range(len(ids)):
        if i==0 or status(tok,ids[:i]) in (COMPLETE,INCOMPLETE):
            _,_,label=classify(tok,ids[:i],ids[i])
            if label in ("BROKEN_BYTE_TRANSITION","EOS_ON_INCOMPLETE","NON_EOS_SPECIAL"):
                out["first_invalid_position"]=i+1
                break
        else:break
    return out
def run(tok,report):
    require(report["stage"]=="S119Q2I8Z76_GPT_HEAD_OWNPREFIX_TRAIN42" and report["TRAIN_SHA256"]==TRAIN_SHA
            and report["Z55_joint_SHA256"]==CHILD_SHA and report["GT12_SHA256"]==GT_SHA
            and report["Z73_joint_pins"]==PINS and report["common_native_cap"]==48
            and report["FP32_calls"]==0 and report["optimizer_steps"]==0 and report["no_DEV_scoring"]
            and report["output_gate"]==0 and set(report["per_case"])==set(ARMS),"exact completed Z76 provenance")
    source=curriculum()
    require(digest(source)==TRAIN_SHA and len(source)==42,"exact authored source")
    gold={r["id"]:verified_native_answer(tok,r)["answer_native_with_eos"] for r in source}
    require(sum(map(len,gold.values()))==842,"exact native TRAIN42")
    results={};details={}
    for arm in ARMS:
        rows=report["per_case"][arm]
        require(len(rows)==42 and {r["id"] for r in rows}==set(gold),"complete saved own-prefix rows")
        counts=Counter();first=Counter();bad=Counter();labels=Counter();common=Counter();out=[]
        for r in rows:
            ids=r["generated_ids"];g=gold[r["id"]]
            d=analyze(tok,ids,g)
            require(d["prefix_match"]==r["gold_prefix_match"] and len(ids)==r["generated_count"],"match original score")
            counts["cases"]+=1
            counts["correct_first_token"]+=int(d["prefix_match"]>=1)
            counts["correct_first_two_tokens"]+=int(d["prefix_match"]>=2)
            counts["correct_first_three_tokens"]+=int(d["prefix_match"]>=3)
            counts["reached_first_nonspace_gold"]+=int(d["first_nonspace_gold_position"] is not None and d["prefix_match"]>=d["first_nonspace_gold_position"])
            counts["invalid_at_first_divergence"]+=int(d["first_invalid_position"] is not None and d["first_invalid_position"]==d["first_divergence"])
            counts["invalid_after_first_divergence"]+=int(d["first_invalid_position"] is not None and d["first_divergence"] is not None and d["first_invalid_position"]>d["first_divergence"])
            counts["no_hard_invalid_transition"]+=int(d["first_invalid_position"] is None)
            if d["first_divergence"] is not None:
                first[d["first_divergence"]]+=1;labels[d["first_error_label"]]+=1;common[d["first_error_chosen"]]+=1
            if d["first_invalid_position"] is not None:bad[d["first_invalid_position"]]+=1
            out.append({"id":r["id"],"kind":r["kind"],**d})
        results[arm]={"counts":dict(counts),"first_divergence_histogram":dict(first),"first_invalid_histogram":dict(bad),
            "first_error_transition_labels":dict(labels),"most_common_first_error_tokens":[{"id":i,"raw":str(tok.convert_ids_to_tokens(i)),"count":n} for i,n in common.most_common(10)]}
        details[arm]=out
    return {"stage":"Z82_OWNPREFIX_FIRST_ERROR","metrics":results,"per_case":details,"source_run":45466960721,
            "new_model_forwards":0,"optimizer_steps":0,"FP32_calls":0,"no_DEV_scoring":True,"output_gate":0}
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    p=argparse.ArgumentParser();p.add_argument("--report",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    result=run(tok,json.loads(a.report.read_text(encoding="utf8")))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("Z82_FIRST_ERROR "+json.dumps({k:v for k,v in result.items() if k!="per_case"},ensure_ascii=False,sort_keys=True),flush=True)
