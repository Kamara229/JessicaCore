"""Read-only byte-level analysis of existing Z58 TRAIN42 predictions.
No model loads, new model calls, DEV or FP32 outputs. Valid incomplete
prefixes are not mistakes unless the next token makes them invalid or EOS.
"""
import argparse,json
from pathlib import Path
from collections import Counter
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import native_bytelevel_full_utf8_status
from research.calibri_nra3.s119q2i8z57_fullanswer_eval import TRAIN_SHA,CHILD_SHA,GT_SHA
COMPLETE="VALID_COMPLETE_UTF8"
INCOMPLETE="INCOMPLETE_UTF8_NEEDS_CONTINUATION"
INVALID="INVALID_UTF8_SEQUENCE"
UNKNOWN="UNRECOGNIZED_TOKENIZER_RAW_GLYPHS"
ARMS=("Z10_A_PARENT","Z55_GPT_ONLY")
def require(ok,msg):
    if not ok:raise ValueError(msg)
def status(tok,ids):
    return native_bytelevel_full_utf8_status(tok,ids)["entire_original_native_BPE_prefix_UTF8_status"]
def classify(tok,prior,chosen):
    before=status(tok,prior)
    require(before in (COMPLETE,INCOMPLETE),"reference prefix must be valid or incomplete")
    after=status(tok,prior+[chosen])
    if chosen==tok.eos_token_id:
        label="EOS_ON_INCOMPLETE" if before==INCOMPLETE else "EOS_ON_COMPLETE"
    elif chosen in tok.all_special_ids:
        label="NON_EOS_SPECIAL"
    elif after in (INVALID,UNKNOWN):
        label="BROKEN_BYTE_TRANSITION"
    elif before==INCOMPLETE and after==COMPLETE:
        label="COMPLETES_PENDING_UTF8"
    elif after==INCOMPLETE:
        label="VALID_INCOMPLETE"
    elif after==COMPLETE:
        label="VALID_COMPLETE"
    else:raise ValueError(after)
    return before,after,label
def summarize(tok,records):
    transitions=Counter();break_ids=Counter();kinds={};tot=Counter();examples=[]
    for row in records:
        sub=kinds.setdefault(row["kind"],Counter())
        prior=[]
        for p in row["positions"]:
            target,chosen=p["target"],p["top1"]
            require(p["position"]==len(prior)+1,"sequential reference positions")
            before,after,label=classify(tok,prior,chosen)
            gold_after=status(tok,prior+[target])
            require(gold_after in (COMPLETE,INCOMPLETE) and
                    (target!=tok.eos_token_id or gold_after==COMPLETE),"reference byte prefix")
            tot["positions"]+=1;sub["positions"]+=1
            tot["pending_prefix_positions"]+=int(before==INCOMPLETE)
            sub["pending_prefix_positions"]+=int(before==INCOMPLETE)
            transitions[f"{before} -> {label}"]+=1
            hard=label in ("BROKEN_BYTE_TRANSITION","EOS_ON_INCOMPLETE","NON_EOS_SPECIAL")
            for k,v in (("hard_transition_errors",hard),
                        ("broken_byte_transitions",label=="BROKEN_BYTE_TRANSITION"),
                        ("pending_prefix_hard_errors",hard and before==INCOMPLETE),
                        ("complete_prefix_hard_errors",hard and before==COMPLETE),
                        ("pending_completed",label=="COMPLETES_PENDING_UTF8"),
                        ("premature_EOS",chosen==tok.eos_token_id and target!=tok.eos_token_id),
                        ("premature_EOS_incomplete",label=="EOS_ON_INCOMPLETE"),
                        ("non_EOS_special",label=="NON_EOS_SPECIAL")):
                tot[k]+=int(v);sub[k]+=int(v)
            if hard:
                break_ids[chosen]+=1
                if len(examples)<20:examples.append({"id":row["id"],"position":p["position"],
                    "gold_prefix_ids":prior[-8:],"target":target,"chosen":chosen,
                    "chosen_raw_piece":str(tok.convert_ids_to_tokens(chosen)),
                    "before":before,"after":after,"error":label})
            prior.append(target)
        require(prior[-1]==tok.eos_token_id,"complete reference")
    require(tot["positions"]==842 and len(records)==42,"all original TRAIN42 reference positions")
    return {"totals":dict(tot),"transition_counts":dict(transitions),
            "by_kind":{k:dict(v) for k,v in kinds.items()},
            "most_frequent_error_ids":[{"token":i,"raw":str(tok.convert_ids_to_tokens(i)),"count":n}
                 for i,n in break_ids.most_common(12)],"examples":examples}
def run(report,tok):
    require(report["stage"]=="S119Q2I8Z58_GPT_TRAIN42_LEARNING_AUDIT" and
            report["TRAIN_SHA256"]==TRAIN_SHA and report["Z55_joint_SHA256"]==CHILD_SHA and
            report["Z55_GT12_SHA256"]==GT_SHA and report["FP32_calls"]==0 and
            report["no_new_or_spent_DEV_scored"] and report["output_gate"]==0 and
            tok.eos_token_id==2 and tok.vocab_size==49152,"pinned Z58 TRAIN-only artifact")
    p,c=(report["per_case"][a] for a in ARMS)
    require([(r["id"],[v["target"] for v in r["positions"]]) for r in p]==
            [(r["id"],[v["target"] for v in r["positions"]]) for r in c],
            "matched exact gold paths")
    result={a:summarize(tok,report["per_case"][a]) for a in ARMS}
    require([result[a]["totals"]["premature_EOS"] for a in ARMS]==[36,49],"match prior observed physical counts")
    return {"stage":"Z62_TRAIN42_UTF8_FORENSICS","arms":result,"new_model_forwards":0,
            "optimizer_steps":0,"FP32_calls":0,"no_DEV_scoring":True,"output_gate":0,
            "candidate_accepted":False,"reference_conditioned_not_autonomous":True}
if __name__=="__main__":
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    p=argparse.ArgumentParser();p.add_argument("--report",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    result=run(json.loads(a.report.read_text(encoding="utf8")),tok)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("Z62_UTF8_FORENSICS",json.dumps({**result,"arms":{k:{x:y for x,y in v.items() if x!="examples"} for k,v in result["arms"].items()}},ensure_ascii=False,sort_keys=True),flush=True)
