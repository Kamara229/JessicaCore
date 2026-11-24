"""Read-only Z103 TRAIN48 lexical collapse and context sensitivity audit.
No models, optimizer, tokenizer, DEV, or new model calls.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
ARMS=("BEFORE","BASE_GROUP_MEAN_CE","MATCHED_TERMINAL_SHARE_CE")
FAMILIES=("definition","cloze","contrast","plural")
WORDS=("мост","река","лес","дом","кот","пёс","снег","дождь","вода","хлеб","мяч","ключ")
PINS={"BASE_GROUP_MEAN_CE":"56056d675b50b627923a881aaacbb8c193f76d8335fcc25f89b3734cadd13d14",
      "MATCHED_TERMINAL_SHARE_CE":"2be2b70d9ef516b27bd0afc94cc80bd9d3a9f0aa406fd306e4e65d935ec79057"}
def require(ok,msg):
    if not ok: raise ValueError(msg)
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def label(r,words=WORDS):
    if not r["valid_UTF8"]: return "invalid_utf8"
    text=r["generated_text"].strip()
    target=r["target"]
    if text==target: return "exact" if r["EOS"] else "exact_no_eos"
    if not text: return "empty"
    if text in words: return "wrong_known_word"
    if target.startswith(text): return "truncated_target"
    if text.startswith(target): return "target_with_suffix"
    return "other_text"
def aggregate(rows):
    valid=[r for r in rows if r["valid_UTF8"]]
    predictions=Counter(r["generated_text"].strip() for r in valid)
    grouped={w:[r for r in rows if r["target"]==w] for w in WORDS}
    return {"cases":len(rows),"exact":sum(r["complete_correct"] for r in rows),
        "errors":dict(Counter(label(r) for r in rows)),
        "distinct_valid_texts":len(predictions),"most_frequent_valid_texts":predictions.most_common(12),
        "all_four_contexts_correct":sum(len(g)==4 and all(r["complete_correct"] for r in g) for g in grouped.values()),
        "same_valid_answer_all_four_contexts":sum(len(g)==4 and all(r["valid_UTF8"] for r in g) and len({r["generated_text"].strip() for r in g})==1 for g in grouped.values()),
        "by_target":{w:{"exact":sum(r["complete_correct"] for r in g),"predictions":[{"family":r["family"],"text":r["generated_text"],"class":label(r)} for r in g]} for w,g in grouped.items()},
        "first_prefix_length":dict(Counter(r["prefix_match"] for r in rows)),
        "first_two_correct":sum(r["prefix_match"]>=2 for r in rows)}
def run(folder):
    path=folder/"S119Q2I8Z103_GPT_TERMINAL_BALANCE.json"
    report=json.loads(path.read_text(encoding="utf8"))
    require(report["stage"]=="S119Q2I8Z103_GPT_TERMINAL_BALANCE" and report["output_gate"]==0 and not report["candidate_accepted"] and report["no_DEV_scoring"] and report["FP32_calls"]==0 and report["TRAIN_only_not_generalization"],"exact research provenance")
    require(report["new_curriculum_SHA256"]=="f542727442812c48e66016d572fcc03a35db2958c28c4a58c18da24480ba8dd0" and report["native_plan_SHA256"]=="fbaa6959b5d3283a6e865b786134da346837df4bb6528766b6ebb7af6e3a960b","fixed TRAIN48")
    for a,pin in PINS.items():
        require(report["saved_research_only"][a]["joint"]["sha256"]==pin==digest(folder/f"S119Q2I8Z103_{a}_FP16_G0.safetensors"),"fixed checkpoint")
        require(report["training"][a]["Adam_updates"]==1764,"matched budget")
    expected={f"S119Q2I8Z87_GPT_{f}_{i+1:02d}":(f,w) for f in FAMILIES for i,w in enumerate(WORDS)}
    source=report["short_word_predictions"]
    require(set(source)==set(ARMS),"three fixed arms")
    metrics={};paired={}
    for arm in ARMS:
        rows=source[arm]
        require(len(rows)==48 and {r["id"] for r in rows}==set(expected),"exact unique TRAIN48")
        for r in rows:
            require((r["family"],r["target"])==expected[r["id"]],"expected word and family")
            require(r["EOS"]==(r["generated_ids"][-1]==2),"natural EOS")
            require(r["complete_correct"]==(r["valid_UTF8"] and r["EOS"] and r["generated_text"].strip()==r["target"]),"exact score")
        metrics[arm]=aggregate(rows)
        require(metrics[arm]["exact"]==report["short_word_metrics"][arm]["correct"],"reconcile original result")
    b={r["id"]:r for r in source[ARMS[1]]};m={r["id"]:r for r in source[ARMS[2]]}
    paired["transitions"]=dict(Counter(label(b[k])+" -> "+label(m[k]) for k in expected))
    paired["gains"]=[k for k in expected if m[k]["complete_correct"] and not b[k]["complete_correct"]]
    paired["losses"]=[k for k in expected if b[k]["complete_correct"] and not m[k]["complete_correct"]]
    return {"stage":"Z106_TRAIN_WORD_FORENSICS","source_run":45486685533,"source_report_sha256":digest(path),"metrics":metrics,"paired":paired,"new_model_forwards":0,"optimizer_steps":0,"FP32_calls":0,"no_DEV_scoring":True,"TRAIN_only_not_generalization":True,"output_gate":0}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    r=run(a.input);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("Z106_TRAIN_FORENSICS "+json.dumps(r,ensure_ascii=False,sort_keys=True),flush=True)
