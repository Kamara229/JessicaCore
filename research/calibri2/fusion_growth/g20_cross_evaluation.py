"""G20: fixed Russian cross-evaluation of actual reloaded G16/G17/G18.
No training or automatic model promotion. Evaluate all predeclared arms,
all G14 prompts, and unseen G19 paired DEV. Export blinded semantic review.
Run only once complete, verified training artifacts are available.
"""
import argparse
import csv
import json
import random
from collections import Counter,defaultdict
from contextlib import nullcontext
from pathlib import Path
import torch
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import digest
from research.calibri2.fusion_growth.g05_134m_joint import restore,G05
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g14_russian_ladder import suite,calibri,normalize,CAPS
from research.calibri2.fusion_growth.g16_russian_connected_language import data
from research.calibri2.fusion_growth import g17_centered_language_readout as g17
from research.calibri2.fusion_growth import g18_early_context_readout as g18

ARMS=("G05","G16_EPOCH1","G16_EPOCH2","G17_RAW","G17_CENTERED","G18_EARLY")
ROOT=Path(__file__).parent/"corpora"/"g19"

def read_jsonl(p):
    return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s.strip()]

def all_rows(tok):
    base=suite()
    g19_train=read_jsonl(ROOT/"TRAIN.jsonl")
    g19_dev=read_jsonl(ROOT/"DEV.jsonl")
    g16_train=[r for v in data(tok,"TRAIN").values() for r in v]
    assert len(base)==128 and len(g19_train)==192 and len(g19_dev)==64
    seen={r["q"] for r in g19_train+g16_train}
    assert not seen.intersection(r["q"] for r in base+g19_dev)
    rows=[{**r,"id":"G14/"+r["id"],"suite":"G14","pair_id":None} for r in base]
    rows += [{"id":"G19/"+r["id"],"suite":"G19","level":"L7",
              "q":r["q"],"accepted":[r["answer"]],"rubric":"Смысловая правильность, соответствие условию, полное грамматически связное предложение.",
              "pair_id":r["pair_id"],"family":r["family"]} for r in g19_dev]
    assert len({r["id"] for r in rows})==192
    assert len({r["q"] for r in rows})==192
    return rows

def read_verified(root,stem,hashes):
    raw=tuple((root/(stem+k)).read_bytes() for k in PIN)
    assert all(digest(v)==hashes[k] for k,v in zip(PIN,raw)),stem
    return raw

def summary(rows,answers):
    groups=defaultdict(list);pairs=defaultdict(list)
    for r,a in zip(rows,answers):
        groups[(r["suite"],r.get("family",r["level"]))].append((r,a))
        if r["pair_id"] is not None:pairs[r["pair_id"]].append(a)
    out={}
    for (cohort,group),items in groups.items():
        aa=[a for _,a in items];scored=[a for a in aa if a["accepted_exact"] is not None]
        cnt=Counter(normalize(a["text"]) for a in aa)
        out[cohort+"/"+group]={"n":len(aa),"exact_scored_n":len(scored),
            "exact":sum(a["accepted_exact"] is True for a in scored) if scored else None,
            "exact_and_EOS":sum(a["accepted_exact"] is True and a["eos"] for a in scored) if scored else None,
            "EOS":sum(a["eos"] for a in aa),"empty":sum(not a["text"].strip() for a in aa),
            "unique_normalized_answers":len(cnt),"most_common":cnt.most_common(3)}
    assert len(pairs)==32 and all(len(x)==2 for x in pairs.values())
    out["G19/pairs"]={"n":32,"both_exact_and_EOS":sum(all(a["accepted_exact"] is True and a["eos"] for a in aa) for aa in pairs.values()),
       "different_normalized_answers":sum(normalize(a[0]["text"])!=normalize(a[1]["text"]) for a in pairs.values())}
    return out

def run(g05,g16dir,g17dir,g18dir,f08,out):
    from transformers import AutoTokenizer
    torch.set_num_threads(2);torch.manual_seed(20270110)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    rows=all_rows(tok)
    original=read_verified(g05,"GROWTH_134M_",PIN)
    g16report=json.loads((g16dir/"G16_REPORT.json").read_text())
    g17report=json.loads((g17dir/"G17_REPORT.json").read_text())
    g18report=json.loads((g18dir/"G18_REPORT.json").read_text())
    assert len(g16report["epochs"])==2
    assert all(r["training"]["steps"]==72 for r in g16report["epochs"])
    for report in (g17report,g18report):
        assert report["base_sha256"]==PIN
    specs=[("G05",original,None,None)]
    for i in (1,2):
        v=read_verified(g16dir,f"EPOCH{i}_",g16report["epochs"][i-1]["sha256"])
        specs.append((f"G16_EPOCH{i}",v,None,None))
    for name,folder,report,arm,mod in (
      ("G17_RAW",g17dir,g17report,"RAW",g17),
      ("G17_CENTERED",g17dir,g17report,"CENTERED",g17),
      ("G18_EARLY",g18dir,g18report,"CENTERED_EARLY_L0",g18)):
        aux=(folder/(arm+".G17R")).read_bytes()
        assert digest(aux)==report["arms"][arm]["residual_sha256"]
        specs.append((name,original,aux,mod))
    assert tuple(x[0] for x in specs)==ARMS
    out.mkdir(parents=True,exist_ok=False)
    tokenizer_bytes=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    (out/"G20_SUITE.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2))
    results={};all_answers={}
    # G19 targets are complete sentences; use same 80-token cap as G14 L7.
    # Native prefix ABI <160 remains enforced by the existing forward.
    for arm,weights,aux,mod in specs:
        st,*rest=restore(weights[:6],weights[6],weights[7])
        module=mod.unpack(aux) if mod else None
        context=mod.install(st,module) if mod else nullcontext()
        answers=[]
        with context,(out/(arm+".jsonl")).open("w",encoding="utf8") as f:
            for row in rows:
                a=calibri(st,tok,row);answers.append(a)
                f.write(json.dumps({"id":row["id"],**a},ensure_ascii=False)+"\n");f.flush()
        all_answers[arm]=answers
        n=sum(map(len,weights))+len(aux or b"")+tokenizer_bytes
        results[arm]={"stored_bytes":n,"known_payload_bpp":8*n/G05,
                      "within_known_budget":8*n/G05<=.7,"scores":summary(rows,answers)}
        print("G20_ARM",arm,json.dumps(results[arm],ensure_ascii=False,sort_keys=True),flush=True)
        del st,rest,module
    # Fixed seeded shuffled, model-identity-blind reviewer table. Reviewers
    # assign independently: meaning, grammatical completeness, no invented
    # facts (0/1 each); no automatically inferred semantic correctness.
    rng=random.Random(20270110);mapping=[]
    with (out/"G20_BLIND_REVIEW.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.writer(f);w.writerow(["review_id","question","reference","rubric","answer","meaning_0_or_1","grammar_0_or_1","grounded_0_or_1","notes"])
        for i,r in enumerate(rows):
            order=list(ARMS);rng.shuffle(order)
            for j,arm in enumerate(order):
                rid=f"R{i+1:03}-{j+1:02}";mapping.append({"review_id":rid,"arm":arm,"question_id":r["id"]})
                w.writerow([rid,r["q"]," | ".join(r["accepted"]),r.get("rubric") or "Правильный ответ и соблюдение инструкции.",
                            all_answers[arm][i]["text"],"","","",""])
    (out/"G20_BLIND_KEY.json").write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
    report={"stage":"G20_FIXED_CROSS_EVAL","arms":results,"questions_per_arm":192,
            "training_updates":0,"candidate_accepted":False,
            "limitations":["G14 is a previously viewed benchmark; G19 is fresh held-out template-based DEV, not an independent broad corpus.",
             "Exact scores are conservative; blinded semantic scores must be filled by an actual reviewer and are not claimed here.",
             "Different outputs in a pair alone do not demonstrate understanding. Both correct and completed is a stricter metric.",
             "Known payload cap includes tokenizer and residual bytes, not executable runtime or RAM. No historical 616 replay or CPU certification.",
             "All six arms predeclared, no test-driven training or checkpoint promotion."]}
    (out/"G20_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G20_ASSESS",json.dumps({"arms":list(ARMS),"candidate_accepted":False}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","g16","g17","g18","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.g16,a.g17,a.g18,a.f08,a.out)
