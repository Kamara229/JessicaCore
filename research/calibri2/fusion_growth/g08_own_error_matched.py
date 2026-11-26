"""G08: isolate actual own-error negative learning on G05 133.701M.
Identical gold CE, number of updates, and native forwards. Only contrast
negative changes (authored vs verified own wrong generation).
G06/G07A unfinished, so pin the same completed G05 parent. No protected
examples, external teacher, route, or deployment.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from research.calibri2.fusion_growth.g01_math_horizontal_fusion import (
    payload,extra_pack,digest,open_bundle)
from research.calibri2.fusion_growth.g02_attention_discrete import (
    indexed_banks,joint_train)
from research.calibri2.fusion_growth.g05_134m_joint import G05,restore,region_split,rate
from research.calibri2.fusion_growth.g06_causal_gradient import PIN,dataset,pair_audit,compact
from research.calibri2.fusion_growth.f10_joint_own_error_learning import own

SEED=20261231
UNIQUE=64
EPOCHS=2

def selected_pairs(tr,rng):
    result={"math":[],"general":[]}
    for kind,rows in tr.items():
        for family in sorted({r["family"] for r in rows}):
            groups={}
            for r in rows:
                if r["family"]==family:
                    groups.setdefault(r["pair"],[]).append(r)
            assert len(groups)==16 and all(len(v)==2 for v in groups.values())
            vals=list(groups.values())
            for i in rng.permutation(16)[:8]:result[kind].extend(vals[int(i)])
    assert sum(map(len,result.values()))==UNIQUE
    return result

def mine(st,tok,ds):
    records={};counts={}
    for kind,rows in ds.items():
        for row in rows:
            assert row["split"]=="TRAIN"
            seq=own(st,tok,row["q"])
            assert seq
            eos=seq[-1]==2
            txt=tok.decode(seq,skip_special_tokens=True).strip()
            div=next((i for i,(a,b) in enumerate(zip(row["g"],seq)) if a!=b),None)
            # A valid exact answer with a different tokenization is not a
            # semantic error. A truncated correct string is not false either.
            if txt==row["answer"]:
                typ="correct" if eos else "correct_text_missing_EOS"
                eligible=False
            elif not txt:
                typ="empty_or_special_only";eligible=div is not None
            elif kind=="math" and txt.lstrip("-").isdigit():
                typ="wrong_numeric" if int(txt)!=int(row["answer"]) else "equivalent_numeric_format"
                eligible=typ=="wrong_numeric" and div is not None
            elif kind=="general" and txt.lower()==row["answer"].lower():
                typ="equivalent_case";eligible=False
            elif kind=="general" and txt.lower()==row["wrong"].lower():
                typ="wrong_context_value";eligible=div is not None
            else:
                # In these explicitly short answer tasks, nonmatching
                # strings are format/content failures, not verified claims
                # about reasoning or factual knowledge.
                typ="other_wrong_or_malformed";eligible=div is not None
            counts[typ]=counts.get(typ,0)+1
            records[row["q"]]={"q":row["q"],"family":row["family"],"gold":row["answer"],
                               "own_ids":seq,"own_text":txt,"EOS":eos,"classification":typ,
                               "own_divergence":div,"eligible":eligible}
    assert len(records)==UNIQUE
    return records,counts

def run(g05,f08,out):
    torch.manual_seed(SEED);torch.set_num_threads(2)
    blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN)
    assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
    ar,sd,legacy=open_bundle(blobs[0])
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    tokenizer_bytes=sum(p.stat().st_size for p in (f08/"tokenizer").rglob("*") if p.is_file())
    tr,dv=dataset(tok,"TRAIN"),dataset(tok,"DEV")
    rng=np.random.default_rng(SEED);selected=selected_pairs(tr,rng)
    assert not {r["q"] for v in selected.values() for r in v}&{r["q"] for v in dv.values() for r in v}
    schedule=[]
    for _ in range(EPOCHS):
        m=[selected["math"][int(i)] for i in rng.permutation(32)]
        g=[selected["general"][int(i)] for i in rng.permutation(32)]
        schedule.extend(r for pair in zip(m,g) for r in pair)
    assert len(schedule)==UNIQUE*EPOCHS
    out.mkdir(parents=True,exist_ok=False)
    def reload(b):return restore(b[:6],b[6],b[7])
    st,*rest=reload(blobs)
    mined,classes=mine(st,tok,selected)
    before=pair_audit(st,dv,tok)
    count=sum(v["eligible"] for v in mined.values())
    (out/"G08_TRAIN_MINED.json").write_text(json.dumps(mined,ensure_ascii=False,indent=2))
    print("G08_BASE",json.dumps({"DEV":compact(before),"own_error_classes":classes,"eligible":count},sort_keys=True),flush=True)
    del st,rest
    results={}
    for name in ("AUTHORED_NEGATIVE","OWN_NEGATIVE"):
        st,second,added,fourth,reg,old,fifth,sixth,reg2=reload(blobs)
        ext=indexed_banks(second,added)|{"fourth/"+k:v for k,v in fourth.items()}|{"fifth/"+k:v for k,v in fifth.items()}|{"sixth/"+k:v for k,v in sixth.items()}
        assert len(ext)==324
        training=[];replacements=0
        for row in schedule:
            r=dict(row);w=mined[r["q"]]
            if name=="OWN_NEGATIVE" and w["eligible"]:
                r["bad"]=w["own_ids"]
                replacements+=1
            assert any(a!=b for a,b in zip(r["g"],r["bad"]))
            training.append(r)
        trained=joint_train(st,tok,training,name,ext)
        _,packet,_,_=payload(ar,sd,st,legacy)
        rb,rb2=region_split(st,reg,old,reg2)
        vals=(packet,extra_pack(second),extra_pack(added),extra_pack(fourth),rb,extra_pack(fifth),extra_pack(sixth),rb2)
        storage=rate(vals[:6],vals[6],vals[7],G05,tokenizer_bytes)
        assert storage["within_known_weight_cap"]
        for k,v in zip(PIN,vals):(out/(name+"_"+k)).write_bytes(v)
        del st,second,added,fourth,reg,old,fifth,sixth,reg2,ext
        st,*rest=reload(vals)
        post=pair_audit(st,dv,tok);train=pair_audit(st,selected,tok)
        del st,rest
        # Exact correction of the same frozen TRAIN failures, distinct
        # from generalization on DEV. No re-mining during this trial.
        indexed={digest(q.encode()):v for q,v in mined.items()}
        corrected=sum(r["exact"] and indexed[r["q_sha256"]]["eligible"] for r in train["records"])
        info={"training":trained,"own_negative_exposures":replacements,"eligible_TRAIN_corrected":corrected,
              "eligible_TRAIN_total":count,"after_physical_reload":post,"TRAIN_audit":train,
              "storage":storage,"sha256":dict(zip(PIN,[digest(v) for v in vals]))}
        results[name]=info
        print("G08_ARM",name,json.dumps({**info,"after_physical_reload":compact(post),"TRAIN_audit":compact(train)},sort_keys=True),flush=True)
    report={"stage":"G08_MATCHED_OWN_ERROR_NEGATIVES","geometry":G05,
            "unique_TRAIN":UNIQUE,"epochs":EPOCHS,"matched_updates":UNIQUE*EPOCHS,
            "matched_native_training_forwards":sum(len(r["g"]) for r in schedule),
            "DEV":48,"mine_classes":classes,"eligible":count,"baseline":before,"arms":results,
            "candidate_accepted":False,"FP32_teacher_calls":0,"output_gate":0,
            "limitations":["Pinned G05, not unfinished G06/G07A. Same G06 authored TRAIN/DEV corpus for comparability, not a new blind benchmark.",
                           "Own wrong strings verified against authored short-answer targets; malformed/empty not claimed as reasoning errors. Equivalent formats excluded.",
                           "Frozen initial own negatives, no extra correction prompts, identical gold-token CE and matched steps; contrast at actual first divergent token.",
                           "No discrete updates in this experiment. Logical geometry unchanged; no protected 616, historical CPU or full runtime certification."]}
    (out/"G08_REPORT.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("G08_ASSESS",json.dumps({"stage":report["stage"],"candidate_accepted":False},sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("g05","f08","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.g05,a.f08,a.out)
