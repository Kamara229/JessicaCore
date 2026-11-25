"""F27: attribute historical sequence drift to original decoder paths.
Read-only comparison of recorded physical candidate outputs from F15/F25
against a fresh, instrumented, digest-verified original K32 replay.
No training, candidate selection, or rate changes.
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch
from research.calibri2.fusion_growth import f15_legacy_j7_replay as j7
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN,digest,unpack
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f13_profile_rebase import extra_open,BANK_NAME,BANK_SHA
from research.calibri2.fusion_growth.f18_retention_attribution import H584,H32
from direct_codebook_v59 import CB59

def run(f01,f08,f15,f25,out):
    torch.set_num_threads(2);torch.manual_seed(20261213)
    original=unpack((f01/"F01.xz.bundle").read_bytes())
    for name,h in PIN.items():assert digest(original[name])==h
    raw=raw_core(original[NAMES[0]])
    assert digest(raw)=="73b05e4cd542e523b4c5cb828d31b358004513c8e64e5e8f76fde2814bdb0aba"
    archive=CB59(raw);side={n:original[n] for n in NAMES[1:]}
    _,extras=extra_open((f08/"legacy_extras.bundle").read_bytes())
    assert digest(extras[BANK_NAME])==BANK_SHA
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    j7.s30.TOK=tok
    oq=j7.source_questions_456(j7.s87.families(tok));nq,_,_=j7.materialize_train_only(tok)
    probes=j7.new_train_paraphrase_probes()
    rows={**{"OLD/"+k:{"question":q} for k,q in oq.items()},
          **{"NEW/"+r["id"]:{"question":r["question"]} for r in nq},
          **{f"PROBE/{i:02d}":{"question":q} for i,(q,_,_) in enumerate(probes)}}
    del oq,nq,probes
    forced={k for k in rows if k.startswith("PROBE/")}
    keys={k:digest(v["question"].encode()) for k,v in rows.items()}
    assert len(rows)==len(set(keys.values()))==616 and len(forced)==32
    groups={v:k.split("/")[0] for k,v in keys.items()}
    probe_ids={keys[k] for k in forced}
    # Preserve exact source selector and execution; wrappers record the
    # actually executed path only after its original return.
    j7.configure_original_selector()
    state=j7.model_state(archive,side,extras,False)
    parallel=j7.s160.cached_full_cpu_parallel
    ordinary=j7.cached_full_cpu_with_neural_copy
    paths={}
    def wrap_parallel(*a,**kw):
        result=parallel(*a,**kw)
        q=a[6] if len(a)>6 else kw["question"]
        h=digest(q.encode());assert h not in paths
        paths[h]="PARALLEL" if result[1]["parallel"] else "S93_FALLBACK"
        return result
    def wrap_ordinary(*a,**kw):
        result=ordinary(*a,**kw)
        q=a[6] if len(a)>6 else kw["question"]
        h=digest(q.encode());assert h not in paths
        paths[h]="S87_S91_AUTOREGRESSIVE"
        return result
    with patch.object(j7.s160,"cached_full_cpu_parallel",wrap_parallel),patch.object(j7,"cached_full_cpu_with_neural_copy",wrap_ordinary):
        base,_,counts=j7.replay(state,tok,rows,forced)
    del state
    ref=json.loads((f15/"F15_BASELINE_IDS.json").read_text())
    assert base==ref and set(paths)==set(base)
    assert j7.output_digest({k:v for k,v in base.items() if k not in probe_ids})==H584
    assert j7.output_digest({k:v for k,v in base.items() if k in probe_ids})==H32
    assert Counter(paths.values())=={"PARALLEL":549,"S93_FALLBACK":47,"S87_S91_AUTOREGRESSIVE":20}
    # Existing inference records only. No candidate is run, fitted, selected,
    # changed, or inspected by the replay wrapper.
    old=json.loads((f15/"F15_REPORT.json").read_text())
    new=json.loads((f25/"F25_REPORT.json").read_text())
    sources=[("F15_DIRECT",f15/"DIRECT_IDS.json",old["arms"]["DIRECT"]["exact_616"]),
             ("F15_DELTA",f15/"DELTA_IDS.json",old["arms"]["DELTA"]["exact_616"]),
             ("F25_ABSOLUTE",f25/"ABSOLUTE_WEIGHT_ERROR_IDS.json",new["arms"]["ABSOLUTE_WEIGHT_ERROR"]["exact_616"]),
             ("F25_RELATIVE",f25/"RELATIVE_OPERATOR_ERROR_IDS.json",new["arms"]["RELATIVE_OPERATOR_ERROR"]["exact_616"])]
    report={"stage":"F27_HISTORICAL_PATH_ATTRIBUTION","baseline_exact":616,
            "original_path_counts":dict(Counter(paths.values())),
            "baseline_route_counts":counts,"arms":{},"optimizer_steps":0,
            "new_parameters":0,"candidate_accepted":False,"output_gate":0,
            "limitations":["Path membership is measured on original K32 only. Candidate per-case path changes are not observed.",
                           "Recorded F15/F25 outputs are compared after baseline replay. No candidate optimization or new inference claim.",
                           "Historical equality is not independent correctness; no new J7 training or model changes."]}
    for name,file,expected in sources:
        pred=json.loads(file.read_text());assert set(pred)==set(base)
        same={k for k in base if base[k]==pred[k]};assert len(same)==expected
        table={}
        for p,n in Counter(paths.values()).items():
            members={k for k in paths if paths[k]==p}
            ok=len(members&same)
            table[p]={"n":n,"exact":ok,"drift":n-ok,
                      "by_group":{g:{"n":sum(groups[k]==g for k in members),
                                     "drift":sum(groups[k]==g and k not in same for k in members)}
                                  for g in ("OLD","NEW","PROBE")}}
        report["arms"][name]={"exact_616":len(same),"paths":table}
        print("F27_ARM",name,json.dumps(report["arms"][name],sort_keys=True),flush=True)
    out.mkdir(parents=True,exist_ok=False)
    (out/"F27_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F27_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f08","f15","f25","out"):p.add_argument("--"+n,type=Path,required=True)
    a=p.parse_args();run(a.f01,a.f08,a.f15,a.f25,a.out)
