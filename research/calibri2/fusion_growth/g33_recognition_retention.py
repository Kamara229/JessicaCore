"""Read-only recognition versus generation, same native prompts across model sizes.
Historical 616 source TRAIN identities, not independent QA. No optimizer.
The historical S194 specialized decoder is a different pipeline, not emulated.
"""
import argparse,json,random,statistics,hashlib
from collections import Counter,defaultdict
from contextlib import nullcontext
from pathlib import Path
import torch
from transformers import AutoTokenizer
from safetensors.torch import load
from direct_codebook_v59 import CB59
from research.calibri2.fusion_growth.f01_physical_budget import NAMES,PIN as ORIGINAL_PIN,unpack,digest
from research.calibri2.fusion_growth.f02_entropy_core import raw_core
from research.calibri2.fusion_growth.f03_rate_train_core import decode
from research.calibri2.fusion_growth.f14_direct_loader import direct
from research.calibri2.fusion_growth.f17_packed_cpu import PARENT
from research.calibri2.fusion_growth.g05_134m_joint import restore
from research.calibri2.fusion_growth.g06_causal_gradient import PIN
from research.calibri2.fusion_growth.g30_l6_parallel_preflight import Residual,install
from research.calibri2.fusion_growth.g32_word_foundation_mps import PARENTS
from research.calibri2.fusion_growth.f21_native_cpu_integration import forward,prompt_features,strict_chat_source_ids
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s122.independent_binding_train_curriculum import materialize_train_only
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes
ARMS=("original","f10","g05","control","repair")

def historical(tok):
    s30.TOK=tok;rows=[]
    for family,pairs in s87.families(tok).items():
        for pair in pairs:
            for side,other in (("left","right"),("right","left")):
                r=pair[side];a=r["answer"];b=pair[other]["answer"]
                rows.append({"id":f'{family}/{pair["pair_id"]}/{side}',"group":"J7_456","q":r["question"],"answer":a,"choices":list(dict.fromkeys([a,b]))})
    assert len(rows)==456
    new,_,_=materialize_train_only(tok)
    groups=defaultdict(list)
    for r in new:groups[r["pair_id"]].append(r)
    for r in new:
        a=r["answer"];choices=list(dict.fromkeys(x["answer"] for x in groups[r["pair_id"]]))
        rows.append({"id":r["id"],"group":"J7_128","q":r["question"],"answer":a,"choices":choices})
    for i,(q,a,g) in enumerate(new_train_paraphrase_probes()):
        # Explicit alternative only on yes/no; no invented counterfactuals for owner prompts.
        choices=["Yes","No"] if g=="numeric_new_wording" else [a]
        rows.append({"id":f"S190/{i}","group":"J7_32","q":q,"answer":a,"choices":choices})
    assert len(rows)==616 and len({r["q"] for r in rows})==616
    return rows

def lexical(g32):
    rows=[json.loads(l) for l in (g32/"PILOT_DEV.jsonl").read_text().splitlines()]
    assert len(rows)==93
    pools=defaultdict(list)
    for r in rows:pools[r["family"]].append(r["answer"])
    rng=random.Random(20270119);out=[]
    for i,r in enumerate(rows):
        a=r["answer"]
        other=sorted(set(pools[r["family"]])-{a},key=lambda x:(abs(len(x)-len(a)),x))
        nearest=other[:min(8,len(other))]
        choices=[a]+rng.sample(nearest,3);rng.shuffle(choices)
        # Closed recognition is measured by scoring actual alternative completions
        # of the identical original prompt, not by injecting correct token prefixes
        # into free generation or adding a text answer selector to the model.
        out.append({"id":f"RU/{i}","group":"RU_"+r["family"],"q":r["q"],"answer":a,"choices":choices})
    return out

def measure(st,tok,r):
    p=strict_chat_source_ids(tok,r["q"]);f=prompt_features(tok,r["q"])
    chat=tok.apply_chat_template([{"role":"user","content":r["q"]}],tokenize=False,add_generation_prompt=True)
    gold=tok.encode(r["answer"],add_special_tokens=False)+[2]
    cache={}
    def z(seq):
        k=tuple(seq)
        if k not in cache:
            cache[k]=forward(st,p+list(seq),f).detach()[0].log_softmax(-1)
        return cache[k]
    def target(a):
        t=tok.encode(chat+a,add_special_tokens=False)
        if t[:len(p)]!=p:return None
        g=t[len(p):]+[2]
        if len(g)<2 or len(p)+len(g)>=160 or 2 in g[:-1] or tok.decode(g[:-1],skip_special_tokens=True).strip()!=a.strip():return None
        return g
    gt=target(r["answer"]);cap=min(max(8,len(gold)+2),159-len(p))
    if cap<=0:return {"excluded":"prompt_budget"}
    ids=[]
    for _ in range(cap):
        ids.append(int(z(ids).argmax()))
        if ids[-1]==2:break
    answer=tok.decode(ids,skip_special_tokens=True).strip()
    result={"generated":ids,"text":answer,"exact_historical_ids":ids==gold,"exact_text_EOS":answer==r["answer"].strip() and ids[-1]==2,
        "first_historical_id":ids[0]==gold[0],"EOS":ids[-1]==2,"empty":not answer,"invalid":"\ufffd" in answer}
    if gt is None:
        result["recognition_excluded"]="gold_native_alignment";return result
    zz=z([]);result["gold_first_rank"]=int((zz>zz[gt[0]]).sum())+1;result["gold_first_prob"]=float(zz[gt[0]].exp())
    result["gold_aligned_ids"]=gt
    opts=[]
    for a in r["choices"]:
        g=target(a)
        if g is None:
            result["recognition_excluded"]="choice_native_alignment";return result
        # Teacher-forced likelihood is a recognition diagnostic, not free generation.
        logs=[float(z(g[:i])[t]) for i,t in enumerate(g)]
        opts.append({"answer":a,"ids":g,"sum_logp":sum(logs),"mean_logp":sum(logs)/len(logs),"text_mean_logp":sum(logs[:-1])/(len(logs)-1)})
    result["options"]=opts
    if len(opts)>1:
        for metric in ("sum_logp","mean_logp","text_mean_logp"):
            best=max(o[metric] for o in opts)
            winners=[o["answer"] for o in opts if abs(o[metric]-best)<1e-8]
            result["correct_"+metric]=len(winners)==1 and winners[0]==r["answer"]
        result["chance"]=1/len(opts)
    return result

def run(f01,f10,g05,f08,g29,g32,lex,out,arm):
    torch.set_num_threads(2);torch.manual_seed(20270119)
    tok=AutoTokenizer.from_pretrained(f08/"tokenizer",local_files_only=True)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    rows=lexical(lex)+historical(tok)
    out.mkdir(parents=True,exist_ok=False)
    (out/"SUITE.json").write_text(json.dumps(rows,ensure_ascii=False))
    ctx=nullcontext();parent_sha=None;head_sha=None
    if arm=="original":
        fs=unpack((f01/"F01.xz.bundle").read_bytes());assert all(digest(fs[k])==v for k,v in ORIGINAL_PIN.items())
        st=direct(CB59(raw_core(fs[NAMES[0]])),{k:fs[k] for k in NAMES[1:]})
    elif arm=="f10":
        raw=(f10/"F10_FP16.bundle").read_bytes();assert digest(raw)==PARENT;st=direct(*decode(raw))
    else:
        blobs=tuple((g05/("GROWTH_134M_"+k)).read_bytes() for k in PIN);assert all(digest(v)==h for v,h in zip(blobs,PIN.values()))
        st=restore(blobs[:6],blobs[6],blobs[7])[0]
        if arm in PARENTS:
            packet=(g29/"adapter.safetensors").read_bytes();parent_sha=digest(packet);assert parent_sha==PARENTS[arm]
            l6=Residual();l6.load_state_dict(load(packet),strict=True);ctx=install(st,6,l6)
            raw=(g32/"word_head_slot.safetensors").read_bytes();head_sha=digest(raw);fields=load(raw)
            st[5].language_head.load_state_dict({k[5:]:v for k,v in fields.items() if k.startswith("head.")},strict=True)
    # Original historical J7 score from actual S194 #43345267876 uses specialized
    # profile/selector/decoder. Native-only path here must not be called S194 parity.
    counts=defaultdict(Counter);ranks=defaultdict(list)
    with ctx,(out/"RESULTS.jsonl").open("w",encoding="utf8") as f:
        for i,r in enumerate(rows):
            m=measure(st,tok,r);d=counts[r["group"]];d["n"]+=1
            for k in ("exact_historical_ids","exact_text_EOS","first_historical_id","EOS","empty","invalid"):
                d[k]+=int(m.get(k,False))
            if "gold_first_rank" in m:ranks[r["group"]].append(m["gold_first_rank"]);d["first_top10"]+=int(m["gold_first_rank"]<=10)
            if "correct_mean_logp" in m:
                d["recognition_n"]+=1;d["chance_sum"]+=m["chance"]
                for k in ("sum_logp","mean_logp","text_mean_logp"):d["correct_"+k]+=int(m["correct_"+k])
            if "excluded" in m or "recognition_excluded" in m:d["excluded_or_alignment"]+=1
            f.write(json.dumps({**r,**m},ensure_ascii=False)+"\n");f.flush()
            if (i+1)%32==0:print("G33_PROGRESS",arm,i+1,json.dumps({k:dict(v) for k,v in counts.items()}),flush=True)
    report={"arm":arm,"l6_sha":parent_sha,"g32_sha":head_sha,"metrics":{k:{**v,"median_gold_first_rank":statistics.median(ranks[k]) if ranks[k] else None} for k,v in counts.items()},"training_updates":0,
        "historical_S194_reference":{"run":43345267876,"full_EOS_456":456,"full_EOS_128":128,"full_EOS_new32":0,"first_new32":16,"different_specialized_decoder":True},
        "limitations":["Native sequential output only; historical S194 specialized decoder not reproduced. Same historical source prompts and exact standalone token targets, not independent evaluation.","616 historical source prompts include English by design, only for retention, no English training.","Four-choice Russian lexical recognition uses candidate completion likelihood and explicit EOS; length-normalized and without-EOS metrics alongside raw sum. Chance 25%. Not free generation.","Some historical pairs share answer or lack alternate; report denominators and alignment exclusions. No claims about complete protected per-identity J7 parity until specialized path separately verified."]}
    (out/"G33_REPORT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("G33_REPORT",json.dumps(report,ensure_ascii=False),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    for n in ("f01","f10","g05","f08","g29","g32","lex","out"):p.add_argument("--"+n,type=Path,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    a=p.parse_args();run(a.f01,a.f10,a.g05,a.f08,a.g29,a.g32,a.lex,a.out,a.arm)
