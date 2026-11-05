"""Physical H32/H64 source fitting, native validation, real answers and speed."""
import argparse,hashlib,json,statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from direct_codebook_v59 import unpack5
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v174.answers import annotate
from v175.evaluate import generate,summarize
from . import protocol as p
from .rotation import apply
from .codec import encode,read
from .native import build

def numerical(archive,model,arm):
    names=("model.embed_tokens.weight","model.layers.0.self_attn.q_proj.weight",
           "model.layers.10.mlp.gate_proj.weight","model.layers.29.mlp.down_proj.weight")
    out={}
    for i,name in enumerate(names):
        h=archive.handles[name]
        code=unpack5(h.packed.tobytes(),h.shape[0]*h.shape[1]//8)
        w=apply(h.centroids[code].reshape(h.shape)*h.row_scales[:,None],p.BLOCKS[arm],True)
        x=(np.random.default_rng(p.SEED+i).normal(size=(2,h.shape[1]))/
           np.sqrt(h.shape[1])).astype(np.float32)
        r=model.lm_head.kernel.linear(h,x)
        v=x@w.T
        relative=float(np.linalg.norm((r-v).astype(np.float64))/
                       max(np.linalg.norm(v.astype(np.float64)),1e-15))
        if relative>3e-4 or not np.isfinite(r).all(): raise AssertionError("Native weighted source "+name)
        out[name]={"relative_native_error":relative}
    h=archive.handles["model.embed_tokens.weight"]
    ids=np.array([0,19,128,2000,49151],dtype=np.int64)
    codes=unpack5(h.packed.tobytes(),h.shape[0]*h.shape[1]//8).reshape(h.shape[0],-1)
    weights=h.centroids[codes[ids].reshape(-1)].reshape(len(ids),h.shape[1])*h.row_scales[ids,None]
    ref=apply(weights,p.BLOCKS[arm],True)
    if not np.allclose(model.lm_head.kernel.embedding(h,ids),ref,rtol=0,atol=3e-6):
        raise AssertionError("Rotated embedding roundtrip")
    if archive.handles["lm_head.weight"] is not h:raise AssertionError("Broken tied source")
    out["tied_source"]=True
    return out

@torch.inference_mode()
def early(models,tok,cases):
    chosen=[x for x in cases if x["question_id"] in p.SMOKE_IDS]
    if len(chosen)!=20:raise AssertionError("Wrong smoke set")
    first={arm:[] for arm in models}
    for case in chosen:
        prompt=tok.apply_chat_template([{"role":"user","content":case["prompt"]}],
                tokenize=False,add_generation_prompt=True)
        ids=torch.tensor([tok.encode(prompt,add_special_tokens=False)],dtype=torch.long)
        for arm,model in models.items():
            logits=model(input_ids=ids,use_cache=False).logits[0,-1]
            if not torch.isfinite(logits).all():raise ValueError("Nonfinite logits")
            first[arm].append(int(logits.argmax()))
    report={k:{"tokens":v,"unique":len(set(v)),
           "FP32_first_match":sum(a==b for a,b in zip(v,first["FP32"]))}
           for k,v in first.items()}
    expand=any(report[k]["unique"]>=p.TRIAGE_MIN_DISTINCT and
               report[k]["FP32_first_match"]>=p.TRIAGE_MIN_FP32_MATCH for k in p.ARMS)
    return report,expand

@torch.inference_mode()
def speed(models,tok,cases):
    chosen=[c for c in cases if c["question_id"] in p.SPEED_IDS]
    for arm in ("CB59",)+p.ARMS:generate(models[arm],tok,chosen[0]["prompt"],cap=8,stop=False)
    out={}
    for arm in p.ARMS:
        rows=[]
        for repeat in range(2):
            for i,case in enumerate(chosen):
                durations={}
                order=("CB59",arm) if (repeat+i)%2==0 else (arm,"CB59")
                for key in order:
                    value=generate(models[key],tok,case["prompt"],cap=32,stop=False)
                    assert len(value["token_ids"])==32
                    durations[key]=value["wall_seconds"]
                rows.append({"seconds":durations,"ratio":durations["CB59"]/durations[arm],
                             "case":case["question_id"],"repeat":repeat})
        out[arm]={"median_speed_ratio":statistics.median(x["ratio"] for x in rows),
                  "pairs":rows,"includes_prefill":True,"new_tokens":32}
    return out

@torch.inference_mode()
def answers(models,tok,cases,out,expanded):
    selected=cases if expanded else [c for c in cases if c["question_id"] in p.REDUCED_QA_CASE_IDS]
    if len(selected)!=(96 if expanded else 24):raise ValueError("Missing frozen QA subset")
    results={arm:[] for arm in ("FP32","CB59")+p.ARMS}
    cap=64 if expanded else 32
    for i,case in enumerate(selected):
        hashes=[]
        for arm in results:
            response=generate(models[arm],tok,case["prompt"],cap=cap)
            results[arm].append(annotate(case,response))
            hashes.append(response["prompt_sha"])
        if len(set(hashes))!=1:raise AssertionError("Different prompts")
        if (i+1)%4==0:
            save(out/"answers_partial.json",{"done":i+1,"answers":results})
            print(json.dumps({"stage":"answers","per_arm":i+1}),flush=True)
    return results,cap

def run(source,out):
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    out.mkdir(parents=True,exist_ok=True)
    original=extract_original(source)
    if original.hash()!=p.SOURCE_SHA:raise ValueError("Original CB59 SHA differs")
    cases=freeze_cases(all_cases())
    protocol={"stage":"v180_signed_orthogonal_H32_H64_pretrained_source",
         "model":MODEL,"revision":MODEL_SHA,"source_SHA256":p.SOURCE_SHA,
         "unique_original":p.TOTAL_UNIQUE,"max_complete_bytes":p.MAX_BYTES,
         "arms":p.BLOCKS,"frozen_dev_cases":cases,"early_case_ids":list(p.SMOKE_IDS),
         "reduced_case_ids":list(p.REDUCED_QA_CASE_IDS),
         "gate_distinct":p.TRIAGE_MIN_DISTINCT,
         "gate_FP32_first_matches":p.TRIAGE_MIN_FP32_MATCH,
         "source_weight_only_no_QA_or_teacher_logits_during_fit":True,
         "no_private_TEST":True,"no_KL_NLL_PPL_acceptance":True,
         "no_production_changes":True}
    save(out/"precommitted_protocol_before_fit.json",protocol)
    teacher=original_teacher()
    state=teacher.state_dict()
    if set(state)!=set(original.meta["full_original_state_keys"]):raise ValueError("Teacher mismatch")
    source_blobs={};seals={}
    for arm in p.ARMS:
        blob,report=encode(original,state,arm)
        filename=out/("CALIBRI_v180_"+arm+".cb59")
        filename.write_bytes(blob)
        if read(filename.read_bytes(),arm).hash()!=report["sha256"]:
            raise AssertionError("Packed source seal mismatch")
        source_blobs[arm]=blob;seals[arm]=report
        save(out/("sealed_"+arm+"_before_QA.json"),report)
        print(json.dumps({"stage":"saved","arm":arm,"bytes":report["bytes"],
             "bpp":report["bpp"],"SHA256":report["sha256"],
             "rmse":report["family_transformed_source_RMSE"]}),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as tmp:
        root=Path(tmp);path=root/"CB59.cb59";path.write_bytes(original.blob)
        (root/"base").mkdir()
        base,_,_=make_model("baseline",SimpleNamespace(original=path,folder=root/"base"))
        models={"FP32":teacher,"CB59":base}
        audits={}
        for arm in p.ARMS:
            model,source_reader,_=build(source_blobs[arm],arm,root/arm)
            audits[arm]=numerical(source_reader,model,arm)
            models[arm]=model
        if any(x.ndim>=2 for key in ("CB59",)+p.ARMS for x in models[key].parameters()):
            raise AssertionError("Dense pretrained source leaked into native model")
        save(out/"native_roundtrip.json",audits)
        triage,expanded=early(models,tokenizer,cases)
        save(out/"first_token_triage.json",{"arms":triage,"expanded":expanded})
        print(json.dumps({"stage":"first_tokens","expanded":expanded,"arms":
              {k:{"unique":v["unique"],"matches":v["FP32_first_match"]} for k,v in triage.items()}}),flush=True)
        timings=speed(models,tokenizer,cases)
        save(out/"native_paired_speed.json",timings)
        print(json.dumps({"stage":"speed","ratios":{k:v["median_speed_ratio"] for k,v in timings.items()}}),flush=True)
        responses,cap=answers(models,tokenizer,cases,out,expanded)
        for arm in p.ARMS:
            if hashlib.sha256((out/("CALIBRI_v180_"+arm+".cb59")).read_bytes()).hexdigest()!=seals[arm]["sha256"]:
                raise AssertionError("Archive changed during QA")
        result={"complete":True,"protocol":protocol,"sealed":seals,
             "native_roundtrip":audits,"triage":triage,"speed":timings,
             "answers":responses,"summary":summarize(responses),
             "questions_per_arm":len(responses["FP32"]),"cap":cap,
             "historical_development_only":True,
             "ungraded_content_requires_semantic_review":True}
        save(out/"v180_complete.json",result)
        print(json.dumps({"stage":"complete","count":result["questions_per_arm"],
              "summary":result["summary"]}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path);parser.add_argument("out",type=Path)
    args=parser.parse_args();run(args.source,args.out)
