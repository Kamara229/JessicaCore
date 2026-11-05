"""High-rate recovery controls: what physical bit-rate restores real answers?

V1/V2 are deliberately OVER 0.7 bpp, not production/release candidates.
This is a diagnosis of how much pretrained source information was lost.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import unpack5
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v174.answers import annotate
from v175.evaluate import generate,summarize
from . import protocol as p
from .codec import encode,CB181
from .native import build

def check_native(reader,model):
    paths=("model.embed_tokens.weight",
           "model.layers.0.self_attn.q_proj.weight",
           "model.layers.10.mlp.gate_proj.weight",
           "model.layers.29.mlp.down_proj.weight")
    out={}
    for i,path in enumerate(paths):
        h=reader.handles[path]
        ids=unpack5(h.packed.tobytes(),h.shape[0]*h.shape[1]//h.group)
        dense=(h.centroids[ids].reshape(h.shape)*h.row_scales[:,None])
        x=(np.random.default_rng(181+i).normal(size=(2,h.shape[1]))/
           np.sqrt(h.shape[1])).astype(np.float32)
        got=model.lm_head.kernel.linear(h,x)
        expected=x@dense.T
        err=float(np.linalg.norm((got-expected).astype(np.float64))/
                  max(np.linalg.norm(expected.astype(np.float64)),1e-15))
        if not np.isfinite(got).all() or err>3e-4:
            raise AssertionError("Misexecuted high rate source: "+path)
        out[path]={"relative_native_C_error":err}
    h=reader.handles["model.embed_tokens.weight"]
    ids=np.array([0,19,128,2000,49151],dtype=np.int64)
    code=unpack5(h.packed.tobytes(),h.shape[0]*h.shape[1]//h.group)
    dense=h.centroids[code].reshape(h.shape)*h.row_scales[:,None]
    if not np.allclose(model.lm_head.kernel.embedding(h,ids),dense[ids],rtol=0,atol=2e-6):
        raise AssertionError("Bad V1/V2 native embedding")
    if reader.handles["lm_head.weight"] is not h:
        raise AssertionError("Tied original source lost")
    out["tied"]=True
    return out

@torch.inference_mode()
def first_tokens(models,tok,cases):
    selected=[c for c in cases if c["question_id"] in p.SMOKE_IDS]
    if len(selected)!=20:raise AssertionError("Missing frozen prompts")
    ids={name:[] for name in models}
    for case in selected:
        prompt=tok.apply_chat_template([{"role":"user","content":case["prompt"]}],
            tokenize=False,add_generation_prompt=True)
        tensor=torch.tensor([tok.encode(prompt,add_special_tokens=False)],dtype=torch.long)
        for arm,model in models.items():
            logits=model(input_ids=tensor,use_cache=False).logits[0,-1]
            if not torch.isfinite(logits).all():raise ValueError("NaN model output")
            ids[arm].append(int(logits.argmax()))
    return {name:{"tokens":values,"unique":len(set(values)),
         "FP32_first_token_matches":sum(a==b for a,b in zip(values,ids["FP32"]))}
         for name,values in ids.items()}

@torch.inference_mode()
def evaluate(models,tok,cases,out):
    selected=[c for c in cases if c["question_id"] in p.CASE_IDS]
    if len(selected)!=24:raise AssertionError("Missing frozen development QA set")
    report={arm:[] for arm in models}
    for i,case in enumerate(selected):
        h=[]
        for arm,model in models.items():
            got=generate(model,tok,case["prompt"],cap=32)
            h.append(got["prompt_sha"])
            report[arm].append(annotate(case,got))
        if len(set(h))!=1:raise AssertionError("Inconsistent source prompts")
        if (i+1)%4==0:
            save(out/"answers_partial.json",{"done":i+1,"answers":report})
            print(json.dumps({"stage":"answers","cases_per_arm":i+1}),flush=True)
    return report

def run(source,out):
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    out.mkdir(parents=True,exist_ok=True)
    original=extract_original(source)
    if original.hash()!=p.SOURCE_SHA:
        raise ValueError("Wrong pinned original CB59")
    cases=freeze_cases(all_cases())
    frozen={"stage":"v181_genuine_pretrained_recovery_rate_oracles",
            "source_SHA256":p.SOURCE_SHA,"model":MODEL,"revision":MODEL_SHA,
            "arms":p.GROUPS,"5bit_per_V1_or_V2_group":True,
            "DO_NOT_PROMOTE_RATE_OVER_0p7":True,
            "test_cases":list(p.CASE_IDS),"first_token_cases":list(p.SMOKE_IDS),
            "source_fit_original_FP32_weights_only":True,
            "all_QA_development_not_private_TEST":True,
            "no_NLL_KL_PPL_criterion":True,
            "no_main_or_live_changes":True}
    save(out/"protocol_before_original_weights.json",frozen)
    teacher=original_teacher()
    if set(teacher.state_dict())!=set(original.meta["full_original_state_keys"]):
        raise AssertionError("Teacher checkpoint mismatch")
    source_blobs={};sealed={}
    for arm in p.GROUPS:
        blob,summary=encode(original,teacher.state_dict(),arm)
        (out/("CALIBRI_v181_"+arm+".cb181")).write_bytes(blob)
        if CB181(blob).hash()!=summary["sha256"]:raise ValueError("Unsealed source")
        if summary["bpp"]<=0.7:raise AssertionError("Not expected higher-rate control")
        source_blobs[arm]=blob;sealed[arm]=summary
        save(out/("sealed_"+arm+".json"),summary)
        print(json.dumps({"stage":"source_saved","arm":arm,"bytes":summary["bytes"],
                          "bpp":summary["bpp"],"sha256":summary["sha256"]}),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as tmp:
        root=Path(tmp)
        path=root/"CB59.cb59";path.write_bytes(original.blob)
        (root/"base").mkdir()
        base,_,_=make_model("baseline",SimpleNamespace(original=path,folder=root/"base"))
        models={"FP32":teacher,"CB59":base};audits={}
        for arm in p.GROUPS:
            candidate,reader,_=build(source_blobs[arm],arm,root/arm)
            audits[arm]=check_native(reader,candidate)
            models[arm]=candidate
        if any(param.ndim>=2 for arm in ("CB59",)+tuple(p.GROUPS) for param in models[arm].parameters()):
            raise AssertionError("Unpaid high-rate native inference")
        save(out/"native_roundtrip.json",audits)
        triage=first_tokens(models,tokenizer,cases)
        save(out/"first_token_recovery.json",triage)
        print(json.dumps({"stage":"first_tokens","matches":
           {k:v["FP32_first_token_matches"] for k,v in triage.items()},
           "unique":{k:v["unique"] for k,v in triage.items()}}),flush=True)
        actual=evaluate(models,tokenizer,cases,out)
        for arm in p.GROUPS:
            filename=out/("CALIBRI_v181_"+arm+".cb181")
            if hashlib.sha256(filename.read_bytes()).hexdigest()!=sealed[arm]["sha256"]:
                raise AssertionError("Changed original source during answer evaluation")
        result={"complete":True,"protocol":frozen,"sealed":sealed,"native_numeric":audits,
                "first_tokens":triage,"actual_answers":actual,
                "summary":summarize(actual),"questions_per_arm":24,"cap":32,
                "source_rates_EXCEED_0p7_NOT_ACCEPTABLE":True,
                "only_oracle_for_quality_recovery_threshold":True,
                "ungraded_requires_semantic_review":True}
        save(out/"v181_complete.json",result)
        print(json.dumps({"stage":"complete","summary":result["summary"]}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path);parser.add_argument("out",type=Path)
    args=parser.parse_args();run(args.source,args.out)
