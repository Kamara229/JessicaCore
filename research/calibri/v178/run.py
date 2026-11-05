"""Measure real paid column-preconditioned source, native speed, real answers."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v174.answers import annotate
from v175.evaluate import generate, summarize
from v177.codec import unpack_indices

from . import protocol as p
from .codec import encode, CB178
from .native import build


def native_roundtrip(archive,model):
    names=("model.embed_tokens.weight","model.layers.0.self_attn.q_proj.weight",
           "model.layers.10.mlp.gate_proj.weight",
           "model.layers.29.mlp.down_proj.weight")
    checks={}
    for i,name in enumerate(names):
        h=archive.handles[name]
        ids=unpack_indices(h.packed.tobytes(),h.shape[0]*h.shape[1]//8,5)
        literal=(h.centroids[ids].reshape(h.shape)*
                 h.row_scales[:,None]*h.col_scales[None,:])
        x=(np.random.default_rng(p.SEED+i).normal(size=(2,h.shape[1])) /
            np.sqrt(h.shape[1])).astype(np.float32)
        native=model.lm_head.kernel.linear(h,x)
        decoded=x@literal.T
        err=float(np.linalg.norm((native-decoded).astype(np.float64)) /
                  max(np.linalg.norm(decoded.astype(np.float64)),1e-15))
        if not np.isfinite(native).all() or err>3e-4:
            raise AssertionError("Native scaled operator != literal decoded source: "+name)
        checks[name]={"relative_direct_native_vs_decoded":err,"shape":list(h.shape)}
    h=archive.handles["model.embed_tokens.weight"]
    ids=np.array([0,1,19,128,2000,49151],dtype=np.int64)
    raw=unpack_indices(h.packed.tobytes(),h.shape[0]*h.shape[1]//8,5)
    literal=(h.centroids[raw].reshape(h.shape)*
             h.row_scales[:,None]*h.col_scales[None,:])[ids]
    native=model.lm_head.kernel.embedding(h,ids)
    if not np.allclose(native,literal,atol=1e-6,rtol=0):
        raise AssertionError("Paid column-scaled native embedding mismatch")
    if archive.handles["model.embed_tokens.weight"] is not archive.handles["lm_head.weight"]:
        raise AssertionError("Source input and output head no longer tied")
    checks["tied_embedding_head_shared"]=True
    return checks


@torch.inference_mode()
def early_probe(models,tokenizer,cases):
    chosen=[x for x in cases if x["question_id"] in p.SMOKE_IDS]
    if len(chosen)!=len(p.SMOKE_IDS):
        raise AssertionError("Early quality questions not frozen")
    ids_by_arm={arm:[] for arm in models}
    for case in chosen:
        prompt=tokenizer.apply_chat_template(
            [{"role":"user","content":case["prompt"]}],
            tokenize=False,add_generation_prompt=True)
        inputs=torch.tensor([tokenizer.encode(prompt,add_special_tokens=False)],dtype=torch.long)
        for arm,model in models.items():
            logits=model(input_ids=inputs,use_cache=False).logits[0,-1]
            if not torch.isfinite(logits).all():
                raise ValueError("Invalid source logits")
            ids_by_arm[arm].append(int(logits.argmax()))
    baseline=ids_by_arm["FP32"]
    info={arm:{"first_token_ids":values,
               "unique_first_tokens":len(set(values)),
               "first_token_agreement_FP32":sum(a==b for a,b in zip(values,baseline))}
          for arm,values in ids_by_arm.items()}
    expand=(info["COL178"]["unique_first_tokens"]>=p.EARLY_GATE_UNIQUE and
            info["COL178"]["first_token_agreement_FP32"]>=p.EARLY_GATE_FP32_FIRST_TOKEN_AGREEMENT)
    return info,expand


@torch.inference_mode()
def speed(models,tokenizer,cases):
    chosen=[x for x in cases if x["question_id"] in p.SPEED_IDS]
    if len(chosen)!=len(p.SPEED_IDS):
        raise AssertionError("Incorrect frozen speed sample")
    for arm in ("CB59","COL178"):
        generate(models[arm],tokenizer,chosen[0]["prompt"],cap=8,stop=False)
    pairs=[]
    for repeat in range(2):
        for i,case in enumerate(chosen):
            order=("CB59","COL178") if (i+repeat)%2==0 else ("COL178","CB59")
            times={}
            for arm in order:
                result=generate(models[arm],tokenizer,case["prompt"],cap=32,stop=False)
                if len(result["token_ids"])!=32:
                    raise AssertionError("Unequal measured tokens")
                times[arm]=result["wall_seconds"]
            pairs.append({"case":case["question_id"],"repeat":repeat,
                          "seconds":times,"ratio":times["CB59"]/times["COL178"]})
    return {"pairs":pairs,"fixed_generated_tokens":32,"includes_prefill":True,
            "median_speed_ratio":statistics.median(x["ratio"] for x in pairs)}


@torch.inference_mode()
def actual_answers(models,tokenizer,cases,out,full):
    selected=cases if full else [x for x in cases if x["question_id"] in p.SMOKE_IDS]+[
        x for x in cases if x["question_id"] in (4,5,6,7)]
    cap=64 if full else 32
    if len(selected)!=(p.FULL_QA if full else p.REDUCED_QA):
        raise AssertionError("Incorrect frozen question count")
    answers={arm:[] for arm in ("FP32","CB59","COL178")}
    for i,case in enumerate(selected):
        hashes=[]
        for arm in answers:
            got=generate(models[arm],tokenizer,case["prompt"],cap=cap)
            hashes.append(got["prompt_sha"])
            answers[arm].append(annotate(case,got))
        if len(set(hashes))!=1:
            raise AssertionError("Different question prompts between comparison arms")
        if (i+1)%4==0:
            save(out/"answers_partial.json",{"completed_questions_per_arm":i+1,"answers":answers})
            print(json.dumps({"stage":"real_answers","questions_each_arm":i+1}),flush=True)
    return answers,cap


def run(source,out):
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    out.mkdir(parents=True,exist_ok=True)
    original=extract_original(source)
    if original.hash()!=p.SOURCE_SHA:
        raise ValueError("Original immutable source was changed")
    cases=freeze_cases(all_cases())
    frozen={"stage":"v178_paid_column_preconditioned_source",
            "model":MODEL,"model_revision":MODEL_SHA,
            "original_CB59_SHA256":p.SOURCE_SHA,
            "original_CB59_bytes":p.SOURCE_BYTES,
            "original_unique_parameter_count":p.TOTAL_UNIQUE,
            "complete_max_allowed_bytes":p.MAX_BYTES,
            "matrix_encoding":"CB178 5-bit 32-codebook / 8 weight + FP16 column scales",
            "column_scaling_source_only":True,
            "column_clamp":p.COLUMN_CLAMP,
            "centroid_kmeans_iters":p.KMEANS_ITERATIONS,
            "no_teacher_logits_or_QA_labels_used_for_source_fit":True,
            "early_20_question_ids":list(p.SMOKE_IDS),
            "early_expand_unique":p.EARLY_GATE_UNIQUE,
            "early_expand_FP32_first_token_matches":p.EARLY_GATE_FP32_FIRST_TOKEN_AGREEMENT,
            "full_QA_count":96,"reduced_QA_count":24,
            "full_QA_max_new_tokens":64,"reduced_QA_max_new_tokens":32,
            "full_QA_only_if_precommitted_early_gate_passes":True,
            "cases":cases,"speed_ids":list(p.SPEED_IDS),
            "physical_bpp_and_actual_answers_not_KL_NLL_PPL":True,
            "historical_development_suite_not_blind_TEST":True,
            "no_production_changes":True}
    save(out/"protocol_frozen_before_fit.json",frozen)
    teacher=original_teacher()
    blob,measurements=encode(original,teacher.state_dict())
    candidate=out/"CALIBRI_v178_paid_column_scales.cb178"
    candidate.write_bytes(blob)
    if CB178(candidate.read_bytes()).hash()!=hashlib.sha256(blob).hexdigest():
        raise AssertionError("Persisted compact archive differs")
    save(out/"candidate_sealed_before_QA.json",measurements)
    print(json.dumps({"stage":"source_saved","bytes":len(blob),
                      "bpp":measurements["bpp"],"SHA256":measurements["candidate_sha256"]}),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as tmp:
        path=Path(tmp)
        original_path=path/"CB59.cb59";original_path.write_bytes(original.blob)
        baseline_folder=path/"CB59";baseline_folder.mkdir()
        baseline,meta,_=make_model("baseline",SimpleNamespace(
            original=original_path,folder=baseline_folder))
        candidate_model,saved,_=build(blob,path/"COL178")
        models={"FP32":teacher,"CB59":baseline,"COL178":candidate_model}
        if any(p.ndim>=2 for arm in ("CB59","COL178") for p in models[arm].parameters()):
            raise AssertionError("Unpaid dense source allocation during native inference")
        numerical=native_roundtrip(saved,candidate_model)
        save(out/"native_C_vs_literal.json",numerical)
        probe,expand=early_probe(models,tokenizer,cases)
        save(out/"early_first_token_triage.json",
             {"first_tokens":probe,"expanded":expand,"quality_proxy_only":True})
        print(json.dumps({"stage":"first_tokens","unique":{k:v["unique_first_tokens"] for k,v in probe.items()},
                          "FP32_agreement":{k:v["first_token_agreement_FP32"] for k,v in probe.items()},
                          "full_QA":expand}),flush=True)
        timings=speed(models,tokenizer,cases)
        save(out/"native_speed_matched.json",timings)
        print(json.dumps({"stage":"native_speed","ratio":timings["median_speed_ratio"]}),flush=True)
        responses,cap=actual_answers(models,tokenizer,cases,out,expand)
        if hashlib.sha256(candidate.read_bytes()).hexdigest()!=measurements["candidate_sha256"]:
            raise AssertionError("Candidate mutated during evaluation")
        complete={"complete":True,"protocol":frozen,"source":measurements,
                  "native_numerical_checks":numerical,
                  "first_token_probe":probe,"expanded":expand,"matched_speed":timings,
                  "answers":responses,"summary":summarize(responses),
                  "per_arm_questions":len(responses["FP32"]),
                  "new_token_cap":cap,
                  "ungraded_responses_require_separate_semantic_review":True,
                  "technical_success_is_not_scientific_success":True}
        save(out/"v178_complete.json",complete)
        print(json.dumps({"stage":"complete","summary":complete["summary"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    run(args.source,args.out)
