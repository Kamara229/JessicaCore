"""Complete v184 paid shared-rank2 QAT -> actual source -> native QA/speed.

Original FP32 is ONLY for source TRAIN label reference and comparison.
All trainable factor bytes, physical source and metadata are charged.
No Jessica, no independent v89 TEST, no production updates.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v174.answers import annotate
from v175.evaluate import generate,summarize
from v176.data import prepare
from . import protocol as p
from .trainable import build as train_build,clear_frozen
from .train import freeze_train_references,train
from .codec import encode,Paid184
from .native import build as native_build


@torch.inference_mode()
def first_tokens(models,tokenizer,cases):
    selected=[case for case in cases if case["question_id"] in p.SMOKE_IDS]
    if len(selected)!=len(p.SMOKE_IDS):
        raise AssertionError("Wrong frozen first-token quality checks")
    all_tokens={name:[] for name in models}
    for case in selected:
        prompt=tokenizer.apply_chat_template(
            [{"role":"user","content":case["prompt"]}],
            tokenize=False,add_generation_prompt=True)
        ids=torch.tensor([tokenizer.encode(prompt,add_special_tokens=False)],dtype=torch.long)
        for name,model in models.items():
            logits=model(input_ids=ids,use_cache=False).logits[0,-1]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite source answer")
            all_tokens[name].append(int(logits.argmax()))
    originals=all_tokens["FP32"]
    return {name:{"token_ids":ids,
                  "distinct_first_tokens":len(set(ids)),
                  "FP32_first_matches":sum(a==b for a,b in zip(ids,originals))}
            for name,ids in all_tokens.items()}


@torch.inference_mode()
def actual_answers(models,tokenizer,cases,out):
    selected=[case for case in cases if case["question_id"] in p.EVAL_IDS]
    if len(selected)!=24:
        raise AssertionError("Unfrozen original public DEV QA")
    result={name:[] for name in models}
    for i,case in enumerate(selected):
        hashes=[]
        for name,model in models.items():
            response=generate(model,tokenizer,case["prompt"],cap=32)
            result[name].append(annotate(case,response))
            hashes.append(response["prompt_sha"])
        if len(set(hashes))!=1:raise AssertionError("Different source prompts")
        if (i+1)%4==0:
            save(out/"QA_partial.json",{"done":i+1,"results":result})
            print(json.dumps({"stage":"actual_QA","cases_per_arm":i+1}),flush=True)
    return result


@torch.inference_mode()
def speed_pairs(models,tokenizer,cases):
    chosen=[x for x in cases if x["question_id"] in p.SPEED_IDS]
    if len(chosen)!=4:
        raise ValueError("Incomplete native CPU throughput cases")
    for arm in ("CB59","CB184"):
        generate(models[arm],tokenizer,chosen[0]["prompt"],cap=8,stop=False)
    rows=[]
    for repeat in range(2):
        for i,item in enumerate(chosen):
            order=("CB59","CB184") if (repeat+i)%2==0 else ("CB184","CB59")
            times={}
            for arm in order:
                result=generate(models[arm],tokenizer,item["prompt"],cap=32,stop=False)
                if len(result["token_ids"])!=32:
                    raise AssertionError("CPU work not identical")
                times[arm]=result["wall_seconds"]
            rows.append({"case":item["question_id"],"repeat":repeat,"secs":times,
                         "candidate_speed_ratio":times["CB59"]/times["CB184"]})
    return {"median_speed_ratio":statistics.median(x["candidate_speed_ratio"] for x in rows),
            "pairs":rows,"new_tokens":32,"includes_prefill":True}


@torch.inference_mode()
def exact_native_training_surrogate(student,native,sources,ids):
    clear_frozen(sources)
    expected=student(input_ids=ids,use_cache=False).logits.detach()
    actual=native(input_ids=ids,use_cache=False).logits.detach()
    error=float(torch.linalg.vector_norm(actual-expected).item()/
                max(torch.linalg.vector_norm(expected).item(),1e-20))
    clear_frozen(sources)
    if not np.isfinite(error) or error>2e-3:
        raise AssertionError("Trained rank2 STE surrogate != saved native physical source: "+str(error))
    return {"relative_source_to_native_logits":error,
            "same_first_token":int(expected[0,-1].argmax())==int(actual[0,-1].argmax()),
            "input_token_length":ids.shape[-1]}


def run(source,out):
    torch.manual_seed(p.SEED)
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    archive=extract_original(source)
    if archive.hash()!=p.SOURCE_SHA or len(archive.blob)!=p.SOURCE_BYTES:
        raise AssertionError("Original v161 wrapped CB59 SHA drift")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases=freeze_cases(all_cases())
    all_samples,origin=prepare(tokenizer)
    indices=(0,2,4,6,16,18,20,22,48,49,50,51)
    samples=[all_samples[i] for i in indices]
    if len(samples)!=p.TRAIN_CASES or any(
            sample.get("question") in {case["question"] for case in cases}
            for sample in samples):
        raise AssertionError("Training or development case collision")
    frozen={"experiment":"v184 physically paid shared rank2 INT8 per row FP16 cross layer B",
        "original_SHA256":p.SOURCE_SHA,"original_bytes":p.SOURCE_BYTES,
        "source_unique_original_parameters":p.TOTAL_UNIQUE,
        "max_complete_0p700_bytes":p.MAX_BYTES,
        "headroom_after_unchanged_CB59":p.HEADROOM,
        "source_rank":p.RANK,"steps":p.STEPS,
        "alpha_learning_rate":p.LR_ALPHA,"beta_learning_rate":p.LR_BETA,
        "training_source":origin,"training_indices":indices,
        "training_is_public_TRAIN_only":True,
        "historical_development_24_cases":list(p.EVAL_IDS),
        "first_token_20_cases":list(p.SMOKE_IDS),
        "CPU_speed_cases":list(p.SPEED_IDS),
        "source_model":MODEL,"revision":MODEL_SHA,
        "native_no_dense_trained_source":True,
        "training_CE_not_acceptance_criterion":True,
        "final_quality_real_correct_answers_not_KL_NLL_PPL":True,
        "frozen_last_train_checkpoint_no_qa_selection":True,
        "private_v89_TEST_unaccessed":True,
        "no_main_live_Jessica_Render_Supabase_APK_etc_changes":True,
        "Jessica_runs_no_tests":True}
    save(out/"preregistered_protocol_before_training.json",frozen)
    teacher=original_teacher()
    references=freeze_train_references(teacher,samples)
    print(json.dumps({"stage":"TRAIN_reference_frozen",
        "samples":len(references),"train_corpus":origin["sample_count"],
        "alpha_source_byte_cap":p.HEADROOM}),flush=True)
    student,sources,alphas,betas=train_build(archive)
    training=train(student,sources,alphas,betas,references)
    save(out/"training_curve.json",training)
    alphas_cpu={name:a.detach().cpu().numpy().copy() for name,a in alphas.items()}
    betas_cpu={width:b.detach().cpu().numpy().copy() for width,b in betas.items()}
    blob,seal=encode(archive,alphas_cpu,betas_cpu)
    candidate=out/"CALIBRI_v184_joint_rank2_paid.cb184"
    candidate.write_bytes(blob)
    paid=Paid184(candidate.read_bytes())
    if paid.hash()!=seal["sha256"]:
        raise AssertionError("Physical model post-save SHA failed")
    seal["alpha_nonzero_8bit_values"]=sum(int(np.count_nonzero(x[0])) for x in paid.alpha.values())
    if seal["alpha_nonzero_8bit_values"]==0:
        raise AssertionError("No learned rank2 source correction in signed physical archive")
    save(out/"candidate_sealed_before_any_QA.json",seal)
    print(json.dumps({"stage":"source_sealed","bytes":seal["complete_physical_bytes"],
        "bpp":seal["complete_bpp"],"sha256":seal["sha256"],
        "alpha_nonzero":seal["alpha_nonzero_8bit_values"]}),flush=True)
    with TemporaryDirectory(dir=out) as tmp:
        tmp=Path(tmp)
        baseline_source=tmp/"original.cb59";baseline_source.write_bytes(archive.blob)
        (tmp/"CB59").mkdir()
        baseline,_,_=make_model("baseline",SimpleNamespace(
            original=baseline_source,folder=tmp/"CB59"))
        candidate_model,payload,native_audit=native_build(blob,tmp/"CB184")
        control={"TRAIN_surrogate_vs_physical_native":
            exact_native_training_surrogate(student,candidate_model,sources,
                references[0]["ids"][:,:8]),"native_source_audit":native_audit}
        if any(x.ndim>=2 for x in candidate_model.parameters()):
            raise AssertionError("Native 184 contains unpaid dense 2D pretrained model tensors")
        if payload.base.hash()!=p.SOURCE_SHA:
            raise AssertionError("Unchanged 5-bit source not present")
        save(out/"physical_native_numerical_controls.json",control)
        print(json.dumps({"stage":"numerical_control",**control["TRAIN_surrogate_vs_physical_native"]}),flush=True)
        models={"FP32":teacher,"CB59":baseline,"CB184":candidate_model}
        first=first_tokens(models,tokenizer,cases)
        save(out/"first_token_20_frozen_DEV.json",first)
        print(json.dumps({"stage":"first_tokens",
            "matches":{k:v["FP32_first_matches"] for k,v in first.items()},
            "unique":{k:v["distinct_first_tokens"] for k,v in first.items()}}),flush=True)
        timing=speed_pairs(models,tokenizer,cases)
        save(out/"native_CPU_paired_speed.json",timing)
        print(json.dumps({"stage":"native_speed",
            "median_vs_CB59":timing["median_speed_ratio"]}),flush=True)
        answers=actual_answers(models,tokenizer,cases,out)
        summary=summarize(answers)
        if hashlib.sha256(candidate.read_bytes()).hexdigest()!=seal["sha256"]:
            raise AssertionError("Source changed while QA was measured")
        output={"complete":True,"precommitted":frozen,
            "sealed_complete_source":seal,"training":training,
            "native_numerics":control,"first_tokens":first,
            "speed":timing,"actual_answers":answers,"QA_summary":summary,
            "per_model_actual_questions":24,"new_token_cap":32,
            "no_private_TEST":True,"no_Jessica_testing":True}
        save(out/"v184_complete.json",output)
        print(json.dumps({"stage":"complete",
            "bpp":seal["complete_bpp"],"speed":timing["median_speed_ratio"],
            "summary":{key:value["all"] for key,value in summary.items()}}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    a=parser.parse_args()
    run(a.source,a.output)
