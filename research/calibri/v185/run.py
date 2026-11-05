"""v185 complete pretrained 5-bit mutable index QAT and native QA.

All first token, generated answer and paired CPU throughput checks
use identical frozen PUBLIC DEVELOPMENT prompts after final source seal.
The independent private v89 TEST is never opened, and Jessica
never receives any instruction to run experiments.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.protocol import freeze_cases
from v174.intervene import extract_original
from v174.run import save
from v175.evaluate import generate,summarize
from v176.data import prepare
from v184.run import first_tokens,actual_answers
from . import protocol as p
from .optim import make_student,train
from .codec import replace_discrete


@torch.inference_mode()
def freeze_public_train(teacher,samples):
    refs=[]
    for item in samples:
        ids=item["ids"][None]
        if item["answer_start"] is None:
            logits=teacher(input_ids=ids,use_cache=False).logits
            positions=torch.arange(max(0,ids.shape[1]-5),ids.shape[1],dtype=torch.long)
            targets=logits[0,positions].argmax(dim=-1)
        else:
            start=item["answer_start"]-1
            positions=torch.arange(start,ids.shape[1]-1,dtype=torch.long)
            targets=ids[0,positions+1].clone()
        if positions.numel()==0:raise ValueError("No TRAIN target positions")
        refs.append({"ids":ids.detach().clone(),
                     "positions":positions.detach().clone(),
                     "targets":targets.detach().clone(),
                     "kind":item["kind"]})
    return refs


@torch.inference_mode()
def physical_numerical_control(student,physical,sources,ids):
    # TRAIN input only. Diagnostics are not behavior quality tests.
    ids=torch.tensor(ids.tolist(),dtype=torch.long)
    from v176.operators import clear
    clear(sources)
    expected=student(input_ids=ids,use_cache=False).logits
    measured=physical(input_ids=ids,use_cache=False).logits
    rel=float(torch.linalg.vector_norm(expected-measured)/
              torch.linalg.vector_norm(expected).clamp_min(1e-20))
    clear(sources)
    if not np.isfinite(rel) or rel>2.e-3:
        raise AssertionError("Native physical packed indices differ from QAT source")
    return {"source_to_native_logits_relative_L2":rel,
            "first_token_matches":int(expected[0,-1].argmax())==
                                   int(measured[0,-1].argmax())}


def speed_pairs(models,tokenizer,cases):
    chosen=[x for x in cases if x["question_id"] in p.SPEED_IDS]
    if len(chosen)!=4:raise AssertionError("Four public development timing probes required")
    for arm in ("CB59","CB185"):
        generate(models[arm],tokenizer,chosen[0]["prompt"],cap=8,stop=False)
    rows=[]
    for repeat in range(2):
        for i,case in enumerate(chosen):
            order=("CB59","CB185") if (repeat+i)%2==0 else ("CB185","CB59")
            cost={}
            for name in order:
                record=generate(models[name],tokenizer,case["prompt"],cap=32,stop=False)
                if len(record["token_ids"])!=32:
                    raise AssertionError("Timing work not identical")
                cost[name]=record["wall_seconds"]
            rows.append({"repeat":repeat,"case_id":case["question_id"],
                         "seconds":cost,"candidate_speed_ratio":cost["CB59"]/cost["CB185"]})
    return {"pairs":rows,"median_speed_ratio":statistics.median(
            x["candidate_speed_ratio"] for x in rows),
            "token_work":32,"prefill_included":True}


def run(source,out):
    out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(p.SEED)
    torch.set_num_threads(2)
    archive=extract_original(source)
    if archive.hash()!=p.SOURCE_SHA or len(archive.blob)!=p.SOURCE_BYTES:
        raise ValueError("Different original paid model source")
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    cases=freeze_cases(all_cases())
    train_samples,origin=prepare(tokenizer)
    selected=[train_samples[i] for i in p.PUBLIC_TRAIN_INDICES]
    if len(selected)!=16 or any(
        s.get("question") in {x["question"] for x in cases} for s in selected):
        raise ValueError("Training and QA questions overlap")
    manifest={"experiment":"v185 original-prepaid 5bit code-index gradient reassignment",
        "model":MODEL,"model_revision":MODEL_SHA,
        "original_source_sha256":p.SOURCE_SHA,
        "original_bytes":p.SOURCE_BYTES,
        "total_original_unique_parameters":p.TOTAL_UNIQUE,
        "max_complete_0p700_bytes":p.MAX_BYTES,
        "source_2d_dictionary_entries":32,"source_group_width":8,
        "optimizer_seed":p.SEED,
        "train_steps":p.STEPS,
        "index_update_operator_schedule":list(p.STEPS_OPERATORS),
        "source_prior_guard":p.SOURCE_GUARD_STRENGTH,
        "max_per_step_5bit_swaps":p.MAX_INDEX_UPDATES_PER_STEP,
        "min_local_projected_improvement":p.MIN_SCORE_IMPROVEMENT,
        "centroid_learning_rate":p.LEARNING_RATE_CODEBOOKS,
        "frozen_public_TRAIN_sample_indices":list(p.PUBLIC_TRAIN_INDICES),
        "historical_public_DEV_question_IDS":list(p.CASE_IDS),
        "separate_DEV_first_token_IDS":list(p.SMOKE_IDS),
        "native_CPU_speed_IDs":list(p.SPEED_IDS),
        "original_public_TRAIN_provenance":origin,
        "discrete_update_ACCEPTED_ONLY_ON_ACTUAL_TRAIN_loss":True,
        "evaluation_not_used_for_source_fitting":True,
        "unseen_private_TEST_unaccessed":True,
        "no_KL_NLL_PPL_answer_acceptance":True,
        "fixed_final_iterate_before_any_DEV_QA":True,
        "LIVE_main_Jessica_Render_Supabase_APK_untouched":True,
        "no_Jessica_chat_test":True}
    save(out/"protocol_sealed_before_TRAIN.json",manifest)
    teacher=original_teacher()
    references=freeze_public_train(teacher,selected)
    print(json.dumps({"stage":"train_references","count":len(references)}),flush=True)
    model,sources=make_student(archive)
    original_state=teacher.state_dict()
    weights={path:original_state[path].detach().cpu().numpy()
             for path in set(p.STEPS_OPERATORS)}
    records=train(model,sources,selected,weights,references)
    save(out/"optimizer_audit.json",records)
    final_bytes,seal=replace_discrete(archive,sources,strict=True)
    dest=out/"CALIBRI_v185_mutated_5bit_full_source.cb59"
    dest.write_bytes(final_bytes)
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=seal["sha256"]:
        raise ValueError("Source SHA changed during physical commit")
    if seal["actually_changed_FP16_codebook_cells"]==0:
        raise AssertionError("No jointly trained source codebooks changed")
    if records["accepted_index_swaps"]!=seal["actual_changed_5bit_group_indices"]:
        # A single code can be modified more than once in theory, so
        # one-to-one equality is not generally guaranteed. Record both.
        seal["total_accepted_train_swaps"] = records["accepted_index_swaps"]
        seal["net_modified_original_5bit_codes"] = seal["actual_changed_5bit_group_indices"]
    seal["seals_before_QA"] = True
    save(out/"physical_source_sealed_before_QA.json",seal)
    print(json.dumps({"stage":"physical_source_sealed",
        "bpp":seal["complete_bpp"],"bytes":seal["bytes"],
        "net_modified_indices":seal["actual_changed_5bit_group_indices"],
        "changed_codebook_cells":seal["actually_changed_FP16_codebook_cells"],
        "sha256":seal["sha256"]}),flush=True)
    with TemporaryDirectory(dir=out) as folder:
        root=Path(folder)
        baseline=root/"original.cb59";baseline.write_bytes(archive.blob)
        (root/"base").mkdir()
        (root/"updated").mkdir()
        original_native,_,_=make_model("baseline",
                  SimpleNamespace(original=baseline,folder=root/"base"))
        updated_native,_,_=make_model("baseline",
                  SimpleNamespace(original=dest,folder=root/"updated"))
        if any(p.ndim>=2 for p in updated_native.parameters()):
            raise AssertionError("Unpaid dense original source in CB185 native")
        numerical=physical_numerical_control(
            model,updated_native,sources,references[0]["ids"][:,:8])
        save(out/"native_matches_trained_source.json",numerical)
        print(json.dumps({"stage":"physical_numeric_control",**numerical}),flush=True)
        models={"FP32":teacher,"CB59":original_native,"CB185":updated_native}
        tokens=first_tokens(models,tokenizer,cases)
        save(out/"first_tokens_20_DEV.json",tokens)
        print(json.dumps({"stage":"first_tokens",
            "matches":{k:x["FP32_first_matches"] for k,x in tokens.items()},
            "unique":{k:x["distinct_first_tokens"] for k,x in tokens.items()}}),flush=True)
        speed=speed_pairs(models,tokenizer,cases)
        save(out/"matched_native_speed.json",speed)
        print(json.dumps({"stage":"speed",
            "median_CB185_divided_by_CB59":speed["median_speed_ratio"]}),flush=True)
        answers=actual_answers(models,tokenizer,cases,out)
        grade=summarize(answers)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=seal["sha256"]:
            raise AssertionError("Source tampered after real QA")
        result={"complete":True,"protocol":manifest,"source_seal":seal,
            "training":records,"numerical":numerical,"first_tokens":tokens,
            "native_speed":speed,"actual_24_DEV_answers":answers,
            "dev_grade":grade,"not_tested_PRIVATE_v89":True,
            "no_Jessica_tests":True}
        save(out/"v185_complete.json",result)
        print(json.dumps({"stage":"complete",
            "bpp":seal["complete_bpp"],
            "changed_indices":seal["actual_changed_5bit_group_indices"],
            "native_speed":speed["median_speed_ratio"],
            "grade":{k:v["all"] for k,v in grade.items()}}),flush=True)


if __name__=="__main__":
    a=argparse.ArgumentParser()
    a.add_argument("source",type=Path)
    a.add_argument("output",type=Path)
    args=a.parse_args()
    run(args.source,args.output)
