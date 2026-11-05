"""Precommitted v177 mixed-precision physical source + native behavioral pilot.

No QA labels affect quantization. All output bytes counted. Never merged
to main; private v89 TEST, LIVE Jessica, Render and Supabase untouched.
"""
import argparse
import gc
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
from direct_runtime_v59 import NativeKernel
from paid_lowrank_v76 import make_model
from v84.sources import original_teacher
from v169.sources import all_cases
from v174.intervene import extract_original
from v174.protocol import freeze_cases
from v174.answers import annotate
from v174.run import save
from v175.evaluate import generate, summarize
from .protocol import ARMS, MAX_BYTES, TOTAL_UNIQUE, SOURCE_SHA, SEED, SPEED_IDS, SMOKE_IDS
from .codec import encode, CB177, unpack_indices
from .native import build_native


def matrix_audit(archive, model):
    names = ("model.embed_tokens.weight",
             "model.layers.0.self_attn.q_proj.weight",
             "model.layers.10.mlp.gate_proj.weight",
             "model.layers.29.mlp.down_proj.weight")
    report = {}
    for i, name in enumerate(names):
        h = archive.handles[name]
        ids = unpack_indices(h.packed.tobytes(), h.shape[0]*h.shape[1]//8,h.bits)
        dense = h.centroids[ids].reshape(h.shape)*h.row_scales[:,None]
        x = (np.random.default_rng(SEED+i).normal(size=(2,h.shape[1])) /
             np.sqrt(h.shape[1])).astype(np.float32)
        real = model.lm_head.kernel.linear(h,x)
        baseline = x@dense.T
        rel = float(np.linalg.norm((real-baseline).astype(np.float64)) /
                    max(np.linalg.norm(baseline.astype(np.float64)),1e-15))
        if not np.isfinite(real).all() or rel>3e-4:
            raise AssertionError("Native variable-bit matrix multiplication discrepancy: " +name)
        report[name] = {"width":h.bits,"relative_native_vs_decoded":rel}
    emb = archive.handles["model.embed_tokens.weight"]
    if archive.handles["lm_head.weight"] is not emb:
        raise AssertionError("Embedding / LM head must share exact handle")
    targets = np.array([0,19,128,2000,49151],dtype=np.int64)
    real = model.lm_head.kernel.embedding(emb,targets)
    ids = unpack_indices(emb.packed.tobytes(),emb.shape[0]*emb.shape[1]//8,emb.bits)
    dense = (emb.centroids[ids].reshape(emb.shape)*emb.row_scales[:,None])[targets]
    if not np.allclose(real,dense,atol=1e-6,rtol=0):
        raise AssertionError("Native variable-bit embedding decoder mismatch")
    report["tied_input_output"] = True
    return report


@torch.inference_mode()
def first_tokens(models, tokenizer, cases):
    chosen = [c for c in cases if c["question_id"] in SMOKE_IDS]
    result = {arm: [] for arm in models}
    for case in chosen:
        question = case["prompt"]
        formatted = tokenizer.apply_chat_template(
            [{"role":"user","content":question}],tokenize=False,add_generation_prompt=True)
        ids = torch.tensor([tokenizer.encode(formatted,add_special_tokens=False)],dtype=torch.long)
        for name, model in models.items():
            logits = model(input_ids=ids,use_cache=False).logits[0,-1,:]
            if not torch.isfinite(logits).all():
                raise AssertionError("Nonfinite prefilling logits")
            result[name].append(int(logits.argmax()))
    if len(chosen) != len(SMOKE_IDS):
        raise AssertionError("Incorrect frozen early-token questions")
    stats = {}
    for name, tokens in result.items():
        stats[name] = {"unique_first_tokens":len(set(tokens)),
                       "agree_with_original_FP32":sum(a==b for a,b in zip(tokens,result["FP32"])),
                       "first_tokens":tokens}
    # This is only an early expensive-generation triage, NOT answer quality.
    # Rule fixed before observing candidate results.
    expand = any(stats[a]["unique_first_tokens"] >= 3 and
                 stats[a]["agree_with_original_FP32"] >= 2 for a in ARMS)
    return stats, expand


@torch.inference_mode()
def timed_speed(models, tokenizer, cases):
    selected = [c for c in cases if c["question_id"] in SPEED_IDS]
    for arm in ("CB59",)+ARMS:
        generate(models[arm],tokenizer,selected[0]["prompt"],cap=8,stop=False)
    record = {a:[] for a in ARMS}
    for arm in ARMS:
        for repeat in range(2):
            for i, case in enumerate(selected):
                order = ("CB59",arm) if (i+repeat)%2==0 else (arm,"CB59")
                timing={}
                for name in order:
                    r=generate(models[name],tokenizer,case["prompt"],cap=32,stop=False)
                    if len(r["token_ids"])!=32:
                        raise AssertionError("Unmatched timed decode")
                    timing[name] = r["wall_seconds"]
                record[arm].append({"case":case["question_id"],"repeat":repeat,
                                    "pair_seconds":timing,
                                    "candidate_speed_ratio":timing["CB59"]/timing[arm]})
    return {arm:{"fixed_new_tokens":32,"includes_prefill":True,
                 "matched_pairs":record[arm],
                 "median_candidate_over_CB59":statistics.median(
                     row["candidate_speed_ratio"] for row in record[arm])} for arm in ARMS}


@torch.inference_mode()
def behavioral(models, tokenizer, cases, count, cap, out):
    picked = cases if count==96 else [c for c in cases if c["question_id"] in SMOKE_IDS]+[
        c for c in cases if c["question_id"] in (4,5,6,7)]
    if len(picked)!=count or len({r["question_id"] for r in picked})!=count:
        raise AssertionError("Incomplete predeclared QA suite")
    responses = {a:[] for a in ("FP32","CB59")+ARMS}
    for ix, case in enumerate(picked):
        hashes=[]
        for arm in ("FP32","CB59")+ARMS:
            result=generate(models[arm],tokenizer,case["prompt"],cap=cap,stop=True)
            responses[arm].append(annotate(case,result))
            hashes.append(result["prompt_sha"])
        if len(set(hashes))!=1:
            raise AssertionError("Different tokenized prompts across arms")
        if (ix+1)%4==0:
            save(out/"answers_partial.json",{"completed":ix+1,"answers":responses})
            print(json.dumps({"stage":"actual_QA","questions_each_arm":ix+1,
                              "chosen_cases":count}),flush=True)
    return responses


def run(source, out):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    out.mkdir(parents=True,exist_ok=True)
    archive = extract_original(source)
    if archive.hash()!=SOURCE_SHA:
        raise ValueError("Original reference source changed")
    cases=freeze_cases(all_cases())
    frozen={"experiment":"v177_variable_physically_paid_matrix_bit_allocation",
            "source_SHA256":SOURCE_SHA,"original_bytes":len(archive.blob),
            "total_unique":TOTAL_UNIQUE,"max_bytes":MAX_BYTES,
            "arms":{"ADAPT_A":"6-bit tied embedding+attention, 4-bit MLP gate/up, 5-bit MLP down",
                    "ADAPT_B":"6-bit tied embedding+MLP down, 4-bit MLP gate/up, 5-bit attention"},
            "source_group_length":8,"index_widths_bits":[4,5,6],
            "retrain_against_original_source_weights_not_QA":True,
            "fit_uses_only_FP32_weights_and_no_QA_or_TEST":True,
            "cases":cases,"early_token_ids":list(SMOKE_IDS),
            "early_expand_gate":"any candidate >=3 unique and >=2 first-token matches to FP32 on 20 historical DEVELOPMENT queries",
            "reduced_suite_count":24,"reduced_new_token_cap":32,
            "expanded_suite_count":96,"expanded_new_token_cap":64,
            "quality_metric":"actual answers and semantic review; no KL/NLL/PPL",
            "not_a_blind_independent_TEST":True,"no_production_changes":True}
    save(out/"protocol_frozen_before_source_fit.json",frozen)
    teacher = original_teacher()
    if set(teacher.state_dict())!=set(archive.meta["full_original_state_keys"]):
        raise AssertionError("Original FP32 reference mismatch")
    blobs, sealing = {}, {}
    for arm in ARMS:
        blob, entry = encode(archive,teacher.state_dict(),arm)
        path=out/("CALIBRI_v177_"+arm+".cb177")
        path.write_bytes(blob)
        if CB177(path.read_bytes()).hash()!=hashlib.sha256(blob).hexdigest():
            raise AssertionError("Unsealed physical candidate")
        blobs[arm]=blob
        sealing[arm]=entry
        save(out/("candidate_"+arm+"_sealed_before_QA.json"),entry)
        print(json.dumps({"stage":"source_saved","arm":arm,"bytes":len(blob),
                          "bpp":entry["bpp"],"SHA256":entry["sha256"]}),flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    with TemporaryDirectory(dir=out) as tmp:
        root=Path(tmp)
        path=root/"CB59.cb59";path.write_bytes(archive.blob)
        base_folder=root/"baseline";base_folder.mkdir()
        baseline,meta,_=make_model("baseline",SimpleNamespace(original=path,folder=base_folder))
        models={"FP32":teacher,"CB59":baseline}
        native_checks={}
        for arm in ARMS:
            candidate,saved,_=build_native(blobs[arm],root/arm)
            native_checks[arm]=matrix_audit(saved,candidate)
            models[arm]=candidate
        if any(p.ndim>=2 for arm in ("CB59",)+ARMS for p in models[arm].parameters()):
            raise AssertionError("Expanded pretrained dense source in inference")
        save(out/"native_C_roundtrip.json",native_checks)
        prefill,expanded=first_tokens(models,tokenizer,cases)
        save(out/"first_token_triage.json",{"measurements":prefill,
             "expanded_to_96":expanded,"early_proxy_not_answer_quality":True})
        print(json.dumps({"stage":"first_token_triage","unique_tokens":
                          {k:v["unique_first_tokens"] for k,v in prefill.items()},
                          "FP32_first_token_agreement":{k:v["agree_with_original_FP32"] for k,v in prefill.items()},
                          "expanded":expanded}),flush=True)
        speed=timed_speed(models,tokenizer,cases)
        save(out/"native_matched_speed.json",speed)
        print(json.dumps({"stage":"native_speed","median_ratio":
                          {k:v["median_candidate_over_CB59"] for k,v in speed.items()}}),flush=True)
        n,cap=(96,64) if expanded else (24,32)
        answers=behavioral(models,tokenizer,cases,n,cap,out)
        report={"complete":True,"protocol":frozen,"sealed_candidates":sealing,
                "native_checks":native_checks,"prefill":prefill,
                "early_gate_expanded":expanded,"native_speed":speed,
                "responses_per_arm":n,"new_token_cap":cap,
                "answers":answers,"summary":summarize(answers),
                "quality_is_not_established_by_fit_loss":True,
                "development_questions_not_blind_TEST":True,
                "low_diversity_not_enough_to_claim_false_96_answer_accuracy":True}
        for arm in ARMS:
            path=out/("CALIBRI_v177_"+arm+".cb177")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=sealing[arm]["sha256"]:
                raise AssertionError("Candidate changed during QA")
        save(out/"v177_report.json",report)
        print(json.dumps({"stage":"complete","expanded":expanded,"responses_per_arm":n,
                          "summary":report["summary"]}),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    x=p.parse_args()
    run(x.source,x.out)
