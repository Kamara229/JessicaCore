"""Actual 135M green-tunnel intervention: paid U/V gain affects real text?

No learned/controller steering claim. This isolates whether existing rank1
CB59-compatible corrective operators have functional control authority.
"""
import argparse,hashlib,json,time,re
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from binary_geometry_v64 import MODEL,MODEL_SHA,EXPECTED_UNIQUE
from paid_lowrank_v76 import make_model
from v118.archive import Paid
from v118.runtime import restore
from v108.generation import build_chat
from v94.generation import generate_one
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from .gates import attach_gates
from .protocol import TASKS,OPEN_TASKS,SCHEDULE

def check(model,tok):
    answers=[]
    for q,truth in TASKS:
        output=generate_one(model,tok,build_chat(tok,q),stop_at_eos=True)
        a=output["text"]
        gold=re.sub(r"[^\w]+"," ",truth).strip().casefold()
        got=re.sub(r"[^\w]+"," ",a).strip().casefold()
        answers.append({
            "question":q,"gold_reference":truth,"actual":a[:200],
            "correct_exact":gold==got,
            "first_turn_eos":tok.eos_token_id in output["ids"],
            "generated_ids":output["ids"]})
    for q in OPEN_TASKS:
        output=generate_one(model,tok,build_chat(tok,q),stop_at_eos=True)
        answers.append({"question":q,"gold_reference":None,
                       "actual":output["text"][:200],
                       "first_turn_eos":tok.eos_token_id in output["ids"],
                       "generated_ids":output["ids"]})
    return {"correct_8_probes":sum(x["correct_exact"] for x in answers[:len(TASKS)]),
            "rows":answers}

def run(path,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    original=Paid(path.read_bytes())
    if (hashlib.sha256(original.base).hexdigest()!=ORIGINAL_CB59_SHA
        or EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("not original packed complete 135M source")
    base=out/"unmodified_original.cb59"
    base.write_bytes(original.base)
    model,physical,_=make_model("baseline",SimpleNamespace(original=base,folder=out))
    if not physical["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("unpaid dense pretrained weights entered")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    results={}
    for name,(attn,mlp) in SCHEDULE:
        installed=attach_gates(model,original,attn,mlp)
        try:
            t=time.perf_counter()
            cases=check(model,tok)
            delta=time.perf_counter()-t
            if any(p.ndim>=2 for p in model.parameters()):
                raise AssertionError("native original weights are not compressed")
        finally:
            restore(model,installed)
        cases["total_seconds"]=delta
        cases["attention_gain"]=attn
        cases["mlp_gain"]=mlp
        results[name]=cases
        print(json.dumps({
            "stage":"v121_real_gate","mode":name,
            "correct_of_8":cases["correct_8_probes"],
            "examples":[z["actual"][:80] for z in cases["rows"][:4]]},
            ensure_ascii=False),flush=True)
    anchor=results["original_paid_joint"]["rows"]
    summary={}
    for name,r in results.items():
        changed=sum(a["generated_ids"]!=b["generated_ids"]
                    for a,b in zip(r["rows"],anchor))
        summary[name]={
            "real_generation_different_from_gain1_baseline":changed,
            "correct_out_of_eight":r["correct_8_probes"],
            "total_genuine_prompts":len(r["rows"]),
            "wall_seconds":r["total_seconds"]}
    report={
        "research":"GreenTunnel v121 gain sensitivity of PREPAID 60 rank1 CB59 corrections",
        "real_model":MODEL,
        "full_original_model_sha_locked":True,
        "original_native_packed_model_bpp":original.stats()["all_in_bpp"],
        "no_extra_model_file_or_hidden_dense_source":True,
        "interventions_use_PREVIOUSLY_PAID_rank1_U_V":True,
        "no_new_NEURAL_controller_trained_in_this_probe":True,
        "existing_8MB_v120_critic_not_used_to_pick_per_prompt_gates":True,
        "all_gains_fixed_before_quality_tests":True,
        "predeclared_task_probes_not_independent_v89_TEST":True,
        "scientific_quality_pass":False,
        "runs":results,"summary":summary,
        "no_prod_or_main_changes":True,
    }
    (out/"v121_gate_real.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"stage":"v121_gate_complete","summary":summary}),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("source",type=Path);ap.add_argument("out",type=Path)
    a=ap.parse_args();run(a.source,a.out)
