"""Actual autonomous text and CPU, not teacher-logit or weight similarity.

This report assesses FIRST complete generated answers, not teacher-forced
training loss. Every mode is charged for actual CPU policy inference.
"""
import re,time,torch
from v108.generation import build_chat
from v94.generation import generate_one
from .protocol import VALID,OPEN
from .steer import Installed

def normalize(s):
    return re.sub(r"[^\w]+"," ",s).strip().casefold()

@torch.inference_mode()
def evaluate(model,tok,paid,policy,conditions="prompt"):
    model.eval();policy.eval()
    output={"qa":[],"open":[]}
    installed=Installed(model,paid)
    questions=[q for q,_ in VALID]+list(OPEN)
    try:
        for ix,q in enumerate(questions):
            q_control=(questions[(ix+5)%len(questions)]
                       if conditions=="shuffled" else q)
            start=time.perf_counter()
            gains=installed.set_question(policy,q_control)
            controller_seconds=time.perf_counter()-start
            answer=generate_one(model,tok,build_chat(tok,q),stop_at_eos=True)
            elapsed=time.perf_counter()-start
            row={"question":q,"actual_text":answer["text"][:650],
                 "generated_ids":answer["ids"],
                 "eos":tok.eos_token_id in answer["ids"],
                 "controller_input_question_sha256":__import__("hashlib").sha256(q_control.encode()).hexdigest(),
                 "controller_seconds":controller_seconds,
                 "whole_answer_with_controller_seconds":elapsed,
                 "minimum_gate":float(gains.min()),
                 "maximum_gate":float(gains.max()),
                 "mean_gate":float(gains.mean())}
            if ix<len(VALID):
                gold=VALID[ix][1]
                row["correct_reference"]=gold
                row["correct"]=normalize(answer["text"])==normalize(gold)
                row["strict_correct_and_eos"]=bool(row["correct"] and row["eos"])
                output["qa"].append(row)
            else:
                ids=[x for x in answer["ids"] if x!=tok.eos_token_id]
                row["distinct_token_fraction"]=len(set(ids))/max(1,len(ids))
                row["open_task_semantic_quality_not_auto_scored"]=True
                output["open"].append(row)
            installed.clear_question()
    finally:
        installed.restore()
    outrows=output["qa"]+output["open"]
    output["summary"]={
        "n_factual":len(output["qa"]),
        "n_open":len(output["open"]),
        "correct_facts":sum(x["correct"] for x in output["qa"]),
        "strict_correct_and_eos":sum(x["strict_correct_and_eos"] for x in output["qa"]),
        "genuine_eos_count":sum(x["eos"] for x in outrows),
        "total_wall_seconds_all_answers_including_neural_controller":sum(
            x["whole_answer_with_controller_seconds"] for x in outrows),
        "total_controller_seconds":sum(x["controller_seconds"] for x in outrows),
        "all_question_gate_min":min(x["minimum_gate"] for x in outrows),
        "all_question_gate_max":max(x["maximum_gate"] for x in outrows),
        "different_prompt_mean_gate_spread":max(x["mean_gate"] for x in outrows)-min(
            x["mean_gate"] for x in outrows),
    }
    return output

@torch.inference_mode()
def speed(model,tok,paid,policy):
    from .steer import Installed
    queries=(VALID[0][0],VALID[3][0],OPEN[0])
    samples=[]
    installed=Installed(model,paid)
    try:
        for roundno in range(2):
            for q in queries:
                start=time.perf_counter()
                installed.set_question(policy,q)
                result=generate_one(model,tok,build_chat(tok,q),stop_at_eos=False)
                elapsed=time.perf_counter()-start
                samples.append({"q":q,"seconds_including_tunnel":elapsed,
                                "generated_token_count":len(result["ids"])})
                installed.clear_question()
    finally:installed.restore()
    count=sum(x["generated_token_count"] for x in samples)
    wall=sum(x["seconds_including_tunnel"] for x in samples)
    return {"tokens_per_second_including_controller":count/wall,
            "actual_generated_tokens":count,"timed_prompt_count":len(samples),
            "total_wall_seconds":wall}
