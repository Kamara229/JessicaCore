"""One-time Z56 frozen full-answer evaluation. No training or FP32 feedback."""
import torch
from collections import Counter
from research.calibri_nra3.s119q2i8z56_gpt_fullanswer_prereg import (
    score_text,MAX_NATIVE_STEPS,ARMS,verify,rows,
)
from research.calibri_nra3.s119q2i8z55_gpt_curriculum import curriculum,digest
DEV_SHA="2140e0bbaa07a57c6cf5384b19937b1671de4dd1c2f9f813ad2a7f0c7a36f2b9"
TRAIN_SHA="d5575db43379d4922d6939082353ae3c16724a142c75621ac9c7a049a561530a"
CHILD_SHA="f6bbcf792a0cdf91482c3c07a4d37228d0cb982dadde6537975495cf16548a88"
GT_SHA="79fd6297ebd350140f15d68bea9146c08e89b767ed6404c7ba311793616d44a6"
def check(ok,msg):
    if not ok: raise ValueError(msg)
def freeze_check(report):
    check(verify()["sha256"]==DEV_SHA and digest(curriculum())==TRAIN_SHA,
          "immutable precommitted Z56 and Z55 training source")
    check(report["stage"]=="S119Q2I8Z55_GPT_ONLY_FULL_SEQUENCE_JOINT_TRAIN"
          and report["GPT_curriculum_SHA256"]==TRAIN_SHA
          and report["FP32_teacher_calls"]==0
          and report["training"]["real_all_neural_and_gt_Adam_updates"]==42
          and report["NEW_real_original_S142_43M_decoder_forwards"]==842
          and report["saved_research_only"]["joint"]["sha256"]==CHILD_SHA
          and report["saved_research_only"]["GT12"]["sha256"]==GT_SHA
          and report["candidate_accepted"] is False and report["output_gate"]==0,
          "pin completed physical Z55 weights before first DEV generation")
def greedy(prompt_ids,step,*,vocab=49152,eos=2):
    check(len(prompt_ids)+MAX_NATIVE_STEPS<160,"preflight exact native chat cap")
    generated=[]
    for _ in range(MAX_NATIVE_STEPS):
        x=step(list(prompt_ids)+generated)
        check(x.shape==(1,vocab) and not x.requires_grad and bool(torch.isfinite(x).all()),
              "frozen full native logits")
        idx=int(x.argmax())
        generated.append(idx)
        if idx==eos: break
    return generated
def summarize(arm_rows):
    check(set(arm_rows)==set(ARMS),"three precommitted arms")
    ids={r["id"] for r in rows()}
    out={}
    for arm,data in arm_rows.items():
        check(len(data)==28 and {r["id"] for r in data}==ids,
              "all cases exactly once per arm")
        def metrics(items):
            return {"n":len(items),**{k:sum(bool(r[k]) for r in items) for k in
                ("whole_answer_and_EOS","whole_text_match","EOS","valid_UTF8")},
                "actual_native_forwards":sum(len(r["generated_ids"]) for r in items),
                "mean_generated_native_length":sum(len(r["generated_ids"]) for r in items)/len(items)}
        out[arm]={**metrics(data),"per_kind":{k:metrics([r for r in data if r["kind"]==k]) for k in sorted({r["kind"] for r in data})}}
    p,c=out[ARMS[0]],out[ARMS[1]]
    gate={"min_complete_correct":c["whole_answer_and_EOS"]>=4,
          "min_complete_gain_vs_parent":c["whole_answer_and_EOS"]-p["whole_answer_and_EOS"]>=2,
          "no_UTF8_regression_vs_parent":c["valid_UTF8"]>=p["valid_UTF8"]}
    return {"metrics":out,"internal_gate":gate,"all_internal_gates_pass":all(gate.values()),
            "FP32_report_only":True,"candidate_accepted":False,"output_gate":0}
