"""Do existing CB59 corrective operators have enough control authority?

A correct answer is used in an example-specific OPTIMIZATION for an
upper-bound diagnostic, not a deployable or universal policy. Neutral and
an already-trained 8MB neural Green are evaluated independently.
"""
import argparse,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from binary_geometry_v64 import EXPECTED_UNIQUE,MODEL,MODEL_SHA
from paid_lowrank_v76 import make_model
from v117.protocol import ORIGINAL_CB59_SHA,TOTAL_UNIQUE
from v118.archive import Paid
from v119.native import build,install,restore
from v122.policy import SharedGreenPolicy
from v122.steer import Installed
from .protocol import SHARED_GREEN_SHA,ORACLE_CASES,TRAIN_MODES,MAX_DELTA,STEPS
from .actuation import Actuator
from .oracle import optimize
from .targets import prompt_input,gold_first_token,score,answer

def exact(text,truth):
    import re
    norm=lambda t:re.sub(r"[^\w]+"," ",t).strip().casefold()
    return norm(text)==norm(truth)

def observation(model,tok,source,prompt,target,other,gold):
    m=score(model,source,target,other)
    a=answer(model,tok,prompt)
    return {"first_token_diagnostic":m,"real_autoregressive_answer":a,
            "whole_answer_exact_and_eos":bool(exact(a["text"],gold) and a["reached_eos"])}

def collect(core,shared,out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    src_bytes=core.read_bytes()
    paid=Paid(src_bytes)
    if (hashlib.sha256(paid.base).hexdigest()!=ORIGINAL_CB59_SHA
        or EXPECTED_UNIQUE!=TOTAL_UNIQUE):
        raise AssertionError("not complete original physical trained 135M core source")
    frozen_file_bytes=shared.read_bytes()
    if (hashlib.sha256(frozen_file_bytes).hexdigest()!=SHARED_GREEN_SHA
        or len(frozen_file_bytes)>10_000_000):
        raise AssertionError("pretrained ONE shared Green neural weights were modified")
    policy=SharedGreenPolicy()
    policy.load_state_dict(load_file(str(shared)),strict=True)
    policy.eval()
    for p in policy.parameters():p.requires_grad_(False)
    native_path=out/"CB59_original_unchanged.cb59"
    native_path.write_bytes(paid.base)
    model,record,_=make_model("baseline",SimpleNamespace(original=native_path,folder=out))
    if not record["zero_dense_pretrained_2D_hot_weights"]:
        raise AssertionError("old native packed model replaced by dense source")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    kernel,reverse=build(out)
    native_original=install(model,kernel,reverse)
    bridge=None
    observations=[]
    try:
        bridge=Installed(model,paid)
        actuator=Actuator(bridge)
        # No training or gold feedback to the ACTUAL frozen 8MB controller.
        for question,truth,category in ORACLE_CASES:
            raw,prompt=prompt_input(tok,question)
            target=gold_first_token(tok,truth)
            unspaced=tok.encode(truth,add_special_tokens=False)[0]
            actuator.neutral()
            neutral=observation(model,tok,raw,prompt,target,unspaced,truth)
            with torch.no_grad():
                green_gains=actuator.frozen_shared(policy,question)
                frozen_green=observation(model,tok,raw,prompt,target,unspaced,truth)
                gain_span=[float(green_gains.min()),float(green_gains.max())]
            item={"question":question,"domain":category,"REFERENCE_ORACLE_ONLY":truth,
                  "true_leading_space_first_subtoken":target,
                  "source_native_neutral":neutral,
                  "same_frozen_SHARED_8MB_GREEN":frozen_green,
                  "shared_neural_actual_gain_range":gain_span,
                  "oracle_diagnostics":{}}
            # Unlike the real Green, these oracle searches SEE THE GOLD.
            # They cannot qualify as an actual generative improvement.
            for mode in TRAIN_MODES:
                item["oracle_diagnostics"][mode]=optimize(
                    model,tok,bridge,raw,prompt,target,unspaced,mode)
                actual=item["oracle_diagnostics"][mode]["real_greedy_output_after_oracle"]
                item["oracle_diagnostics"][mode]["whole_answer_exact_and_eos"]=bool(
                    exact(actual["text"],truth) and actual["reached_eos"])
            observations.append(item)
            print(json.dumps({"stage":"v127_DIAGNOSTIC_ORACLE",
                  "domain":category,"question":question,
                  "native_rank":neutral["first_token_diagnostic"]["target_first_subtoken_rank"],
                  "frozen_8MB_rank":frozen_green["first_token_diagnostic"]["target_first_subtoken_rank"],
                  "six_oracle_rank":item["oracle_diagnostics"]["shared_six_oracle"]["first_token_diagnostic"]["target_first_subtoken_rank"],
                  "sixty_oracle_rank":item["oracle_diagnostics"]["individual_sixty_oracle"]["first_token_diagnostic"]["target_first_subtoken_rank"]},
                  ensure_ascii=False),flush=True)
        # Wrong-question control: apply the stored oracle's gains for q[i]
        # to q[(i+1)%n]. No gradient or gold used while inferring new q.
        shuffled={}
        for mode in TRAIN_MODES:
            shifted=[]
            for j,row in enumerate(observations):
                later=observations[(j+1)%len(observations)]
                question=later["question"]
                gold=later["REFERENCE_ORACLE_ONLY"]
                raw,prompt=prompt_input(tok,question)
                target=gold_first_token(tok,gold)
                alternative=tok.encode(gold,add_special_tokens=False)[0]
                params=torch.tensor(row["oracle_diagnostics"][mode]["parameters_after_ORACLE_FIT"])
                with torch.no_grad():
                    actuator.oracle(params,mode)
                    result=observation(model,tok,raw,prompt,target,alternative,gold)
                shifted.append({"source_reference_question":row["question"],
                                "actual_new_question":question,
                                **result})
            shuffled[mode]=shifted
    finally:
        if bridge is not None:bridge.restore()
        restore(model,native_original)
    if hashlib.sha256(core.read_bytes()).digest()!=hashlib.sha256(src_bytes).digest():
        raise AssertionError("original physical source changed after oracle studies")
    if hashlib.sha256(shared.read_bytes()).hexdigest()!=SHARED_GREEN_SHA:
        raise AssertionError("fixed 8MB Green weights changed")
    def counts(name):
        if name=="neutral":return [r["source_native_neutral"] for r in observations]
        if name=="frozen":return [r["same_frozen_SHARED_8MB_GREEN"] for r in observations]
        return [r["oracle_diagnostics"][name] for r in observations]
    def summary(rows):
        return {
          "cases":len(rows),
          "gold_first_subtoken_top1":sum(
              x["first_token_diagnostic"]["target_first_subtoken_is_top1"] for x in rows),
          "mean_gold_first_subtoken_rank":sum(
              x["first_token_diagnostic"]["target_first_subtoken_rank"] for x in rows)/len(rows),
          "whole_autoregressive_correct_and_eos":sum(
              x["whole_answer_exact_and_eos"] for x in rows),
          "changed_sequence_from_neutral":sum(
              x.get("real_greedy_output_after_oracle",
                    x.get("real_autoregressive_answer"))["token_ids"]!=
              observations[j]["source_native_neutral"]["real_autoregressive_answer"]["token_ids"]
              for j,x in enumerate(rows)),
        }
    stats={key:summary(counts(key)) for key in ("neutral","frozen",)+TRAIN_MODES}
    report={
      "experiment":"v127 prereg real 135M oracle actuator capacity, not an AI quality claim",
      "original_135M_full_core_SHA256":hashlib.sha256(src_bytes).hexdigest(),
      "original_trained_CB59_SHA_locked":True,
      "real_core_bpp_EXCLUDES_single_shared_controller":paid.stats()["all_in_bpp"],
      "real_core_full_physical_bytes":len(src_bytes),
      "same_frozen_green_neural_sha256":SHARED_GREEN_SHA,
      "one_frozen_shared_neural_file_bytes":len(frozen_file_bytes),
      "green_neural_weights_changed":False,
      "full_deployment_including_green_if_used_once_bytes":len(src_bytes)+len(frozen_file_bytes),
      "ORACLE_gold_is_EXPLICITLY_available_to_per_question_optimization":True,
      "oracle_per_question_train_steps":STEPS,
      "oracle_per_query_backprop_cost_NOT_in_live_inference_budget":True,
      "native_5bit_source_C_forward_backward_used":True,
      "two_predeclared_oracle_actuation_modes_6D_vs_60D":True,
      "max_operator_gain":1.+MAX_DELTA,
      "genuine_same_source_abstract_correctness":stats,
      "examples":observations,
      "wrong_question_oracle_action_controls":shuffled,
      "no_neural_green_policy_training_in_this_diagnostic":True,
      "no_new_source_weights_or_CBL_ARCHIVE_formed":True,
      "SCIENCE_real_quality_PASS":False,
      "formal_v89_INDEPENDENT_TEST_UNOPENED":True,
      "trillion_model_universal_transfer_NOT_PROVEN":True,
      "no_production_main_Render_Supabase_APK_LIVE_changes":True,
    }
    (out/"v127_oracle_gate_reachability.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v127_REAL_CB59_ORACLE_GATE_DONE",
        "core_bpp":paid.stats()["all_in_bpp"],
        "frozen_green_bytes":len(frozen_file_bytes),
        "rank_and_answer_quality":stats,
        "oracle_is_NOT_real_neural_policy":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("real_paid_CB59",type=Path)
    p.add_argument("frozen_8MB_shared_Green",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    collect(args.real_paid_CB59,args.frozen_8MB_shared_Green,args.out)
