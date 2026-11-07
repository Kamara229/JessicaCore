"""CALIBRI2 S0 microscopic TRAIN-only learnability experiment, two frozen arms.

Is a single opposing TRAIN Yes/No pair learnable AT ALL in our 22M paid
representation? Two identically random 0.691bpp models, fixed 64 steps:
  A: paid FP16 centroids only (5-bit indices frozen)
  B: identical optimizer plus sparse 5-bit index proposals every 4 steps
Compare on EXACT SAME memorized TRAIN pair with no QA/validation.
This diagnoses whether code updates interfere with the ability to learn
contradictory first-token behavior, NOT knowledge/generalization.
No pretrained model weights or teacher; no Jessica ordinary-chat tests.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v174.run import save
from v176.operators import clear
from v185.optim import propose_codes,apply_codes
from v185.codec import replace_discrete
from v186.pairs import build_pairs
from v186.fit import paired_loss
from v200.rollout import probe_greedy_train
from calibri2.s0.bootstrap import CANDIDATES,build_config
from calibri2.s0.train import create_small_paid_model

SEED=2021
LR=.008
STEPS=64
STOPS=(0,8,16,32,64)
SHA0="2e807eb9c4f82018bcc72f0924726377cafb70df1ad529aefbd8513b6e538d80"
PAIR_ID=0

def snapshot(model,sources,pair):
    with torch.no_grad():
        sides={}
        for side,other in (("left","right"),("right","left")):
            clear(sources)
            ids=pair[side]["prompt_ids"][None]
            logits=model(input_ids=ids,use_cache=False).logits[0,-1]
            if not torch.isfinite(logits).all():
                raise AssertionError("Non-finite tiny source logits")
            gold=int(pair[side]["first_expected_id"])
            opposite=int(pair[other]["first_expected_id"])
            top=int(logits.argmax())
            lps=torch.log_softmax(logits,dim=-1)
            sides[side]={
                "target_first_token":gold,
                "other_answer_token":opposite,
                "greedy_first_token":top,
                "first_token_correct":top==gold,
                "gold_minus_opposite_logit":float(logits[gold]-logits[opposite]),
                "first_token_gold_probability":float(lps[gold].exp()),
                "greedy_probability":float(lps[top].exp()),
                "first_token_rank":int((logits>logits[gold]).sum())+1,
            }
        clear(sources)
        output=probe_greedy_train(model,sources,[pair],max_new_tokens=6)
        v=output[pair["pair_id"]]["sides"]
        for side in ("left","right"):
            sides[side]["full_tokens_with_EOS_correct"]=v[side]["exact_with_EOS"]
            sides[side]["generated"]=v[side]["generated_tokens"]
            sides[side]["expected"]=v[side]["gold_tokens_with_EOS"]
        return {
          "same_greedy_token_for_opposite_questions":
              sides["left"]["greedy_first_token"]==sides["right"]["greedy_first_token"],
          "both_first_answers_correct":all(
              sides[side]["first_token_correct"] for side in ("left","right")),
          "both_full_answers_and_EOS_correct":all(
              sides[side]["full_tokens_with_EOS_correct"] for side in ("left","right")),
          "sides":sides,
        }

def train_arm(name, source_bytes, cfg, pair, result_dir):
    torch.manual_seed(SEED)
    archive=CB59(source_bytes)
    model,sources=create_small_paid_model(archive,cfg)
    model.eval()
    optimizer=torch.optim.Adam([x.centroids for x in sources.values()],lr=LR)
    target_names=sorted(sources.keys())
    history=[];progress=[]
    for step in range(STEPS+1):
        if step in STOPS:
            diag=snapshot(model,sources,pair)
            progress.append({"step":step,**diag})
            print(json.dumps({
                "stage":"CALIBRI2_S0_ONE_PAIR_MICROFIT",
                "arm":name,"step":step,
                "both_correct_first":diag["both_first_answers_correct"],
                "both_correct_full":diag["both_full_answers_and_EOS_correct"],
                "same_greedy":diag["same_greedy_token_for_opposite_questions"],
                "sides":{k:{"prediction":v["greedy_first_token"],
                            "gold_rank":v["first_token_rank"],
                            "margin":v["gold_minus_opposite_logit"]}
                         for k,v in diag["sides"].items()}
            }),flush=True)
        if step==STEPS:break
        target_name=target_names[step%len(target_names)]
        target=sources[target_name]
        captured=[]
        def hook(_m,_x,w):
            if not any(w is t for t in captured):
                w.retain_grad();captured.append(w)
        h=target.register_forward_hook(hook)
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        try:
            loss,fields=paired_loss(model,target,pair)
            objective_before=float(loss.detach())
            loss.backward()
        finally:
            h.remove()
        if not captured or any(w.grad is None for w in captured):
            raise AssertionError("No differentiable full-model target grad")
        dense_grad=sum(w.grad.detach().cpu() for w in captured).numpy().copy()
        if any(x.centroids.grad is None for x in sources.values()):
            raise AssertionError("Some paid codebook not trained by full context")
        grad_norm=float(torch.nn.utils.clip_grad_norm_(
             [x.centroids for x in sources.values()],1.,error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for x in sources.values():x.centroids.clamp_(-16,16)
        clear(sources)
        attempted=accepted=0
        if name=="centroids_and_5bit" and (step+1)%4==0:
            with torch.inference_mode():
                current_loss,_=paired_loss(model,target,pair)
                base=float(current_loss)
                current=target().detach().cpu().numpy().copy()
            candidate=propose_codes(target,dense_grad,current)
            attempted=candidate["proposed_groups"]
            if attempted>512:raise AssertionError("Index reassign exceeds precommitted limit")
            if attempted:
                with torch.no_grad():apply_codes(target,candidate,True)
                clear(sources)
                with torch.inference_mode():
                    next_loss,_=paired_loss(model,target,pair)
                if np.isfinite(float(next_loss)) and float(next_loss)<base-1e-7:
                    accepted=attempted
                else:
                    with torch.no_grad():apply_codes(target,candidate,False)
        clear(sources)
        history.append({"step":step+1,"loss_before":objective_before,
                        "full_model_gradient_norm":grad_norm,
                        "first_token_margins":fields["first_token_logit_gaps"],
                        "code_indices_proposed":attempted,
                        "code_indices_accepted":accepted})
    out_bytes,change=replace_discrete(archive,sources,strict=False)
    out_file=result_dir/f"CALIBRI2_S0_{name}_64steps.cb59"
    out_file.write_bytes(out_bytes)
    n=archive.meta["original_unique_parameter_count"]
    bpp=8*len(out_bytes)/n
    if len(out_bytes)!=len(source_bytes) or bpp>.700:
        raise AssertionError("Full physical neural source grew beyond 0.700bpp")
    if hashlib.sha256(out_bytes).hexdigest()!=change["sha256"]:
        raise AssertionError("Final source sha differs after physical export")
    result={
      "arm":name,"initial_sha256":SHA0,"final_source_sha256":change["sha256"],
      "same_frozen_training_pair":PAIR_ID,"num_unique_params":n,
      "n_stored_paid_codebook_matrices":len(sources),
      "all_stored_parameters_counted":True,
      "physical_bytes":len(out_bytes),"physical_bpp":bpp,
      "initial_pretrained_weights_used":False,
      "QA_or_VALID_or_TEST_used":False,
      "single_pair_microfit_only_not_generalization_or_parity":True,
      "training_steps":STEPS,"snapshots":progress,
      "actual_trainable_update_history":history,
      "physical_changed_source_details":change,
    }
    save(result_dir/f"ACTUAL_S0_microfit_{name}.json",result)
    return result

def main():
    a=argparse.ArgumentParser()
    a.add_argument("random_source",type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    raw=args.random_source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SHA0:
        raise AssertionError("Cannot diagnose modified random source")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    pair=build_pairs(tok,"TRAIN")[PAIR_ID]
    assert pair["domain"]=="yesno_numeric" and pair["split"]=="TRAIN"
    cfg=build_config(next(c for c in CANDIDATES if c[0]=="small_320x6"))
    arms=[train_arm(name,raw,cfg,pair,args.out) for name in (
          "centroids_only","centroids_and_5bit")]
    save(args.out/"ACTUAL_S0_SINGLE_PAIR_AB_LEARNABILITY.json",{
      "purpose":"Is an opposing simple TRAIN Yes/No pair learnable from scratch within paid codebooks and 5-bit indices?",
      "arch":"small_320x6","trained_original_FP32_teacher":False,
      "public_QA_DEV_VALID_TEST_leakage":False,
      "two_arms_equal_random_initial_source":True,
      "only_difference_5bit_indices_training":True,
      "training_not_an_inference_speed_test":True,
      "arms":[{"name":r["arm"],
              "first_tokens_correct_end":r["snapshots"][-1]["both_first_answers_correct"],
              "full_answers_correct_end":r["snapshots"][-1]["both_full_answers_and_EOS_correct"],
              "single_token_collapse_end":r["snapshots"][-1]["same_greedy_token_for_opposite_questions"],
              "final_paid_bpp":r["physical_bpp"],
              "source_sha256":r["final_source_sha256"]}
              for r in arms]
    })
    print(json.dumps({"stage":"CALIBRI2_S0_ONE_PAIR_AB_DONE",
        "results":[{"arm":r["arm"],
          "both_first_correct":r["snapshots"][-1]["both_first_answers_correct"],
          "both_full_correct":r["snapshots"][-1]["both_full_answers_and_EOS_correct"],
          "source_bpp":r["physical_bpp"]} for r in arms]}),flush=True)

if __name__=="__main__":
    main()
