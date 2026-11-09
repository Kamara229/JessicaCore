"""CALIBRI2 S0 four-domain TRAIN-only bounded retention pilot.
No pretrained FP32, no heldout data, no scale-up. Not run automatically.
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
STEPS=384
STOPS=(0,96,192,288,384)
SHA0="2e807eb9c4f82018bcc72f0924726377cafb70df1ad529aefbd8513b6e538d80"
PAIR_IDS=(0,6,12,18)

def snapshot(model,sources,pairs,tok):
    """Frozen four-prompt greedy full answer + EOS, not a selection metric."""
    with torch.no_grad():
        clear(sources)
        raw=probe_greedy_train(model,sources,pairs,max_new_tokens=12)
        sides={}
        first=[]
        repeats=0
        for pair in pairs:
            key=str(pair["pair_id"])
            sides[key]={}
            for side in ("left","right"):
                d=raw[pair["pair_id"]]["sides"][side]
                ids=d["generated_tokens"]
                gold=d["gold_tokens_with_EOS"]
                first.append(ids[0] if ids else None)
                repeats+=max(0,len(ids)-len(set(ids)))
                sides[key][side]={
                    "generated_tokens":ids,"expected_tokens_with_EOS":gold,
                    "generated_text":tok.decode([i for i in ids if i!=tok.eos_token_id]),
                    "expected_text":tok.decode([i for i in gold if i!=tok.eos_token_id]),
                    "exact_with_EOS":bool(d["exact_with_EOS"]),
                    "first_token_correct":bool(ids and ids[0]==gold[0]),
                    "eos_terminated":bool(ids and ids[-1]==gold[-1]),
                }
        clear(sources)
        return {"pairs":sides,
                "full_correct":sum(int(v["exact_with_EOS"]) for p in sides.values() for v in p.values()),
                "first_correct":sum(int(v["first_token_correct"]) for p in sides.values() for v in p.values()),
                "distinct_first_tokens":len(set(first)),
                "repetition_excess":repeats}

def train_arm(name, source_bytes, cfg, pairs, result_dir, tok):
    torch.manual_seed(SEED)
    archive=CB59(source_bytes)
    model,sources=create_small_paid_model(archive,cfg)
    model.eval()
    optimizer=torch.optim.Adam([x.centroids for x in sources.values()],lr=LR)
    target_names=sorted(sources.keys())
    history=[];progress=[]
    for step in range(STEPS+1):
        if step in STOPS:
            diag=snapshot(model,sources,pairs,tok)
            progress.append({"step":step,**diag})
            print(json.dumps({"stage":"CALIBRI2_S0_FOUR_PAIR_RETENTION",
                "arm":name,"step":step,**diag}),flush=True)
        if step==STEPS:break
        pair=pairs[step%len(pairs)]
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
        if (step+1)%4==0:
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
    out_file=result_dir/f"CALIBRI2_S0_{name}_384steps.cb59"
    out_file.write_bytes(out_bytes)
    n=archive.meta["original_unique_parameter_count"]
    bpp=8*len(out_bytes)/n
    if len(out_bytes)!=len(source_bytes) or bpp>.700:
        raise AssertionError("Full physical neural source grew beyond 0.700bpp")
    if hashlib.sha256(out_file.read_bytes()).hexdigest()!=change["sha256"]:
        raise AssertionError("Final source sha differs after physical export")
    reloaded=CB59(out_file.read_bytes())
    reload_model,reload_sources=create_small_paid_model(reloaded,cfg)
    reload_diag=snapshot(reload_model,reload_sources,pairs,tok)
    if reload_diag!={k:v for k,v in progress[-1].items() if k!="step"}:
        raise AssertionError("Reloaded CB59 differs from in-memory final TRAIN generation")
    result={
      "reload_verified":True,"arm":name,"initial_sha256":SHA0,"final_source_sha256":change["sha256"],
      "frozen_training_pairs":PAIR_IDS,"num_unique_params":n,
      "n_stored_paid_codebook_matrices":len(sources),
      "all_stored_parameters_counted":True,
      "physical_bytes":len(out_bytes),"physical_bpp":bpp,
      "initial_pretrained_weights_used":False,
      "QA_or_VALID_or_TEST_used":False,
      "four_pair_TRAIN_memorization_not_generalization_or_parity":True,
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
    all_pairs=build_pairs(tok,"TRAIN")
    pairs=[all_pairs[i] for i in PAIR_IDS]
    assert [p["domain"] for p in pairs]==["yesno_numeric","yesno_binding","arithmetic","binding"]
    assert all(p["split"]=="TRAIN" for p in pairs)
    cfg=build_config(next(c for c in CANDIDATES if c[0]=="small_320x6"))
    manifest=[{"pair_id":p["pair_id"],"domain":p["domain"],"split":p["split"],
               "sides":{side:{"question":p[side]["question"],"answer":p[side]["answer"],
                              "gold_tokens_with_EOS":p[side]["targets"].tolist()}
                        for side in ("left","right")}} for p in pairs]
    manifest_bytes=json.dumps(manifest,sort_keys=True,ensure_ascii=False).encode()
    save(args.out/"FROZEN_FOUR_PAIR_TRAIN_MANIFEST.json",
         {"sha256":hashlib.sha256(manifest_bytes).hexdigest(),"pairs":manifest})
    arms=[train_arm("centroids_and_5bit",raw,cfg,pairs,args.out,tok)]
    save(args.out/"ACTUAL_S0_FOUR_PAIR_RETENTION_384.json",{
      "purpose":"Frozen TRAIN-only four opposing pair retention with interleaved rehearsal",
      "steps_per_arm":384,"arms":arms,
      "no_pretrained_FP32":True,"no_public_DEV_VALID_QA_TEST":True,
      "not_generalization_or_FP32_parity":True})
    print(json.dumps({"stage":"CALIBRI2_S0_FOUR_PAIR_RETENTION_DONE",
       "arms":[{"arm":r["arm"],"final_full_correct":r["snapshots"][-1]["full_correct"],
       "physical_bpp":r["physical_bpp"],"sha256":r["final_source_sha256"]}
       for r in arms]}),flush=True)

if __name__=="__main__":
    main()
