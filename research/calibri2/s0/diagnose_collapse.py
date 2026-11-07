"""CALIBRI 2.0 S0: read-only token-choice collapse diagnosis on frozen TRAIN.

Compare initial, step16,32,48,64 CB59 checkpoints. Measure complete logits
at answer start, gold-vs-opposite gap, answer distribution, rank/probability
and prompt-side sensitivity. No training, no QA/VALID/TEST, no FP32 teacher,
no new neural weights, no changes to LIVE. Do not select model by these scores.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch
from transformers import AutoTokenizer
from direct_codebook_v59 import CB59
from binary_geometry_v64 import MODEL,MODEL_SHA
from v174.run import save
from v176.operators import clear
from v186.pairs import build_pairs
from calibri2.s0.bootstrap import CANDIDATES,build_config
from calibri2.s0.train import create_small_paid_model,PROBE_PAIR_IDS

EXPECTED_SOURCE="2e807eb9c4f82018bcc72f0924726377cafb70df1ad529aefbd8513b6e538d80"
EXPECTED_FINISH="3a87686b629f2fd3d19a3ac4ee028a970969dfb88e5b7870b53fec5b1edc49d2"
EXPECTED_PARAMS=22368320
EXACT_STEPS=(0,16,32,48,64)
MAX_BPP=0.700

def inspect_checkpoint(step,path,cfg,pairs,tok):
    buf=Path(path).read_bytes()
    sha=hashlib.sha256(buf).hexdigest()
    if (step==0 and sha!=EXPECTED_SOURCE) or (step==64 and sha!=EXPECTED_FINISH):
        raise AssertionError("Frozen initial/final checkpoint digest mismatch")
    cb=CB59(buf)
    n=cb.meta["original_unique_parameter_count"]
    if n!=EXPECTED_PARAMS or 8*len(buf)/n>MAX_BPP:
        raise AssertionError("Checkpoint not the original small 22M compact source")
    student,sources=create_small_paid_model(cb,cfg)
    student.eval()
    result=[]
    try:
        with torch.inference_mode():
            for pair in pairs:
                if pair["split"]!="TRAIN" or pair["pair_id"] not in PROBE_PAIR_IDS:
                    raise AssertionError("Probes only from frozen already-used TRAIN split")
                item={"pair_id":pair["pair_id"],"domain":pair["domain"],
                      "opposing_first_gold_token_ids":{
                        side:int(pair[side]["first_expected_id"])
                        for side in ("left","right")}}
                for side,opposite in (("left","right"),("right","left")):
                    prompt=pair[side]["prompt_ids"].reshape(1,-1)
                    gold=int(pair[side]["first_expected_id"])
                    opp=int(pair[opposite]["first_expected_id"])
                    clear(sources)
                    logits=student(input_ids=prompt,use_cache=False).logits[0,-1]
                    if logits.numel()!=cfg.vocab_size or not torch.isfinite(logits).all():
                        raise AssertionError("Malformed full original vocab logits")
                    log_p=torch.log_softmax(logits,dim=-1)
                    top_logp,top_ids=log_p.topk(5)
                    prediction=int(logits.argmax())
                    gold_logit=float(logits[gold])
                    margin=float(logits[gold]-logits[opp])
                    rank=1+int((logits>logits[gold]).sum())
                    p_gold=float(log_p[gold].exp())
                    p_pred=float(log_p[prediction].exp())
                    entropy=float(-(log_p.exp()*log_p).sum())
                    eos=int(tok.eos_token_id)
                    item[side]={
                        "expected_first_token_id":gold,
                        "opposing_expected_token_id":opp,
                        "greedy_token_id":prediction,
                        "greedy_text":tok.decode([prediction]),
                        "greedy_logprob":float(log_p[prediction]),
                        "greedy_probability":p_pred,
                        "gold_probability":p_gold,
                        "gold_rank_all_vocab":rank,
                        "gold_logit_minus_opposite_logit":margin,
                        "gold_over_opposite":margin>0,
                        "EOS_probability":float(log_p[eos].exp()),
                        "entropy_nats":entropy,
                        "top5_tokens":[{"id":int(i),"text":tok.decode([int(i)]),
                                        "probability":float(l.exp())}
                                       for l,i in zip(top_logp,top_ids)],
                    }
                item["different_greedy_tokens_across_opposite_pair"]=(
                    item["left"]["greedy_token_id"]!=item["right"]["greedy_token_id"])
                result.append(item)
    finally:
        clear(sources)
        del student,sources
    side_items=[r[side] for r in result for side in ("left","right")]
    hist=Counter(x["greedy_token_id"] for x in side_items)
    all_yes_no=[x for r in result if r["domain"].startswith("yesno")
                for x in (r["left"],r["right"])]
    summary={
        "step":step,"complete_physical_bytes":len(buf),"source_sha256":sha,
        "bits_per_original_parameter":8*len(buf)/n,
        "total_frozen_TRAIN_prompt_sides":len(side_items),
        "greedy_first_token_diversity":len(hist),
        "greedy_token_frequency":[{"id":i,"text":tok.decode([i]),"count":count}
                                  for i,count in hist.most_common()],
        "first_gold_top1_correct":sum(x["greedy_token_id"]==x["expected_first_token_id"]
                                     for x in side_items),
        "first_gold_top5":sum(x["gold_rank_all_vocab"]<=5 for x in side_items),
        "gold_rank_median":statistics.median(x["gold_rank_all_vocab"] for x in side_items),
        "gold_prob_mean":statistics.mean(x["gold_probability"] for x in side_items),
        "top1_prob_mean":statistics.mean(x["greedy_probability"] for x in side_items),
        "entropy_nats_mean":statistics.mean(x["entropy_nats"] for x in side_items),
        "yesno_greedy_correct":sum(
            x["greedy_token_id"]==x["expected_first_token_id"] for x in all_yes_no),
        "yesno_gold_over_opposite_token":sum(x["gold_over_opposite"] for x in all_yes_no),
        "opposite_pairs_with_distinct_greedy_tokens":sum(
            r["different_greedy_tokens_across_opposite_pair"] for r in result),
        "first_gold_margin_mean":statistics.mean(
             x["gold_logit_minus_opposite_logit"] for x in side_items),
        "no_heldout_QA_TEST_used":True,
        "no_model_training_or_selection":True,
        "full_prediction_records":result
    }
    return summary

def main():
    p=argparse.ArgumentParser()
    p.add_argument("initial_archive",type=Path)
    p.add_argument("train_checkpoints",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(2021)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    all_pairs=build_pairs(tok,"TRAIN")
    selected=[all_pairs[i] for i in PROBE_PAIR_IDS]
    if len(selected)!=8 or len({p["pair_id"] for p in selected})!=8:
        raise AssertionError("Frozen repeated TRAIN probe identity drift")
    cfg=build_config(next(s for s in CANDIDATES if s[0]=="small_320x6"))
    paths={0:a.initial_archive}
    for step in EXACT_STEPS[1:]:
        paths[step]=a.train_checkpoints/("CALIBRI2_S0_step%03d.cb59"%step)
    snapshots=[]
    for step in EXACT_STEPS:
        snap=inspect_checkpoint(step,paths[step],cfg,selected,tok)
        snapshots.append(snap)
        print(json.dumps({
            "stage":"CALIBRI2_S0_ACTUAL_TRAIN_LOGITS_DIAG",
            "step":step,
            "first_token_correct":snap["first_gold_top1_correct"],
            "yesno_greedy_correct":snap["yesno_greedy_correct"],
            "yesno_pairwise_gold_above_opposite":snap["yesno_gold_over_opposite_token"],
            "distinct_greedy_tokens":snap["greedy_first_token_diversity"],
            "distinct_opposite_pairs":snap["opposite_pairs_with_distinct_greedy_tokens"],
            "entropy_nats":round(snap["entropy_nats_mean"],5),
            "prediction_hist":snap["greedy_token_frequency"]
        },ensure_ascii=False),flush=True)
    summary={
      "research":"CALIBRI2 S0 token collapse checkpoint diagnosis",
      "checkpoint_steps":list(EXACT_STEPS),
      "fixed_original_architecture":"small_320x6",
      "frozen_TRAIN_probe_pair_ids":list(PROBE_PAIR_IDS),
      "initial_source_sha256":EXPECTED_SOURCE,
      "trained_source_sha256":EXPECTED_FINISH,
      "all_original_5bit_architecture_matrices":43,
      "trained_neural_weights_originally_pretrained":False,
      "no_teacher_or_QA_VALID_TEST_access":True,
      "read_only_no_checkpoint_selected_by_this_diagnostic":True,
      "trajectory":[{k:v for k,v in s.items() if k!="full_prediction_records"}
                     for s in snapshots],
      "observed_real_first_token_logit_records":snapshots,
    }
    save(a.out/"ACTUAL_CALIBRI2_S0_LOGIT_COLLAPSE_TRAIN_DIAG.json",summary)
    if snapshots[-1]["greedy_first_token_diversity"]>1:
        print("CAUTION latest record no longer shows universal one-token collapse",
              flush=True)
    print(json.dumps({"stage":"CALIBRI2_S0_READ_ONLY_DIAG_SUCCESS",
        "checkpoints":len(snapshots),"physical_cap_held":all(
         s["bits_per_original_parameter"]<=.7 for s in snapshots),
        "no_training_or_heldout_QA":True}),flush=True)

if __name__=="__main__":
    main()
