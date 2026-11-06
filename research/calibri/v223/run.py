"""CALIBRI v223: sealed v206/v220 divergence in second-token logits on TRAIN.

v221: all 48 first-token choices identical. v222: 12/48 generated
answers first differ on token 2; v220 introduced 5 adjacent repeats.
This script diagnoses the cause via actual complete-logit vectors at
second-token positions on exactly the original 48 opposing TRAIN prompts.
It is read-only; no DEV/VALID/private TEST or weight training, selection,
or deployment. FP32 used only as TRAIN teacher diagnostic.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from direct_codebook_v59 import CB59
from v84.sources import original_teacher
from v174.run import save
from v176.operators import clear
from v196 import protocol as P
from v196.data import training_inputs
from v196.recover import build_trainable_from_paid_source
from v212.run import SOURCE_V206_SHA256
from v221.run import V220_SOURCE_SHA256

@torch.inference_mode()
def logits_on(model, context):
    logits=model(input_ids=context,use_cache=False).logits[0,-1].float()
    if logits.shape!=(P.VOCAB_SIZE,) or not torch.isfinite(logits).all():
        raise AssertionError("TRAIN model output not a complete finite vocabulary")
    return logits.detach()

def summarize(logits, gold, first, tokenizer):
    winner=int(logits.argmax())
    if not (0<=gold<len(logits) and 0<=first<len(logits)):
        raise AssertionError("Invalid target or previous token")
    top=torch.topk(logits,5)
    is_gold=winner==gold
    best_not_gold=torch.topk(logits,2).values[1 if is_gold else 0]
    return {"winner_id":winner,
            "winner_token":tokenizer.decode([winner]),
            "gold_second_id":gold,
            "gold_second_token":tokenizer.decode([gold]),
            "gold_second_correct":is_gold,
            "gold_second_margin":float(logits[gold]-best_not_gold),
            "logp_gold":float(F.log_softmax(logits,dim=-1)[gold]),
            "repeat_first_token":winner==first,
            "logp_first_token":float(F.log_softmax(logits,dim=-1)[first]),
            "top5":[{"id":int(k),"token":tokenizer.decode([int(k)]),
                     "logit":float(v)} for k,v in zip(top.indices,top.values)]}

@torch.inference_mode()
def run(v206_path,v220_path,out):
    torch.manual_seed(223)
    torch.set_num_threads(2)
    sources={}
    for name,path,sha in (("v206",v206_path,SOURCE_V206_SHA256),
                           ("v220",v220_path,V220_SOURCE_SHA256)):
        raw=Path(path).read_bytes()
        if len(raw)!=P.SOURCE_BYTES or sha256(raw).hexdigest()!=sha:
            raise AssertionError("Unsealed or wrong physical model: "+name)
        model,paid=build_trainable_from_paid_source(CB59(raw))
        if len(paid)!=211:raise AssertionError("Original 211-codebook layout missing")
        model.eval()
        sources[name]=(model,paid)
    teacher=original_teacher().cpu().eval()
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    selected,pairs,valid_pairs,_=training_inputs(tok)
    if len(selected)!=20 or len(pairs)!=24 or len(valid_pairs)!=P.VALID_PAIRS:
        raise AssertionError("TRAIN-only input provenance drift")
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    save(out/"FROZEN_v223_before_second_token_diagnostic.json",{
        "experiment":"TRAIN_second_token_logit_divergence_v206_v220_FP32",
        "v206_physical_SHA256":SOURCE_V206_SHA256,
        "v220_physical_SHA256":V220_SOURCE_SHA256,
        "original_fp32_revision":MODEL_SHA,
        "TRAIN_opposing_prompts":48,"no_dev_valid_test":True,
        "no_fit_no_optimization_or_source_selection":True,
        "no_Jessica_LIVE_main_Render_Supabase_APK_changes":True})
    rows=[]
    for pair in pairs:
        if pair.get("split")!="TRAIN":raise AssertionError("Non-TRAIN case passed")
        for side in ("left","right"):
            sample=pair[side]
            prompt=sample["prompt_ids"].long()[None].detach()
            targets=sample["targets"].detach()
            if len(targets)<2:raise AssertionError("Incomplete frozen TRAIN gold")
            first_logits={}
            for name,(model,paid) in sources.items():
                clear(paid)
                first_logits[name]=logits_on(model,prompt)
                clear(paid)
            first_old=int(first_logits["v206"].argmax())
            first_new=int(first_logits["v220"].argmax())
            if first_old!=first_new:
                raise AssertionError("v221 first-token equivalence drift")
            context=torch.cat((prompt,prompt.new_tensor([[first_old]])),dim=1)
            gold=int(targets[1])
            probes={"FP32":summarize(logits_on(teacher,context),gold,first_old,tok)}
            for name,(model,paid) in sources.items():
                clear(paid)
                probes[name]=summarize(logits_on(model,context),gold,first_old,tok)
                clear(paid)
            a=probes["v206"]["winner_id"]
            b=probes["v220"]["winner_id"]
            # Different winners might reflect a narrow or wide logit gap.
            # Compare the exact two alternatives in each original distribution.
            pair_conf={}
            for name,(model,paid) in sources.items():
                clear(paid)
                v=logits_on(model,context)
                probs=F.log_softmax(v,dim=-1)
                pair_conf[name]={
                    "logp_v206_choice":float(probs[a]),
                    "logp_v220_choice":float(probs[b]),
                    "logit_v206_minus_v220_choice":float(v[a]-v[b])}
                clear(paid)
            rows.append({"split":"TRAIN","pair_id":pair["pair_id"],
                         "side":side,"domain":pair["domain"],
                         "same_first_token_id":first_old,
                         "gold_first_token_id":int(targets[0]),
                         "gold_second_token_id":gold,
                         "second_token_changed":a!=b,
                         "teacher_and_packed_second_token_probes":probes,
                         "two_competitor_packed_logit_contrast":pair_conf})
        print(json.dumps({"stage":"v223_second_token_margin_progress",
                          "pairs_completed":len(rows)//2}),flush=True)
    if len(rows)!=48 or len({(x["pair_id"],x["side"]) for x in rows})!=48:
        raise AssertionError("Partial original 48 TRAIN pair-side prompts")
    changed=[x for x in rows if x["second_token_changed"]]
    if len(changed)!=12:
        raise AssertionError("v222 sealed TRAIN second-token divergence count drift")
    stats={
        "confirmed_48_original_TRAIN_same_first_choice":True,
        "different_second_token_choices":len(changed),
        "v206_second_gold_correct":sum(x["teacher_and_packed_second_token_probes"]["v206"]["gold_second_correct"] for x in rows),
        "v220_second_gold_correct":sum(x["teacher_and_packed_second_token_probes"]["v220"]["gold_second_correct"] for x in rows),
        "FP32_second_gold_correct_on_v206_prefix":sum(x["teacher_and_packed_second_token_probes"]["FP32"]["gold_second_correct"] for x in rows),
        "v206_second_repeat_first_token":sum(x["teacher_and_packed_second_token_probes"]["v206"]["repeat_first_token"] for x in rows),
        "v220_second_repeat_first_token":sum(x["teacher_and_packed_second_token_probes"]["v220"]["repeat_first_token"] for x in rows),
        "changed_PROMPTS_gold_and_repeat":[{
          "pair_id":x["pair_id"],"side":x["side"],"domain":x["domain"],
          "FP32_second":x["teacher_and_packed_second_token_probes"]["FP32"]["winner_token"],
          "v206_second":x["teacher_and_packed_second_token_probes"]["v206"]["winner_token"],
          "v220_second":x["teacher_and_packed_second_token_probes"]["v220"]["winner_token"],
          "v206_gold_margin":x["teacher_and_packed_second_token_probes"]["v206"]["gold_second_margin"],
          "v220_gold_margin":x["teacher_and_packed_second_token_probes"]["v220"]["gold_second_margin"],
          "v220_repeat_first":x["teacher_and_packed_second_token_probes"]["v220"]["repeat_first_token"]}
          for x in changed],
        "NOT_a_new_independent_quality_test":True,
        "DEV16_v220_loss_not_assumed_to_share_TRAIN_cause":True}
    save(out/"ACTUAL_v223_TRAIN_second_token_full_logit_diagnostic.json",
         {"completed":True,"source_is_original_sealed":True,
          "all_48_TRAIN_rows":rows,"summary":stats,
          "not_training_or_selecting_weights":True,
          "Jessica_tests_ran":False})
    print(json.dumps({"stage":"v223_TRAIN_second_token_logit_summary",
                      "changed":len(changed),
                      "second_gold_v206":stats["v206_second_gold_correct"],
                      "second_gold_v220":stats["v220_second_gold_correct"],
                      "second_repeats_v206":stats["v206_second_repeat_first_token"],
                      "second_repeats_v220":stats["v220_second_repeat_first_token"]}),
          flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("v206",type=Path)
    ap.add_argument("v220",type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    run(a.v206,a.v220,a.out)
