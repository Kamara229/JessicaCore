"""S43: why did 31/64 real S41 TRAIN free answers switch to Yes?

Read-only, before/after physical CB59 first-token and noun logits on the
same frozen S30 TRAIN prompts. No training/selection and no other data.
Never use S29 QA, inspected S26 QA, an uncompressed FP32 export or
Jessica chat tests. LIVE, Render, Supabase and APK are untouched.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
from v176.operators import clear
from v186.pairs import build_pairs
from v174.run import save
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s37 import paid_cb59_margin_protected_recovery as s37
from calibri2.s40.physical_noun_logits_S38_vs_S39 import physical_model

SHA_S39="92d87d6b2fce6871c98a99257ae752e4a777604a10e1be60bfca058cd61e5bba"
SHA_S41="a5a7582b407f7bea442ba5e49d3ef2ff6c6d1a1fad236e0898517c811b08ee8e"
SOURCE_NAMES=("S39","S41")

@torch.inference_mode()
def trace(model,sources,tok,new,old,stage):
    color_ids={c:int(tok.encode(c,add_special_tokens=False)[0])
               for c in ("red","green","blue","yellow")}
    yes_id=int(tok.encode("Yes",add_special_tokens=False)[0])
    no_id=int(tok.encode("No",add_special_tokens=False)[0])
    rows=[]
    for pair in new:
        for side in ("left","right"):
            row=pair[side]
            clear(sources)
            log=model(input_ids=row["prompt_ids"][None],use_cache=False).logits[0,-1]
            expected=int(row["targets"][0])
            actual=int(log.argmax())
            yes_margin=float(log[yes_id]-log[expected])
            color_scores={c:float(log[cid]) for c,cid in color_ids.items()}
            noun=expected
            noun_log=model(input_ids=row["ids"][None],use_cache=False).logits[0][row["positions"][1]]
            noun_gold=int(row["targets"][1])
            gen=s37.score_free(model,sources,tok,row)
            if gen["actual"]:
                gen_first=int(gen["actual"][0])
            else:
                gen_first=None
            rows.append({
                "id":f'{pair["pair_id"]}_{side}',"question":row["question"],
                "gold_answer":row["answer"],"gold_first_color":tok.decode([expected]),
                "gold_first_color_id":expected,"gold_first_color_rank":int((log>log[expected]).sum())+1,
                "global_first_winner_id":actual,"global_first_winner":tok.decode([actual]),
                "first_global_winner_logit_minus_gold":float(log[actual]-log[expected]),
                "Yes_minus_gold_color_logit":yes_margin,
                "Yes_minus_No_first_logit":float(log[yes_id]-log[no_id]),
                "color_logits":color_scores,
                "free_first_token_id":gen_first,
                "free_first_token_is_Yes":gen_first==yes_id,
                "full_free_token_ids":gen["actual"],
                "free_generated_text":gen["text"],"free_full_correct":gen["full_correct"],
                "teacher_forced_gold_noun_rank":int((noun_log>noun_log[noun_gold]).sum())+1
            })
    old_status=s37.old_guard(model,sources,tok,old)
    return {
      "stage":stage,
      "all_64_new_TRAIN":rows,
      "free_first_is_yes":sum(r["free_first_token_is_Yes"] for r in rows),
      "free_first_is_gold_color":sum(r["free_first_token_id"]==r["gold_first_color_id"] for r in rows),
      "first_gold_color_top1":sum(r["gold_first_color_rank"]==1 for r in rows),
      "first_yes_beats_gold_color":sum(r["Yes_minus_gold_color_logit"]>0 for r in rows),
      "free_full_correct":sum(r["free_full_correct"] for r in rows),
      "old_historical_8_FULL":sum(x["full_correct"] for x in old_status.values()),
      "old_historical_correct_ids":sorted(s37.preserved_ids(old_status)),
      "per_color_free_Yes_counts":{
         color:{"total":sum(r["gold_first_color"]==color for r in rows),
           "Yes":sum(r["gold_first_color"]==color and r["free_first_token_is_Yes"] for r in rows)}
         for color in ("red","green","blue","yellow")},
      "most_common_global_first_token":dict(Counter(r["global_first_winner"] for r in rows).most_common())
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("s39",type=Path)
    p.add_argument("s41",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    assert len(new)==32 and len(old)==24
    model39,sources39=physical_model(a.s39,SHA_S39)
    model41,sources41=physical_model(a.s41,SHA_S41)
    initial=trace(model39,sources39,tok,new,old,"S39")
    trained=trace(model41,sources41,tok,new,old,"S41")
    if [r["id"] for r in initial["all_64_new_TRAIN"]] != [r["id"] for r in trained["all_64_new_TRAIN"]]:
        raise AssertionError("S43 TRAIN prompt identities drifted")
    if initial["free_first_is_yes"]!=1 or trained["free_first_is_yes"]!=31:
        raise AssertionError("S42 documented free-first-Yes count not reproduced")
    if initial["old_historical_8_FULL"]!=7 or trained["old_historical_8_FULL"]!=7:
        raise AssertionError("Historical old-7 TRAIN skill signatures changed")
    if not set(initial["old_historical_correct_ids"]).issubset(set(trained["old_historical_correct_ids"])):
        raise AssertionError("Previously correct old TRAIN identities changed")
    changes=[]
    for before,after in zip(initial["all_64_new_TRAIN"],trained["all_64_new_TRAIN"]):
        changes.append({"id":before["id"],"gold":before["gold_answer"],
             "free_yes_S39":before["free_first_token_is_Yes"],
             "free_yes_S41":after["free_first_token_is_Yes"],
             "first_gold_color_rank_before":before["gold_first_color_rank"],
             "first_gold_color_rank_after":after["gold_first_color_rank"],
             "Yes_minus_gold_color_delta":after["Yes_minus_gold_color_logit"]-before["Yes_minus_gold_color_logit"],
             "gold_noun_rank_after":after["teacher_forced_gold_noun_rank"]})
    report={
      "stage":"S43_REAL_S39_S41_FREE_FIRST_YES_COLLAPSE_64_TRAIN",
      "source_SHA256":{"S39":SHA_S39,"S41":SHA_S41},
      "only_TRAIN":True,"no_training_or_model_selection":True,
      "S29_heldout_QA_and_S26_inspected_QA_never_used":True,
      "Jessica_ordinary_chat_test_NOT_RUN":True,
      "LIVE_unmodified":True,"source_CB59_paid_under_0_7_bpp":True,
      "before":initial,"after":trained,"per_pair_delta":changes,
      "diagnostic_goal":"Trace whether first-token Yes outranks gold color after training; avoid assuming teacher-forced noun rank explains free Yes",
      "no_new_model":True
    }
    save(a.out/"S43_S39_VS_S41_FIRST_YES_64_TRAIN.json",report)
    print(json.dumps({"event":"S43_FREE_FIRST_REAL_CB59_DIAG",
      "S39_first_yes":initial["free_first_is_yes"],
      "S41_first_yes":trained["free_first_is_yes"],
      "S39_color_top1":initial["first_gold_color_top1"],
      "S41_color_top1":trained["first_gold_color_top1"],
      "S39_yes_beats_gold":initial["first_yes_beats_gold_color"],
      "S41_yes_beats_gold":trained["first_yes_beats_gold_color"],
      "S41_by_color":trained["per_color_free_Yes_counts"],
      "old_7_preserved":True},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
