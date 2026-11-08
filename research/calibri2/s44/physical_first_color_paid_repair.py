"""CALIBRI S44: recover actual first-color token in paid 36M CB59, TRAIN only.

S43 measured S41 31/64 unconstrained Yes starts, and just 18/64 correct
first colors (S39 was 44/64). Improve actual first-token gold-vs-global
wrong winner margins, not a hard-coded output filter or forced decoding.

Same exact S41 3,141,513-byte 0.6908 bpp physical checkpoint. Optimize only
previously paid 85 K32x8 FP16 centroids; all 5-bit indices frozen. Accept
updates only with both contrasts' color-margin improvements, paired noun
margin retention, seven historically correct full answers plus Yes/No
margins. Check all 64 free full outputs after real serialization/reload.

Research GitHub branch only; no S29 QA, inspected S26 QA, Jessica chat tests,
dense FP32 checkpoint, additional sidecar, LIVE/Render/Supabase/APK changes.
TRAIN changes are NOT evidence of unseen FP32 parity or generality.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.codec import replace_discrete
from v186.pairs import build_pairs
from v174.run import save
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS,SOURCE_BYTES,MODEL_SPEC,PAID_OPERATORS
from calibri2.s20.multiword_curriculum import take_grads
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s37 import paid_cb59_margin_protected_recovery as s37
from calibri2.s41 import global_winner_noun_paid_centroids as s41
from calibri2.s43 import physical_first_token_yes_collapse as s43

SOURCE_SHA="a5a7582b407f7bea442ba5e49d3ef2ff6c6d1a1fad236e0898517c811b08ee8e"
SEED=4444
PAIR_INDICES=(24,8,25,9,26,10,27,11,28,12,29,13)
LR_CANDIDATES=(.65,.24,.08,.022)
COLOR_IMPROVE=.008
NOUN_GUARD_EPS=.02
MAX_FREE_ANSWER_TOKENS=10

def color_logits(model,sources,row):
    clear(sources)
    return model(input_ids=row["prompt_ids"][None],use_cache=False).logits[0,-1]

def color_gold_vs_winner(logits,target):
    alt=logits.detach().clone()
    alt[target]=-torch.inf
    wrong=int(alt.argmax())
    return logits[target]-logits[wrong],wrong

def color_loss(model,sources,pair):
    parts=[]
    for side in ("left","right"):
        row=pair[side]
        logits=color_logits(model,sources,row)
        gold=int(row["targets"][0])
        margin,wrong=color_gold_vs_winner(logits,gold)
        parts.append(F.softplus(1.0-margin))
    return torch.stack(parts).mean()

@torch.inference_mode()
def margins(model,sources,pair,yes_token):
    result=[]
    for side in ("left","right"):
        row=pair[side]
        logits=color_logits(model,sources,row)
        gold=int(row["targets"][0])
        margin,wrong=color_gold_vs_winner(logits,gold)
        result.append({"side":side,"gold_id":gold,
                       "wrong_global_id":wrong,
                       "gold_minus_global_wrong_margin":float(margin),
                       "gold_minus_Yes_margin":float(logits[gold]-logits[yes_token]),
                       "first_color_top1":int(logits.argmax())==gold})
    return result

def mean_color_margin(rows):
    return sum(r["gold_minus_global_wrong_margin"] for r in rows)/2.

def min_color_margin(rows):
    return min(r["gold_minus_global_wrong_margin"] for r in rows)

def mean_noun_margin(rows):
    return sum(r["margin_gold_minus_global_winner"] for r in rows)/2.

def noun_mean_rank(rows):
    return sum(r["gold_rank"] for r in rows)/2.

def snapshot_summary(model,sources,tok,new,old):
    trace=s43.trace(model,sources,tok,new,old,"S44")
    gold_noun_ranks=[r["teacher_forced_gold_noun_rank"] for r in trace["all_64_new_TRAIN"]]
    trace["gold_noun_median_rank"]=statistics.median(gold_noun_ranks)
    trace["gold_noun_top1"]=sum(x==1 for x in gold_noun_ranks)
    return trace

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA or len(raw)!=SOURCE_BYTES:
        raise AssertionError("S44 requires exact immutable S41 physically paid CB59")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS or 8*len(raw)/SOURCE_PARAMS>.7:
        raise AssertionError("S44 paid size/architecture changed")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    if len(new)!=32 or len(old)!=24:
        raise AssertionError("S44 TRAIN selection identities changed")
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    keys=sorted(sources)
    params=[sources[k].centroids for k in keys]
    if len(keys)!=PAID_OPERATORS or sum(p.numel() for p in params)!=PAID_OPERATORS*32*8:
        raise AssertionError("S44 illegal change to physical paid codebook")
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=PAID_OPERATORS*32*8:
        raise AssertionError("S44 illegally trainable dense weights")

    old_initial=s37.old_guard(model,sources,tok,old)
    old_ids=s37.preserved_ids(old_initial)
    if len(old_ids)!=7 or not {"1_left","2_right"}.issubset(old_ids):
        raise AssertionError("S44 S41 old-seven history absent")
    yes_id=tok.encode("Yes",add_special_tokens=False)[0]
    no_id=tok.encode("No",add_special_tokens=False)[0]
    yesrow,norow=old[1]["left"],old[2]["right"]
    y0=s37.first_margin(model,sources,yesrow,yes_id,no_id)
    n0=s37.first_margin(model,sources,norow,no_id,yes_id)
    if y0<=.010 or n0<=.010:
        raise AssertionError("S44 historical Yes/No initial margins absent")
    yfloor=min(.008,y0*.35)
    nfloor=min(.008,n0*.35)
    before=snapshot_summary(model,sources,tok,new,old)
    if before["free_first_is_yes"]!=31 or before["first_gold_color_top1"]!=18:
        raise AssertionError("S43 baseline was not reproducible from S41 physical checkpoint")
    if before["old_historical_8_FULL"]!=7 or before["free_full_correct"]!=0:
        raise AssertionError("S41 strict full-answer baseline changed")

    log=[]
    accepted=0
    changed_operators=set()
    for step,which in enumerate(PAIR_INDICES):
        pair=new[which]
        old_anchor=old[(step*5)%4]
        initial=margins(model,sources,pair,yes_id)
        initial_noun=s41.pair_winner_margins(model,sources,pair)
        old_m=mean_color_margin(initial)
        old_min=min_color_margin(initial)
        old_noun_margin=mean_noun_margin(initial_noun)
        _,g,_=take_grads(model,sources,keys,lambda:color_loss(model,sources,pair))
        _,guard_old,_=take_grads(model,sources,keys,lambda:s30.train_loss(model,old_anchor,False))
        _,guard_yes,_=take_grads(model,sources,keys,lambda:s37.margin_loss(model,sources,yesrow,yes_id,no_id))
        _,guard_no,_=take_grads(model,sources,keys,lambda:s37.margin_loss(model,sources,norow,no_id,yes_id))
        direction,_=s37.normalize(g)
        for grad in (guard_old,guard_yes,guard_no):
            guard,_=s37.normalize(grad)
            direction,_=s37.project_against(direction,guard)
        direction,_=s37.normalize(direction)
        preserved=[p.detach().clone() for p in params]
        trials=[]
        chosen=None
        for lr in LR_CANDIDATES:
            with torch.no_grad():
                for p,base,grad in zip(params,preserved,direction):
                    p.copy_((base-lr*grad).clamp(-16,16).half().float())
            candidate=margins(model,sources,pair,yes_id)
            noun=s41.pair_winner_margins(model,sources,pair)
            cm=mean_color_margin(candidate)
            cmin=min_color_margin(candidate)
            nm=mean_noun_margin(noun)
            improved=cm>old_m+COLOR_IMPROVE and cmin>old_min+COLOR_IMPROVE
            noun_retained=nm>=old_noun_margin-NOUN_GUARD_EPS
            accepted_now=False
            lost=None
            yes_now=None
            no_now=None
            if improved and noun_retained:
                yes_now=s37.first_margin(model,sources,yesrow,yes_id,no_id)
                no_now=s37.first_margin(model,sources,norow,no_id,yes_id)
                if yes_now>=yfloor and no_now>=nfloor:
                    kept=s37.preserved_ids(s37.old_guard(model,sources,tok,old))
                    lost=sorted(old_ids-kept)
                    accepted_now=not lost
            trials.append({
               "lr":lr,"color_mean_after":cm,"color_min_after":cmin,
               "nouns_mean_after":nm,
               "color_both_sides_improved":improved,
               "new_noun_margin_retained":noun_retained,
               "Yes_margin_after":yes_now,"No_margin_after":no_now,
               "lost_historical_TRAIN_ids":lost,"accepted":accepted_now
            })
            if accepted_now:
                chosen=lr
                accepted+=1
                changed_operators.update(k for k,p,b in zip(keys,params,preserved) if not torch.equal(p,b))
                break
        if chosen is None:
            with torch.no_grad():
                for p,base in zip(params,preserved):
                    p.copy_(base)
        clear(sources)
        log.append({"step":step+1,"TRAIN_pair":pair["pair_id"],
                   "before_color":initial,"before_noun":initial_noun,
                   "chosen_lr":chosen,"trials":trials})
        print(json.dumps({"event":"S44_PHYSICAL_FIRST_COLOR_TRAIN",
          "step":step+1,"accepted":accepted,
          "pair":pair["pair_id"],"chosen_lr":chosen},ensure_ascii=False),flush=True)

    after=snapshot_summary(model,sources,tok,new,old)
    if not old_ids.issubset(set(after["old_historical_correct_ids"])):
        raise AssertionError("S44 regressed one of seven historical full answers")
    y1=s37.first_margin(model,sources,yesrow,yes_id,no_id)
    n1=s37.first_margin(model,sources,norow,no_id,yes_id)
    if y1<yfloor or n1<nfloor:
        raise AssertionError("S44 lost protected historical Yes/No margin")
    output=args.out/"CALIBRI2_S44_PAID_FIRST_COLOR_REPAIR.cb59"
    physical=None
    reload_info=None
    if accepted:
        packed,physical=replace_discrete(archive,sources,strict=False)
        if len(packed)!=SOURCE_BYTES or physical["complete_bpp"]>.7:
            raise AssertionError("S44 physical bpp or bytes violation")
        if physical["actually_changed_FP16_codebook_cells"]<=0:
            raise AssertionError("S44 no paid FP16 training persisted")
        if hashlib.sha256(packed).hexdigest()==SOURCE_SHA:
            raise AssertionError("S44 candidate checkpoint identical to source")
        compact_model,compact_src=create_small_paid_model(CB59(packed),build_config(MODEL_SPEC))
        loaded=snapshot_summary(compact_model,compact_src,tok,new,old)
        for scalar in ("free_first_is_yes","first_gold_color_top1","free_full_correct","old_historical_8_FULL","old_historical_correct_ids","gold_noun_top1"):
            if loaded[scalar]!=after[scalar]:
                raise AssertionError("S44 physically reloaded CB59 changed "+scalar)
        for item_a,item_b in zip(after["all_64_new_TRAIN"],loaded["all_64_new_TRAIN"]):
            for field in ("id","free_first_token_id","full_free_token_ids","teacher_forced_gold_noun_rank"):
                if item_a[field]!=item_b[field]:
                    raise AssertionError("S44 physical CB59 reload mismatched "+field)
        output.write_bytes(packed)
        reload_info={"sha256":hashlib.sha256(packed).hexdigest(),
                     "all_64_free_full_and_noun_ranks_equal":True,
                     "all_7_historical_full_correct":True}

    report={
      "stage":"S44_PHYSICAL_PAID_36M_FIRST_COLOR_REPAIR_TRAIN_ONLY",
      "source_S41_SHA256":SOURCE_SHA,
      "physical_bytes":SOURCE_BYTES,
      "original_unique_parameters":SOURCE_PARAMS,
      "complete_physical_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
      "trainable_only_paid_FP16_centroid_cells":PAID_OPERATORS*32*8,
      "paid_5bit_indices_unchanged":True,
      "S29_QA_and_inspected_S26_QA_never_used":True,
      "only_TRAIN":True,"Jessica_chat_tests_NOT_RUN":True,"LIVE_untouched":True,
      "before":before,"after":after,
      "first_color_top1_before":before["first_gold_color_top1"],
      "first_color_top1_after":after["first_gold_color_top1"],
      "free_answers_start_Yes_before":before["free_first_is_yes"],
      "free_answers_start_Yes_after":after["free_first_is_yes"],
      "new_full_greedy_answers_before":before["free_full_correct"],
      "new_full_greedy_answers_after":after["free_full_correct"],
      "old_seven_initial_exact_ids":sorted(old_ids),
      "old_seven_final_exact_ids":after["old_historical_correct_ids"],
      "Yes_No_initial_margins":[y0,n0],
      "Yes_No_final_margins":[y1,n1],
      "accepted_steps":accepted,"attempted_steps":len(PAIR_INDICES),
      "changed_paid_dictionary_names":sorted(changed_operators),
      "candidate_trials":log,"physical_rewrite":physical,
      "reloaded_physical_CHECK":reload_info,
      "new_CB59_saved":output.exists(),
      "unseen_FP32_quality_parity_NOT_ESTABLISHED":True,
    }
    save(args.out/"S44_PHYSICAL_PAID_FIRST_COLOR_TRAIN_ONLY.json",report)
    print(json.dumps({"event":"S44_REAL_FIRST_COLOR_TRAIN_DONE",
      "accepted":accepted,
      "color_top1_before":report["first_color_top1_before"],
      "color_top1_after":report["first_color_top1_after"],
      "Yes_first_before":report["free_answers_start_Yes_before"],
      "Yes_first_after":report["free_answers_start_Yes_after"],
      "new_full_after":report["new_full_greedy_answers_after"],
      "old7_protected":old_ids.issubset(set(after["old_historical_correct_ids"])),
      "bpp":report["complete_physical_bpp"],
      "new_physical_CB59":output.exists()},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
