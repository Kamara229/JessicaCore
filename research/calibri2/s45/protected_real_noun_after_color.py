"""CALIBRI S45 — physical CB59 global-noun recovery while preserving S44 color.

S44 fixed 31/64 free first token Yes to zero, and improved correct color
18→34/64, but no full 64 TRAIN color+noun+EOS answers. The shortest
distance to a correct new full answer is S44 frozen TRAIN pair 5006
(two free first tokens 'red', with noun ranks 20 and 134).
S45 targets actual global wrong noun in these TRAIN examples, not EOS
or a hardcoded output filter. Every trial must preserve the identities
of all 34 currently-correct first color responses, avoid new Yes first
tokens and retain exactly all 7 historical old full+EOS TRAIN skills.

Only 85 existing paid K32x8 FP16 centroid matrices learn; 5-bit indices
are fixed. Exact same 3,141,513 bytes (complete 0.690812875 bpp) are
saved and strict real physical reload/free-greedy outcomes verified.
No QA, no model selection on held-out data, no dense FP32 export,
no new sidecar, no Jessica chat tests, LIVE/Render/Supabase/APK changes.
This is an isolated TRAIN capacity experiment, not proof of FP32 parity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL,MODEL_SHA
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

SOURCE_SHA="c566d0cfaaa497c1ce0cb44ff8321765c226545c8417d9d5375e0011cdb188d0"
SEED=4545
TRAIN_INDICES=(6,2,6,5,1,6,3,7,6,5,2,6)
LR_CANDIDATES=(1.4,.5,.16,.045)
IMPROVE=0.006
YES_NO_FLOOR_CAP=.006

def noun_margin(model,sources,pair):
    return s41.pair_winner_margins(model,sources,pair)

def mean_margin(rows):
    return sum(float(r["margin_gold_minus_global_winner"]) for r in rows)/len(rows)

def min_margin(rows):
    return min(float(r["margin_gold_minus_global_winner"]) for r in rows)

@torch.inference_mode()
def color_integrity(model,sources,new,yes_id):
    # Preserve IDENTITIES of the 34 protected first-color TRAIN examples,
    # not merely their total (which could hide a regression).
    out={}
    for pair in new:
        for side in ("left","right"):
            row=pair[side]
            clear(sources)
            logits=model(input_ids=row["prompt_ids"][None],use_cache=False).logits[0,-1]
            winner=int(logits.argmax())
            out[f'{pair["pair_id"]}_{side}']={
                "correct":winner==int(row["targets"][0]),
                "starts_Yes":winner==yes_id,
                "winner_token_id":winner,
            }
    return out

def full_probe(model,sources,tok,new,old):
    new_rows={}
    ranks={}
    for pair in new:
        for side in ("left","right"):
            row=pair[side]
            name=f'{pair["pair_id"]}_{side}'
            new_rows[name]=s37.score_free(model,sources,tok,row)
            logits=s41.noun_row_logits(model,sources,row)
            target=int(row["targets"][1])
            ranks[name]=int((logits>logits[target]).sum())+1
    old_rows=s37.old_guard(model,sources,tok,old)
    return {
        "new_64_full_correct":sum(x["full_correct"] for x in new_rows.values()),
        "new_gold_noun_top1":sum(rank==1 for rank in ranks.values()),
        "new_gold_noun_top10":sum(rank<=10 for rank in ranks.values()),
        "new_gold_noun_median_rank":statistics.median(ranks.values()),
        "new_generated":new_rows,
        "new_noun_gold_ranks":ranks,
        "old_8_full_correct":sum(x["full_correct"] for x in old_rows.values()),
        "old_correct_ids":sorted(s37.preserved_ids(old_rows))
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=a.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA or len(raw)!=SOURCE_BYTES:
        raise AssertionError("S45 requires immutable S44 REAL physical CB59 source")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS or 8*len(raw)/SOURCE_PARAMS>.7:
        raise AssertionError("Source no longer within physical CB59 0.7 bpp contract")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new,old=s30.make_training_pairs(tok),build_pairs(tok,"TRAIN")
    if len(new)!=32 or len(old)!=24:
        raise AssertionError("Frozen TRAIN data identity changed")
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    keys=sorted(sources)
    paid=[sources[k].centroids for k in keys]
    if len(keys)!=PAID_OPERATORS or sum(x.numel() for x in paid)!=PAID_OPERATORS*32*8:
        raise AssertionError("Wrong paid dictionaries")
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=PAID_OPERATORS*32*8:
        raise AssertionError("Unpaid parameters cannot be trained")

    yes_id=int(tok.encode("Yes",add_special_tokens=False)[0])
    no_id=int(tok.encode("No",add_special_tokens=False)[0])
    old_start=s37.old_guard(model,sources,tok,old)
    old_ids=s37.preserved_ids(old_start)
    if len(old_ids)!=7 or "1_left" not in old_ids or "2_right" not in old_ids:
        raise AssertionError("S44 old-seven baseline not reproducible")
    before_color=color_integrity(model,sources,new,yes_id)
    protected_colors={k for k,v in before_color.items() if v["correct"]}
    if len(protected_colors)!=34 or any(x["starts_Yes"] for x in before_color.values()):
        raise AssertionError("S44 first-color 34 and Yes=0 guard not reproducible")
    old_yes_row=old[1]["left"]
    old_no_row=old[2]["right"]
    initial_yes=s37.first_margin(model,sources,old_yes_row,yes_id,no_id)
    initial_no=s37.first_margin(model,sources,old_no_row,no_id,yes_id)
    if initial_yes<=.01 or initial_no<=.01:
        raise AssertionError("S44 old Yes/No margins absent")
    yes_floor=min(YES_NO_FLOOR_CAP,.3*initial_yes)
    no_floor=min(YES_NO_FLOOR_CAP,.3*initial_no)
    before=full_probe(model,sources,tok,new,old)
    if before["new_64_full_correct"]!=0 or before["old_8_full_correct"]!=7:
        raise AssertionError("S44 initial full quality changed")

    proposals=[]
    accepted=0
    for step,index in enumerate(TRAIN_INDICES):
        pair=new[index]
        old_anchor=old[(step*5)%4]
        before_noun=noun_margin(model,sources,pair)
        before_mean=mean_margin(before_noun)
        before_min=min_margin(before_noun)
        _,primary,_=take_grads(model,sources,keys,lambda:s41.pair_winner_loss(model,sources,pair))
        _,old_grad,_=take_grads(model,sources,keys,lambda:s30.train_loss(model,old_anchor,False))
        _,yes_grad,_=take_grads(model,sources,keys,
            lambda:s37.margin_loss(model,sources,old_yes_row,yes_id,no_id))
        _,no_grad,_=take_grads(model,sources,keys,
            lambda:s37.margin_loss(model,sources,old_no_row,no_id,yes_id))
        direction,_=s37.normalize(primary)
        for g in (old_grad,yes_grad,no_grad):
            guard,_=s37.normalize(g)
            direction,_=s37.project_against(direction,guard)
        direction,_=s37.normalize(direction)
        saved=[p.detach().clone() for p in paid]
        selected=None
        attempts=[]
        for lr in LR_CANDIDATES:
            with torch.no_grad():
                for p,prior,grad in zip(paid,saved,direction):
                    p.copy_((prior-lr*grad).clamp(-16,16).half().float())
            after_noun=noun_margin(model,sources,pair)
            mean1=mean_margin(after_noun)
            min1=min_margin(after_noun)
            improved=(min1>before_min+IMPROVE and mean1>before_mean+IMPROVE)
            yes_after=no_after=None
            lost_old=lost_color=None
            yes_appeared=None
            keep=False
            if improved:
                yes_after=s37.first_margin(model,sources,old_yes_row,yes_id,no_id)
                no_after=s37.first_margin(model,sources,old_no_row,no_id,yes_id)
                if yes_after>=yes_floor and no_after>=no_floor:
                    colors=color_integrity(model,sources,new,yes_id)
                    lost_color=sorted(protected_colors-
                        {k for k,x in colors.items() if x["correct"]})
                    yes_appeared=sum(x["starts_Yes"] for x in colors.values())
                    if not lost_color and yes_appeared==0:
                        old_now=s37.preserved_ids(s37.old_guard(model,sources,tok,old))
                        lost_old=sorted(old_ids-old_now)
                        keep=not lost_old
            attempts.append({
                "learning_rate":lr,"noun_margin_min_before":before_min,
                "noun_margin_min_after":min1,
                "noun_margin_mean_before":before_mean,
                "noun_margin_mean_after":mean1,
                "both_noun_sides_improved":improved,
                "Yes_historical_margin":yes_after,"No_historical_margin":no_after,
                "lost_34_previously_correct_colors":lost_color,
                "free_first_Yes_top1_count":yes_appeared,
                "lost_7_historical_full_ids":lost_old,
                "accepted":keep,
            })
            if keep:
                selected=lr
                accepted+=1
                break
        if selected is None:
            with torch.no_grad():
                for p,orig in zip(paid,saved):
                    p.copy_(orig)
        clear(sources)
        proposals.append({"step":step+1,"TRAIN_pair":pair["pair_id"],
                          "selected_learning_rate":selected,
                          "attempts":attempts})
        print(json.dumps({"event":"S45_REAL_PAID_NOUN_TRAIN",
          "step":step+1,"accepted":accepted,"pair":pair["pair_id"],
          "selected":selected},ensure_ascii=False),flush=True)

    after=full_probe(model,sources,tok,new,old)
    final_color=color_integrity(model,sources,new,yes_id)
    final_correct={k for k,v in final_color.items() if v["correct"]}
    if not protected_colors.issubset(final_correct):
        raise AssertionError("S45 lost originally correct first-token color")
    if any(x["starts_Yes"] for x in final_color.values()):
        raise AssertionError("S45 reintroduced Yes-first collapse")
    if not old_ids.issubset(set(after["old_correct_ids"])):
        raise AssertionError("S45 regressed one of seven protected old full answers")
    yes_after=s37.first_margin(model,sources,old_yes_row,yes_id,no_id)
    no_after=s37.first_margin(model,sources,old_no_row,no_id,yes_id)
    if yes_after<yes_floor or no_after<no_floor:
        raise AssertionError("S45 regressed historical Yes/No margins")
    output=a.out/"CALIBRI2_S45_FIRST_COLOR_GUARDED_NOUN_PAID.cb59"
    physical=None
    reload=None
    if accepted:
        packed,physical=replace_discrete(archive,sources,strict=False)
        if len(packed)!=SOURCE_BYTES or physical["complete_bpp"]>.7:
            raise AssertionError("S45 physical size contract violation")
        if physical["actually_changed_FP16_codebook_cells"]<=0 or physical["actual_changed_5bit_group_indices"]!=0:
            raise AssertionError("No real paid FP16 update or illegal 5-bit change")
        if hashlib.sha256(packed).hexdigest()==SOURCE_SHA:
            raise AssertionError("S45 did not modify physical CB59")
        check,check_sources=create_small_paid_model(CB59(packed),build_config(MODEL_SPEC))
        reread=full_probe(check,check_sources,tok,new,old)
        reread_colors=color_integrity(check,check_sources,new,yes_id)
        for key in ("new_64_full_correct","new_gold_noun_top1",
                    "new_gold_noun_top10","old_8_full_correct","old_correct_ids",
                    "new_noun_gold_ranks"):
            if reread[key]!=after[key]:
                raise AssertionError("S45 compact reload changed "+key)
        for key in after["new_generated"]:
            if reread["new_generated"][key]["actual"]!=after["new_generated"][key]["actual"]:
                raise AssertionError("S45 exact new full generated response changed after reload")
        if reread_colors!=final_color:
            raise AssertionError("S45 first-color exact signatures changed after reload")
        output.write_bytes(packed)
        reload={"sha256":hashlib.sha256(packed).hexdigest(),
                "all_64_exact_full_and_noun_ranks_reloaded":True,
                "all_34_first_colors_and_old7_stable":True}

    report={
      "stage":"S45_REAL_PHYSICAL_CB59_GUARDED_NOUN_RECOVERY_TRAIN",
      "source_sha256":SOURCE_SHA,
      "original_unique_parameters":SOURCE_PARAMS,
      "physical_bytes":SOURCE_BYTES,
      "complete_physical_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
      "trainable_paid_FP16_codebook_cells":PAID_OPERATORS*32*8,
      "paid_5bit_indices_fixed":True,
      "only_TRAIN_for_training_and_guarding":True,
      "S29_QA_and_previously_inspected_S26_QA_not_used":True,
      "no_dense_FP32_export_no_green_sidecar":True,
      "no_Jessica_chat_test":True,"no_LIVE_change":True,
      "protected_initial_color_token_ids":sorted(protected_colors),
      "protected_historical_7_old_full_ids":sorted(old_ids),
      "historical_YesNo_margins_before":[initial_yes,initial_no],
      "historical_YesNo_margins_after":[yes_after,no_after],
      "before":before,"after":after,
      "first_color_correct_before":len(protected_colors),
      "first_color_correct_after":len(final_correct),
      "first_Yes_top1_after":sum(x["starts_Yes"] for x in final_color.values()),
      "gold_noun_top1_after":after["new_gold_noun_top1"],
      "new_full_64_after":after["new_64_full_correct"],
      "old_7_preserved":old_ids.issubset(set(after["old_correct_ids"])),
      "attempted_steps":len(TRAIN_INDICES),"accepted_steps":accepted,
      "all_candidates":proposals,
      "physical_rewrite":physical,"physical_reload":reload,
      "physical_cb59_saved":output.exists(),
      "unseen_FP32_parity_NOT_ESTABLISHED":True,
    }
    save(a.out/"S45_REAL_FIRST_COLOR_GUARDED_NOUN_TRAIN.json",report)
    print(json.dumps({"event":"S45_PAID_NOUN_RECOVERY_FINAL",
      "accepted":accepted,
      "noun_top1_before":before["new_gold_noun_top1"],
      "noun_top1_after":after["new_gold_noun_top1"],
      "full_new_before":before["new_64_full_correct"],
      "full_new_after":after["new_64_full_correct"],
      "first_color_34_guaranteed":True,
      "first_color_correct_after":len(final_correct),
      "old_7_preserved":True,
      "Yes_top1_after":report["first_Yes_top1_after"],
      "physical_bpp":report["complete_physical_bpp"],
      "physical_saved":output.exists()},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
