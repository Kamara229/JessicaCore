"""S41: target the ACTUAL wrong noun winning over the gold noun in real CB59.

S40 proved training pairs have gold-noun top-1 = 0/64 with gold rank
typically > 20,000 and bag/book/lamp instead of bowl/drum/rope/ring.
S39 only improved gold-vs-opposite rare noun, never gold-vs-GLOBAL
highest-scoring wrong token. S41 computes that global-winner margin
inside the physical paid centroid-gradient graph and accepts steps
only when BOTH contrast sides improve, all seven previously correct
historical TRAIN full responses survive, Yes and No remain protected.

The baseline is exact SHA-pinned S39 36M CB59. Only paid K32x8 85
FP16 codebooks (21,760 values) change; 5-bit indices fixed. No dense
model export, new sidecar, extra bits or reference FP32 checkpoint.
Only S30 TRAIN prompt pairs, no S29 immutable QA or inspected S26 QA.
Jessica chat is not a test runner. LIVE, Render, Supabase, APK untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
from torch.nn import functional as F
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

SOURCE_SHA="92d87d6b2fce6871c98a99257ae752e4a777604a10e1be60bfca058cd61e5bba"
SEED=4141
STEPS=20
LR_CANDIDATES=(1.0,.40,.12,.035)
IMPROVE=0.008
MARGIN_FLOOR=.008

def noun_row_logits(model,sources,row):
    clear(sources)
    if len(row["positions"])!=3 or len(row["targets"])!=3:
        raise AssertionError("TRAIN nouns must be a single token with final EOS")
    return model(input_ids=row["ids"][None],use_cache=False).logits[0][row["positions"][1]]

def gold_vs_real_winner(logits, gold_id):
    # Select strongest GLOBAL wrong token; do not cherry-pick just the foil noun.
    alternative=logits.detach().clone()
    alternative[gold_id]=-torch.inf
    wrong=int(alternative.argmax())
    return logits[gold_id]-logits[wrong],wrong

def pair_winner_loss(model,sources,pair):
    objectives=[]
    for side in ("left","right"):
        row=pair[side]
        logits=noun_row_logits(model,sources,row)
        gold=int(row["targets"][1])
        margin,_=gold_vs_real_winner(logits,gold)
        objectives.append(F.softplus(1.0-margin))
    return torch.stack(objectives).mean()

@torch.inference_mode()
def pair_winner_margins(model,sources,pair):
    out=[]
    for side in ("left","right"):
        row=pair[side]
        log=noun_row_logits(model,sources,row)
        gold=int(row["targets"][1])
        diff,wrong=gold_vs_real_winner(log,gold)
        rank=1+int((log>log[gold]).sum())
        out.append({"side":side,"gold_id":gold,"wrong_winner_id":wrong,
                    "margin_gold_minus_global_winner":float(diff),
                    "gold_rank":rank,"gold_top1":rank==1})
    return out

def measured_pair_margin(rows):
    return min(v["margin_gold_minus_global_winner"] for v in rows)

def measured_pair_mean_margin(rows):
    return sum(v["margin_gold_minus_global_winner"] for v in rows)/2

@torch.inference_mode()
def summarize_train(model,sources,tok,new,old):
    rows=[]
    for pair in new:
        for side in ("left","right"):
            row=pair[side]
            l=noun_row_logits(model,sources,row)
            gold=int(row["targets"][1])
            margin,winner=gold_vs_real_winner(l,gold)
            seq=s37.score_free(model,sources,tok,row)
            rows.append({"id":str(pair["pair_id"])+"_"+side,
              "gold":row["answer"],
              "predicted_winner_token":tok.decode([winner]),
              "noun_gold_rank":int(1+(l>l[gold]).sum()),
              "noun_gold_minus_global_winner_logit":float(margin),
              "full_correct_with_eos":seq["full_correct"],
              "generated_text":seq["text"]})
    old_results=s37.old_guard(model,sources,tok,old)
    return {"new_64_TRAIN_full_answers":sum(r["full_correct_with_eos"] for r in rows),
            "new_64_TRAIN_noun_top1":sum(r["noun_gold_rank"]==1 for r in rows),
            "new_64_TRAIN_gold_noun_median_rank":statistics.median(r["noun_gold_rank"] for r in rows),
            "new_64_TRAIN_gold_global_winner_mean_margin":statistics.mean(r["noun_gold_minus_global_winner_logit"] for r in rows),
            "historical_8_TRAIN_full_answers":sum(r["full_correct"] for r in old_results.values()),
            "historical_TRAIN_full_ids":sorted(s37.preserved_ids(old_results)),
            "all_64_new_TRAIN":rows}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("source",type=Path)
    a.add_argument("out",type=Path)
    args=a.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw=args.source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA or len(raw)!=SOURCE_BYTES:
        raise AssertionError("S41 must start from the EXACT verified S39 paid CB59")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Physical source architecture mismatch")
    if 8*len(raw)/SOURCE_PARAMS>0.7:
        raise AssertionError("S41 complete source exceeds 0.7 bpp")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    if len(new)!=32 or len(old)!=24:
        raise AssertionError("TRAIN identities changed")
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    keys=sorted(sources)
    paid=[sources[k].centroids for k in keys]
    if len(keys)!=PAID_OPERATORS or sum(p.numel() for p in paid)!=PAID_OPERATORS*32*8:
        raise AssertionError("Unexpected paid centroid capacity")
    if sum(p.numel() for p in model.parameters() if p.requires_grad)!=PAID_OPERATORS*32*8:
        raise AssertionError("Unpaid dense training is forbidden")

    baseline=s37.small_train_probe(model,sources,tok,new,old)
    old_protected=s37.preserved_ids(baseline["historical_TRAIN"]["all_full_answers"])
    if len(old_protected)!=7 or "1_left" not in old_protected or "2_right" not in old_protected:
        raise AssertionError("Old S39 exact-protection baseline absent")
    if baseline["new_TRAIN"]["fully_correct"]!=0:
        raise AssertionError("Unexpected S39 new TRAIN capability baseline")
    yes_id=tok.encode("Yes",add_special_tokens=False)[0]
    no_id=tok.encode("No",add_special_tokens=False)[0]
    yes_row,no_row=old[1]["left"],old[2]["right"]
    yes0=s37.first_margin(model,sources,yes_row,yes_id,no_id)
    no0=s37.first_margin(model,sources,no_row,no_id,yes_id)
    if yes0<.015 or no0<.015:
        raise AssertionError("S39 protected Yes/No margins unexpectedly fragile")
    yes_min=min(.010, .35*yes0)
    no_min=min(.010, .35*no0)
    before=summarize_train(model,sources,tok,new,old)
    if before["new_64_TRAIN_noun_top1"]!=0 or before["new_64_TRAIN_full_answers"]!=0:
        raise AssertionError("S40 observed 0/64 noun and 0/64 full changed")

    decisions=[]
    accepted=0
    altered=set()
    for step in range(STEPS):
        pair=new[(step*11)%len(new)]
        prior=pair_winner_margins(model,sources,pair)
        worst0=measured_pair_margin(prior)
        mean0=measured_pair_mean_margin(prior)
        old_anchor=old[(step*5)%4]
        _,g,new_matrix=take_grads(model,sources,keys,
                lambda:pair_winner_loss(model,sources,pair))
        _,g_old,_=take_grads(model,sources,keys,
                lambda:s30.train_loss(model,old_anchor,False))
        _,g_yes,_=take_grads(model,sources,keys,
                lambda:s37.margin_loss(model,sources,yes_row,yes_id,no_id))
        _,g_no,_=take_grads(model,sources,keys,
                lambda:s37.margin_loss(model,sources,no_row,no_id,yes_id))
        direction,_=s37.normalize(g)
        keep_old,_=s37.normalize(g_old)
        keep_yes,_=s37.normalize(g_yes)
        keep_no,_=s37.normalize(g_no)
        for guard in (keep_old,keep_yes,keep_no):
            direction,_=s37.project_against(direction,guard)
        direction,_=s37.normalize(direction)
        snapshot=[p.detach().clone() for p in paid]
        chosen=None
        trials=[]
        for lr in LR_CANDIDATES:
            with torch.no_grad():
                for p,base,grad in zip(paid,snapshot,direction):
                    p.copy_((base-lr*grad).clamp(-16,16).half().float())
            candidate=pair_winner_margins(model,sources,pair)
            worst1=measured_pair_margin(candidate)
            mean1=measured_pair_mean_margin(candidate)
            ok_delta=(worst1>=worst0+IMPROVE and mean1>=mean0+IMPROVE)
            current_yes=None
            current_no=None
            remaining=None
            keep=False
            if ok_delta:
                current_yes=s37.first_margin(model,sources,yes_row,yes_id,no_id)
                current_no=s37.first_margin(model,sources,no_row,no_id,yes_id)
                if current_yes>=yes_min and current_no>=no_min:
                    preserved=s37.preserved_ids(s37.old_guard(model,sources,tok,old))
                    remaining=sorted(old_protected-preserved)
                    keep=not remaining
            trials.append({"lr":lr,"before_worst_noun_margin":worst0,
                "after_worst_noun_margin":worst1,
                "after_mean_noun_margin":mean1,
                "both_sides_improve":ok_delta,
                "yes_margin":current_yes,"no_margin":current_no,
                "lost_historical_TRAIN_ids":remaining,"accepted":keep})
            if keep:
                chosen=lr
                accepted+=1
                altered.update(k for k,p,prev in zip(keys,paid,snapshot) if not torch.equal(p,prev))
                break
        if chosen is None:
            with torch.no_grad():
                for p,prev in zip(paid,snapshot):
                    p.copy_(prev)
        clear(sources)
        decisions.append({"step":step+1,"TRAIN_pair":pair["pair_id"],
             "chosen_lr":chosen,"before":prior,"candidates":trials})
        print(json.dumps({"event":"S41_REAL_PAID_GLOBAL_NOUN_PROGRESS",
             "step":step+1,"accepted_steps":accepted,
             "TRAIN_pair":pair["pair_id"],"chosen_lr":chosen},ensure_ascii=False),flush=True)

    after=summarize_train(model,sources,tok,new,old)
    if not old_protected.issubset(set(after["historical_TRAIN_full_ids"])):
        raise AssertionError("S41 historic seven old TRAIN full answers regressed")
    yes1=s37.first_margin(model,sources,yes_row,yes_id,no_id)
    no1=s37.first_margin(model,sources,no_row,no_id,yes_id)
    if yes1<yes_min or no1<no_min:
        raise AssertionError("S41 protected historic Yes/No token margins fell")
    output=args.out/"CALIBRI2_S41_36M_GLOBAL_NOUN_PAID.cb59"
    reload=None
    physical=None
    if accepted:
        packed,physical=replace_discrete(archive,sources,strict=False)
        if len(packed)!=SOURCE_BYTES or physical["complete_bpp"]>0.7:
            raise AssertionError("S41 physical model exceeds 0.7 bpp")
        if physical["actually_changed_FP16_codebook_cells"]==0:
            raise AssertionError("S41 accepted invisible packed changes")
        if hashlib.sha256(packed).hexdigest()==SOURCE_SHA:
            raise AssertionError("S41 accepted identical CB59")
        model2,source2=create_small_paid_model(CB59(packed),build_config(MODEL_SPEC))
        after_reload=summarize_train(model2,source2,tok,new,old)
        if after_reload!=after:
            raise AssertionError("S41 physical reload disagrees on all 64 exact full answers/ranks")
        reload={"sha256":hashlib.sha256(packed).hexdigest(),
                "full_64_new_and_8_old_TRAIN_identical":True,
                "physical_model_bytes":len(packed)}
        output.write_bytes(packed)

    report={
      "stage":"S41_REAL_CB59_PAID_CENTROID_GLOBAL_NOUN_WINNER_RECOVERY",
      "source_S39_SHA256":SOURCE_SHA,
      "unique_original_params":SOURCE_PARAMS,
      "physical_bytes":SOURCE_BYTES,"physical_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
      "paid_FP16_centroids_only":True,"paid_5bit_indices_FROZEN":True,
      "no_unpaid_tensors_or_FP32_sidecar":True,
      "S29_immutable_QA_and_S26_previously_used_QA_UNTOUCHED":True,
      "only_TRAIN_for_all_learning_selection_and_reporting":True,
      "Jessica_ordinary_chat_TEST_NOT_RUN":True,"LIVE_UNMODIFIED":True,
      "S40_original_global_noun_rank_was_far_from_top1":True,
      "precommitted_steps":STEPS,"accepted_steps":accepted,
      "changed_paid_operator_dictionaries":sorted(altered),
      "historical_seven_protected":sorted(old_protected),
      "Yes_No_margins":{"start_yes":yes0,"end_yes":yes1,
                        "start_no":no0,"end_no":no1},
      "before":before,"after":after,
      "all_attempted_candidate_steps":decisions,
      "physical_rewrite":physical,"physical_reload":reload,
      "new_CB59_checkpoint_created":output.exists(),
      "TRAIN_only_not_evidence_for_FP32_parity_or_unseen_generalization":True,
    }
    save(args.out/"S41_REAL_PAID_GLOBAL_NOUN_WINNER_TRAIN_ONLY.json",report)
    print(json.dumps({"event":"S41_REAL_PAID_GLOBAL_NOUN_FINAL",
       "accepted":accepted,
       "before_global_noun_top1":before["new_64_TRAIN_noun_top1"],
       "after_global_noun_top1":after["new_64_TRAIN_noun_top1"],
       "before_full_new_64":before["new_64_TRAIN_full_answers"],
       "after_full_new_64":after["new_64_TRAIN_full_answers"],
       "protected_historical_TRAIN":after["historical_8_TRAIN_full_answers"],
       "bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
       "physical_created":output.exists()},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
