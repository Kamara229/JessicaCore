"""S38: physically compressed 36M CB59 balanced YES and NO protected learning.

Data-driven continuation of sealed S37 compact archive (0.690812875 bpp).
S37's 9/12 rejected compact updates lost historical TRAIN skill 2_right,
which must answer 'No'. Do not blindly maximize 'Yes' for 1_left!
Project compact dictionary gradients against *both* opposite-answer
TRAIN contexts, then, if a candidate damages either, locally correct
that specific context's margin in paid codebook space (FP16-snap).
Strict all-six original historical TRAIN exact+EOS guard is final arbiter.

32 different S30 TRAIN-only new color/noun pairs, 32 attempted steps.
No S29 independent QA, no previously inspected S26 QA, no TEST/VALID.
No original 135M teacher, no dense FP32 checkpoint or hidden sidecar.
No Jessica ordinary-chat test, no LIVE/Render/Supabase/APK modifications.
A successful TRAIN trial is NOT generalization or FP32 quality parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from direct_codebook_v59 import CB59
from v176.operators import clear
from v185.codec import replace_discrete
from v186.pairs import build_pairs
from v174.run import save
from calibri2.s0.bootstrap import build_config
from calibri2.s0.train import create_small_paid_model
from calibri2.s8.broad_replay import SOURCE_PARAMS, SOURCE_BYTES, MODEL_SPEC, PAID_OPERATORS
from calibri2.s20.multiword_curriculum import take_grads
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s34 import continued_protected_color_noun_fp32 as s34
from calibri2.s37 import paid_cb59_margin_protected_recovery as s37

SOURCE_SHA = "e7f8b549eebb117039e2953de6270c605b05b27e6e45fe44c882d68b0f829a0a"
STEPS = 32
SEED = 3838
LR_CANDIDATES = (0.11, 0.035, 0.008)
YES_BLEND = 0.12
NO_BLEND = 0.36
OLD_BLEND = 0.13
MAX_REPAIR_PASSES_PER_CONTEXT = 2
MARGIN_FLOOR_CAP = 0.015
IMPROVE = 1e-7


def check_and_score(model, sources, tok, pair, protected, old, yesrow,
                    norow, yes_id, no_id, floor_yes, floor_no,
                    loss_before, repair=False, params=None):
    """Keep new TRAIN gain and every historical exact+EOS answer."""
    after = s37.paid_loss(model, sources, tok, pair)
    if after + IMPROVE >= loss_before:
        return {"new_loss":after,"improved":False,"safe":False,
                "lost_protected":None,"after_margin_yes":None,
                "after_margin_no":None,"repairs":[]}
    yesmargin = s37.first_margin(model,sources,yesrow,yes_id,no_id)
    nomargin = s37.first_margin(model,sources,norow,no_id,yes_id)
    trace = []
    if repair and (nomargin < floor_no or yesmargin < floor_yes):
        for context,row,gold,foil,floor in (
            ("2_right_No",norow,no_id,yes_id,floor_no),
            ("1_left_Yes",yesrow,yes_id,no_id,floor_yes),
        ):
            for i in range(MAX_REPAIR_PASSES_PER_CONTEXT):
                pre = s37.first_margin(model,sources,row,gold,foil)
                if pre >= floor:
                    break
                clear(sources)
                model.zero_grad(set_to_none=True)
                objective = s37.margin_loss(model,sources,row,gold,foil)
                objective.backward()
                grad = [
                    p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                    for p in params
                ]
                magnitude = float(sum(g.double().square().sum() for g in grad))
                if not np.isfinite(magnitude) or magnitude <= 1e-10:
                    raise AssertionError("S38 invalid paid margin repair gradient")
                # Constraint projection on the ACTUAL 21,760 paid codebook cells.
                factor = min(0.15, 1.25*(floor-pre)/magnitude)
                with torch.no_grad():
                    for p,g in zip(params,grad):
                        p.copy_((p-factor*g).clamp(-16,16).half().float())
                clear(sources)
                post = s37.first_margin(model,sources,row,gold,foil)
                trace.append({"context":context,"attempt":i+1,
                              "before":pre,"after":post,"lr_projection":factor})
                del grad
                if post >= floor:
                    break
        yesmargin = s37.first_margin(model,sources,yesrow,yes_id,no_id)
        nomargin = s37.first_margin(model,sources,norow,no_id,yes_id)
        after = s37.paid_loss(model,sources,tok,pair)
    if (after+IMPROVE>=loss_before or
        yesmargin<floor_yes or nomargin<floor_no):
        return {"new_loss":after,"improved":after+IMPROVE<loss_before,
                "safe":False,"lost_protected":None,
                "after_margin_yes":yesmargin,"after_margin_no":nomargin,
                "repairs":trace}
    old_results = s37.old_guard(model,sources,tok,old)
    retained = s37.preserved_ids(old_results)
    return {"new_loss":after,"improved":True,
            "safe":protected.issubset(retained),
            "lost_protected":sorted(protected-retained),
            "after_margin_yes":yesmargin,"after_margin_no":nomargin,
            "repairs":trace}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    raw = a.source.read_bytes()
    if len(raw)!=SOURCE_BYTES or hashlib.sha256(raw).hexdigest()!=SOURCE_SHA:
        raise AssertionError("S38 requires exact new S37 physically compact 36M CB59")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Unexpected 36M unique parameters")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    assert len(new)==32 and len(old)==24
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    names=sorted(sources)
    params=[sources[k].centroids for k in names]
    assert len(params)==PAID_OPERATORS
    assert sum(p.numel() for p in params)==PAID_OPERATORS*32*8
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==len(params)*32*8
    old_before=s37.old_guard(model,sources,tok,old)
    protected=s37.preserved_ids(old_before)
    if len(protected)!=6 or "2_right" not in protected or "1_left" not in protected:
        raise AssertionError("S38 original six identities changed from sealed S37")
    before=s37.small_train_probe(model,sources,tok,new,old)
    yesrow=old[1]["left"]
    norow=old[2]["right"]
    assert yesrow["answer"]=="Yes" and norow["answer"]=="No"
    yes_id=tok.encode("Yes",add_special_tokens=False)[0]
    no_id=tok.encode("No",add_special_tokens=False)[0]
    start_yes=s37.first_margin(model,sources,yesrow,yes_id,no_id)
    start_no=s37.first_margin(model,sources,norow,no_id,yes_id)
    if start_yes<=0 or start_no<=0:
        raise AssertionError("Sealed S37 must answer both Yes and No before S38")
    floor_yes=min(MARGIN_FLOOR_CAP,0.4*start_yes)
    floor_no=min(MARGIN_FLOOR_CAP,0.4*start_no)
    decisions=[]
    accepts=0
    for step in range(STEPS):
        pair=new[(step*7)%len(new)]
        oldpair=old[(step*5)%4]
        baseline=s37.paid_loss(model,sources,tok,pair)
        _,g,_=take_grads(model,sources,names,
              lambda:s34.loss_s34(model,"new",pair,tok))
        _,h,_=take_grads(model,sources,names,
              lambda:s30.train_loss(model,oldpair,False))
        _,gy,_=take_grads(model,sources,names,
              lambda:s37.margin_loss(model,sources,yesrow,yes_id,no_id))
        _,gn,_=take_grads(model,sources,names,
              lambda:s37.margin_loss(model,sources,norow,no_id,yes_id))
        primary,prim_norm=s37.normalize(g)
        anchor,oldnorm=s37.normalize(h)
        yesvec,yesnorm=s37.normalize(gy)
        novec,nonorm=s37.normalize(gn)
        direction,pc_old=s37.project_against(primary,anchor)
        direction,pc_yes=s37.project_against(direction,yesvec)
        direction,pc_no=s37.project_against(direction,novec)
        direction=[x + OLD_BLEND*h + YES_BLEND*y + NO_BLEND*n
                   for x,h,y,n in zip(direction,anchor,yesvec,novec)]
        direction,totalnorm=s37.normalize(direction)
        saved=[p.detach().clone() for p in params]
        trial=[]
        chosen=None
        for lr in LR_CANDIDATES:
            with torch.no_grad():
                for p,b,d in zip(params,saved,direction):
                    p.copy_((b-lr*d).clamp(-16,16).half().float())
            clear(sources)
            first=check_and_score(
                model,sources,tok,pair,protected,old,
                yesrow,norow,yes_id,no_id,
                floor_yes,floor_no,baseline,
                repair=False,params=params)
            candidate=first
            if not first["safe"] and first["improved"]:
                candidate=check_and_score(
                    model,sources,tok,pair,protected,old,
                    yesrow,norow,yes_id,no_id,
                    floor_yes,floor_no,baseline,
                    repair=True,params=params)
            trial.append({"lr":lr,
                          "before_repair":first,
                          "after_repair":candidate})
            if candidate["safe"]:
                chosen=lr
                accepts+=1
                break
        if chosen is None:
            with torch.no_grad():
                for p,b in zip(params,saved):
                    p.copy_(b)
        clear(sources)
        decisions.append({"step":step+1,"new_TRAIN_pair":pair["pair_id"],
                          "historical_TRAIN_anchor":oldpair["pair_id"],
                          "accepted":chosen is not None,
                          "chosen_lr":chosen,
                          "source_loss_before":baseline,
                          "gradient_norms":{"task":prim_norm,
                                            "old":oldnorm,
                                            "Yes":yesnorm,"No":nonorm},
                          "projection_coefs":{
                              "old":pc_old,"Yes":pc_yes,"No":pc_no},
                          "candidate_trials":trial})
        if step==0 or (step+1)%4==0:
            print(json.dumps({"event":"S38_PHYSICAL_BALANCED_TRAIN",
                  "step":step+1,"accepted":accepts,"last_accepted_lr":chosen},
                  ensure_ascii=False),flush=True)

    after=s37.small_train_probe(model,sources,tok,new,old)
    last=s37.preserved_ids(after["historical_TRAIN"]["all_full_answers"])
    if not protected.issubset(last):
        raise AssertionError("S38 final lost original protected historical TRAIN skill")
    end_yes=s37.first_margin(model,sources,yesrow,yes_id,no_id)
    end_no=s37.first_margin(model,sources,norow,no_id,yes_id)
    if end_yes<floor_yes or end_no<floor_no:
        raise AssertionError("S38 final protected contextual Yes/No margins declined")

    physical=None
    reloaded=None
    out_file=a.out/"CALIBRI2_S38_36M_PAID_TRAIN.cb59"
    if accepts:
        packed,physical=replace_discrete(archive,sources,strict=False)
        if len(packed)!=SOURCE_BYTES or physical["complete_bpp"]>.7:
            raise AssertionError("S38 illegal physical footprint")
        if physical["actual_changed_5bit_group_indices"]!=0:
            raise AssertionError("S38 did not authorize index changes")
        if physical["actually_changed_FP16_codebook_cells"]<=0:
            raise AssertionError("No real new compact codebook cells")
        if hashlib.sha256(packed).hexdigest()==SOURCE_SHA:
            raise AssertionError("Accepted steps did not change physical checkpoint")
        paid,reloaded_sources=create_small_paid_model(CB59(packed),build_config(MODEL_SPEC))
        repeated=s37.small_train_probe(paid,reloaded_sources,tok,new,old)
        same=repeated==after
        if not same:
            raise AssertionError("CB59 reload not exact on full TRAIN generations")
        repeated_yes=s37.first_margin(paid,reloaded_sources,yesrow,yes_id,no_id)
        repeated_no=s37.first_margin(paid,reloaded_sources,norow,no_id,yes_id)
        if abs(repeated_yes-end_yes)>0.003 or abs(repeated_no-end_no)>0.003:
            raise AssertionError("Reload changed first token margins")
        out_file.write_bytes(packed)
        reloaded={"identical_full_TRAIN_responses":same,
                  "margin_Yes":repeated_yes,
                  "margin_No":repeated_no,
                  "sha256":hashlib.sha256(packed).hexdigest()}
    report={
      "stage":"S38_36M_PAID_CB59_BALANCED_YES_NO_MARGIN_TRAIN",
      "S37_source_SHA256":SOURCE_SHA,
      "S37_original_bytes":len(raw),
      "original_parameters":SOURCE_PARAMS,
      "full_physical_bpp":8*len(raw)/SOURCE_PARAMS,
      "all_85_compact_codebooks_trained":True,
      "fivebit_indices_frozen":True,
      "no_dense_FP32_weights_exported":True,
      "no_hidden_green_tunnel_weight_sidecar":True,
      "QA_VALID_TEST_NEVER_USED_OR_READ":True,
      "Jessica_chat_test_never_run":True,
      "LIVE_Render_Supabase_APK_unchanged":True,
      "old_protected_six_TRAIN_ids":sorted(protected),
      "balanced_Yes_No_TRAIN_id":["1_left","2_right"],
      "initial_margin_1left_Yes":start_yes,
      "initial_margin_2right_No":start_no,
      "required_margin_floor_Yes":floor_yes,
      "required_margin_floor_No":floor_no,
      "final_margin_1left_Yes":end_yes,
      "final_margin_2right_No":end_no,
      "accepted_steps":accepts,
      "attempted_steps":STEPS,
      "before_greedy_TRAIN":before,
      "after_greedy_TRAIN":after,
      "all_32_candidate_trials":decisions,
      "physical_new_checkpoint":physical,
      "reloaded_greedy_parity":reloaded,
      "checkpoint_created":out_file.exists(),
      "quality_not_yet_independent":True,
    }
    save(a.out/"S38_ACTUAL_PHYSICAL_BALANCED_YES_NO_TRAIN.json",report)
    print(json.dumps({
      "event":"S38_PHYSICAL_BALANCED_FINAL",
      "accepted":accepts,"attempted":STEPS,
      "new_full_before":before["new_TRAIN"]["fully_correct"],
      "new_full_after":after["new_TRAIN"]["fully_correct"],
      "old_full_before":before["historical_TRAIN"]["fully_correct"],
      "old_full_after":after["historical_TRAIN"]["fully_correct"],
      "paid_file_created":out_file.exists(),
      "bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
      "S29_qa_used":False},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
