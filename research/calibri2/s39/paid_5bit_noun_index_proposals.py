"""S39 — train actual 5-bit indices, not an expanded 36M FP32 checkpoint.

Start from SHA-pinned S38 physical 0.6908bpp CB59 weights. S37/S38 changed
paid FP16 centroids but achieved 0/16 on new TRAIN color+noun full answers.
Here centroids and row scales remain FROZEN. Propose top gradient-ranked
5-bit group changes in selected paid matrices (including tied embedding/
head) and accept only if a two-word noun contrast improves, new TRAIN
objective strictly improves, all seven historically correct old TRAIN
full-answer+EOS identities stay correct, and old Yes and No TRAIN margins
remain positive. Mutations are reversible in-memory until final sealing.

This is TRAIN only; no S29 QA/VALID/TEST, no inspected S26 QA, no
unpaid Green Tunnel sidecar, no dense original weight export. Neither
TRAIN success nor native speed is evidence of unseen FP32 quality parity.
No Jessica chat test, no LIVE/Render/Supabase/APK changes.
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
from v185.optim import propose_codes, apply_codes
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

SOURCE_SHA="6b74f67b4d2324e6e0a1d99bdf525b8fdda81f03f008069ef10aedbcd26c8b34"
SEED=3939
STEPS=24
MAX_SWAPS=128
BATCH_SIZES=(8,32,128)
IMPROVE=1e-7
NOUN_MARGIN_IMPROVE=0.001


@torch.inference_mode()
def noun_foil_margin(model,sources,pair):
    """The smaller gold-vs-opposite-noun logit margin of both TRAIN sides."""
    clear(sources)
    left,right=pair["left"],pair["right"]
    if len(left["targets"])!=3 or len(right["targets"])!=3:
        raise AssertionError("Expected one color, one noun, EOS")
    if int(left["targets"][0]) != int(right["targets"][0]):
        raise AssertionError("TRAIN contrast changed color")
    if int(left["targets"][1]) == int(right["targets"][1]):
        raise AssertionError("TRAIN contrast changed noun")
    llog=model(input_ids=left["ids"][None],use_cache=False).logits[0][left["positions"][1]]
    clear(sources)
    rlog=model(input_ids=right["ids"][None],use_cache=False).logits[0][right["positions"][1]]
    a=int(left["targets"][1]);b=int(right["targets"][1])
    return min(float(llog[a]-llog[b]),float(rlog[b]-rlog[a]))


def subset_proposal(proposal,n):
    indices=proposal["indices"][:n]
    return {"indices":indices,
            "new_codes":proposal["new_codes"][:n],
            "old_codes":proposal["old_codes"][:n],
            "proposed_groups":len(indices)}


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
        raise AssertionError("Not exact immutable S38 CB59 source")
    archive=CB59(raw)
    if archive.meta["original_unique_parameter_count"]!=SOURCE_PARAMS:
        raise AssertionError("Wrong S39 original unique 36M parameter count")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    new=s30.make_training_pairs(tok)
    old=build_pairs(tok,"TRAIN")
    assert len(new)==32 and len(old)==24
    model,sources=create_small_paid_model(archive,build_config(MODEL_SPEC))
    keys=sorted(sources)
    if len(keys)!=PAID_OPERATORS or "model.embed_tokens.weight" not in keys:
        raise AssertionError("Original 85 paid dictionary names missing")
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable!=PAID_OPERATORS*32*8:
        raise AssertionError("Illegal trainable non-codebook weights")
    # No centroid optimizer is used or constructed; only 5-bit groups change.
    baseline=s37.small_train_probe(model,sources,tok,new,old)
    initial_old=s37.preserved_ids(baseline["historical_TRAIN"]["all_full_answers"])
    if len(initial_old)!=7 or "1_left" not in initial_old or "2_right" not in initial_old:
        raise AssertionError("S38 old TRAIN skill baseline changed")
    yesrow=old[1]["left"]
    norow=old[2]["right"]
    yes_id=tok.encode("Yes",add_special_tokens=False)[0]
    no_id=tok.encode("No",add_special_tokens=False)[0]
    margin_yes=s37.first_margin(model,sources,yesrow,yes_id,no_id)
    margin_no=s37.first_margin(model,sources,norow,no_id,yes_id)
    if margin_yes<0.02 or margin_no<0.02:
        raise AssertionError("S38 protected historical TRAIN margins absent")
    threshold_yes=min(0.015,0.3*margin_yes)
    threshold_no=min(0.015,0.3*margin_no)
    direct_name="model.embed_tokens.weight"
    # Rotate learning through tied embedding and 23 independently paid
    # attention/MLP operators. The named order is frozen by sorted keys.
    operators=[direct_name]+[name for name in keys if name!=direct_name][:STEPS-1]
    if len(operators)!=STEPS:
        raise AssertionError("S39 declared 24 matrix coverage changed")
    records=[]
    accepted=0
    groups_accepted=0
    for step,name in enumerate(operators):
        pair=new[(step*7)%len(new)]
        source=sources[name]
        base_loss=s37.paid_loss(model,sources,tok,pair)
        base_noun=noun_foil_margin(model,sources,pair)
        clear(sources)
        with torch.inference_mode():
            old_weight=source().detach().cpu().numpy().copy()
        # Matrix reconstruction gradients are TEMPORARY for index proposal.
        # The only persisted model updates are actual paid 5-bit indices.
        _,_,matrix_grad=take_grads(
            model,sources,keys,
            lambda:s34.loss_s34(model,"new",pair,tok),target_name=name)
        if matrix_grad is None or matrix_grad.shape!=old_weight.shape:
            raise AssertionError("Missing actual paid matrix gradient")
        proposal=propose_codes(source,matrix_grad,old_weight)
        if proposal["proposed_groups"]>512:
            raise AssertionError("S39 exceeded preexisting index proposer cap")
        trials=[]
        chosen=None
        for count in BATCH_SIZES:
            n=min(count,MAX_SWAPS,proposal["proposed_groups"])
            if n<=0 or any(v["swaps"]==n for v in trials):
                continue
            part=subset_proposal(proposal,n)
            with torch.no_grad():
                apply_codes(source,part,True)
            clear(sources)
            loss_after=s37.paid_loss(model,sources,tok,pair)
            noun_after=noun_foil_margin(model,sources,pair)
            improved=loss_after+IMPROVE<base_loss
            better_noun=noun_after>=base_noun+NOUN_MARGIN_IMPROVE
            keep=False
            lost=None
            yes_after=None
            no_after=None
            if improved and better_noun:
                yes_after=s37.first_margin(model,sources,yesrow,yes_id,no_id)
                no_after=s37.first_margin(model,sources,norow,no_id,yes_id)
                if yes_after>=threshold_yes and no_after>=threshold_no:
                    retained=s37.preserved_ids(s37.old_guard(model,sources,tok,old))
                    lost=sorted(initial_old-retained)
                    keep=not lost
            trials.append({
                "swaps":n,"new_loss":loss_after,"noun_margin_after":noun_after,
                "new_loss_improved":improved,"noun_margin_improved":better_noun,
                "Yes_margin":yes_after,"No_margin":no_after,
                "lost_original_old_TRAIN_ids":lost,"accepted":keep,
            })
            with torch.no_grad():
                apply_codes(source,part,False)
            clear(sources)
            if keep:
                chosen=part
                break
        if chosen is not None:
            with torch.no_grad():
                apply_codes(source,chosen,True)
            clear(sources)
            accepted+=1
            groups_accepted+=chosen["proposed_groups"]
        records.append({
            "step":step+1,"operator":name,"TRAIN_pair_id":pair["pair_id"],
            "proposed_5bit_groups":proposal["proposed_groups"],
            "TRAIN_primary_before":base_loss,
            "noun_gold_foil_margin_before":base_noun,
            "all_group_batch_candidates":trials,
            "accepted":chosen is not None,
            "actual_accepted_5bit_groups":0 if chosen is None else chosen["proposed_groups"],
        })
        print(json.dumps({"event":"S39_REAL_PAID_5BIT_TRAIN_PROGRESS",
                          "step":step+1,"accepted":accepted,
                          "accepted_groups":groups_accepted,
                          "operator":name},ensure_ascii=False),flush=True)
    after=s37.small_train_probe(model,sources,tok,new,old)
    final_old=s37.preserved_ids(after["historical_TRAIN"]["all_full_answers"])
    if not initial_old.issubset(final_old):
        raise AssertionError("S39 lost original seven historically correct old TRAIN answers")
    final_yes=s37.first_margin(model,sources,yesrow,yes_id,no_id)
    final_no=s37.first_margin(model,sources,norow,no_id,yes_id)
    if final_yes<threshold_yes or final_no<threshold_no:
        raise AssertionError("S39 lost original Yes/No margins")
    physical=None
    reload=None
    output=a.out/"CALIBRI2_S39_36M_INDEX_TRAIN_PAID.cb59"
    if accepted:
        packed,physical=replace_discrete(archive,sources,strict=False)
        if len(packed)!=SOURCE_BYTES or physical["complete_bpp"]>0.7:
            raise AssertionError("S39 new CB59 does not meet physical 0.7bpp contract")
        if physical["actual_changed_5bit_group_indices"]<=0:
            raise AssertionError("S39 did not physically change any indices")
        if physical["actually_changed_FP16_codebook_cells"]!=0:
            raise AssertionError("S39 changed forbidden paid FP16 dictionaries")
        if hashlib.sha256(packed).hexdigest()==SOURCE_SHA:
            raise AssertionError("S39 accepted unchanged new physical model")
        reload_model,reload_sources=create_small_paid_model(CB59(packed),build_config(MODEL_SPEC))
        repeated=s37.small_train_probe(reload_model,reload_sources,tok,new,old)
        if repeated!=after:
            raise AssertionError("S39 reloaded true compact weights changed strict full generation")
        reload_yes=s37.first_margin(reload_model,reload_sources,yesrow,yes_id,no_id)
        reload_no=s37.first_margin(reload_model,reload_sources,norow,no_id,yes_id)
        if abs(reload_yes-final_yes)>0.003 or abs(reload_no-final_no)>0.003:
            raise AssertionError("S39 reload changed historical Yes/No logits")
        output.write_bytes(packed)
        reload={"sha256":hashlib.sha256(packed).hexdigest(),
                "all_full_TRAIN_answers_identical_after_reloading":True,
                "margin_yes":reload_yes,"margin_no":reload_no}
    report={
      "stage":"S39_ACTUAL_PHYSICAL_36M_FIVEBIT_NOUN_INDEX_TRAIN",
      "source_S38_SHA256":SOURCE_SHA,
      "unique_original_parameters":SOURCE_PARAMS,
      "physical_model_bytes":SOURCE_BYTES,
      "physical_bpp":8*SOURCE_BYTES/SOURCE_PARAMS,
      "only_paid_5bit_group_indices_trained":True,
      "FP16_codebooks_frozen":True,
      "no_exported_FP32_dense_weight_or_unpaid_sidecar":True,
      "no_S29_QA_or_S26_inspected_QA_used":True,
      "only_TRAIN_for_learning_and_selection":True,
      "Jessica_chat_test_not_run":True,
      "LIVE_unmodified":True,
      "steps":STEPS,
      "accepted_steps":accepted,
      "accepted_5bit_groups":groups_accepted,
      "initial_old_seven_protected_TRAIN_ids":sorted(initial_old),
      "initial_Yes_margin":margin_yes,
      "initial_No_margin":margin_no,
      "end_Yes_margin":final_yes,
      "end_No_margin":final_no,
      "before_greedy_TRAIN":baseline,
      "after_greedy_TRAIN":after,
      "all_24_train_candidate_trials":records,
      "physical_rewrite":physical,
      "physical_reload":reload,
      "physical_checkpoint_created":output.exists(),
      "full_answer_new_generalization_unverified":True,
      "not_FP32_quality_parity":True,
    }
    save(a.out/"S39_ACTUAL_36M_CB59_5BIT_INDEX_TRAIN_ONLY.json",report)
    print(json.dumps({"event":"S39_REAL_PHYSICAL_INDEX_TRAIN_FINAL",
          "accepted_steps":accepted,"paid_5bit_groups":groups_accepted,
          "old_before":baseline["historical_TRAIN"]["fully_correct"],
          "old_after":after["historical_TRAIN"]["fully_correct"],
          "new_before":baseline["new_TRAIN"]["fully_correct"],
          "new_after":after["new_TRAIN"]["fully_correct"],
          "physical_created":output.exists(),
          "bpp":8*SOURCE_BYTES/SOURCE_PARAMS},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
