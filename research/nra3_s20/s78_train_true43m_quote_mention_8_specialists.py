"""NRA2 S78: NEW specialized eight mini-areas on true original43M GT MENTIONS.

S77 demonstrated original S52 asserted-factual source beam missed 24/120
constructed S74 TRAIN event anchors, ALL due to quote exclusion. New
strictly disconnected source-only quote-inclusive MENTION top8 covered
120/120 and is NOT itself an assertion/recipient predictor. S78 fits
eight INDEPENDENT domain heads from ACTUAL original S75 43M/GT H8.

No oracle input to forward. Constructed labels/teacher event anchors
exist ONLY in loss/diagnostic and are not human-adjudicated gold.
No negatives on unanchored event candidates (they may be real mentions);
therefore no claim of standalone mention precision or autonomous event
recognition. Quotation/negation/time are labels of an EVENT, not sentences.
The 30 grouped synthetic TRAIN diagnostics repeat the same 10 templates,
so absolutely NO independent generalization claim.

Old S52/S57/S59/S62/S65 outputs/weights and original 616 J7 unchanged,
output gate=0. This does not connect any new result to original decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import tensor_sha
from research.nra3_s20.s72_nra2_situation_microareas import AREAS
from research.nra3_s20.s74_newtrain120_source_ONLY import (
    source_sha256,GROUP_TRAIN,GROUP_DIAG,
)
from research.nra3_s20.s77_real43m_gt_mentionhood_vs_asserted_candidate_audit import (
    source_only_event_proposals,
)

STAGE="NRA2_S78_ORIGINAL43M_GT_S77_QUOTE_INCLUSIVE_MENTION_8SPECIALIST_SHADOW"
SEED=781119
EPOCHS=75
LR=0.01
WD=0.01
N_PAID=7496
FP16_FILE="NRA2_S78_TRUE43M_GT_MENTION8_SITUATION_SPECIALISTS_7496_FP16.safetensors"
MANIFEST="NRA2_S78_TRUE43M_GT_MENTION8_TRAIN90_DIAG30_SHADOW_MANIFEST.json"

def require(v,msg):
    if not bool(v):
        raise ValueError("NRA2 S78 frozen original43M GT mention mini-areas: "+msg)


class SpecialistMiniArea(nn.Module):
    """Independently learned event-local factor, not one exclusive class."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(35,8),nn.GELU(),nn.Linear(8,1))

    def forward(self,x):
        return self.net(x).squeeze(-1)


class S78MentionSituationSpecialists(nn.Module):
    """All forward arguments are read-only source features and source ranks."""
    def __init__(self):
        super().__init__()
        self.event_proj=nn.Linear(320,8,bias=False)
        self.global_proj=nn.Linear(320,8,bias=False)
        self.areas=nn.ModuleList([SpecialistMiniArea() for _ in AREAS])
        self.output_gate=0
        self.candidate_accepted=False
        self.activation_allowed=False
        self.online_causal_certified=False
        require(sum(p.numel() for p in self.parameters())==N_PAID,
                "NRA2 S78 exact 7496 separate specialist paid budget")

    def forward(self,original_native339,valid_words,
                source_only_mention_top8,top8_allowed):
        require(self.output_gate==0
                and not self.candidate_accepted
                and not self.activation_allowed
                and not self.online_causal_certified,
                "S78 no output activation")
        x=original_native339
        require(x.ndim==3 and x.shape[0]==120 and x.shape[-1]==339
                and x.dtype==torch.float32 and not x.requires_grad
                and torch.isfinite(x).all()
                and valid_words.shape==x.shape[:2]
                and valid_words.dtype==torch.bool
                and top8_allowed.shape==source_only_mention_top8.shape==(120,8)
                and top8_allowed.dtype==torch.bool
                and source_only_mention_top8.dtype==torch.long
                and bool(((source_only_mention_top8>=0)
                          &(source_only_mention_top8<x.shape[1])).all()),
                "S78 ONLY genuine detached original43M H8 native + source top8")
        b,w,_=x.shape;k=8
        h=x[:,:,:320].detach()
        meta=x[:,:,320:339].detach()
        ctx=(h*valid_words[:,:,None]).sum(1)/(
            valid_words.sum(1,keepdim=True).clamp_min(1))
        z_event=self.event_proj(h).gather(
            1,source_only_mention_top8[:,:,None].expand(b,k,8))
        z_global=self.global_proj(ctx)[:,None,:].expand(b,k,8)
        meta_ev=meta.gather(
            1,source_only_mention_top8[:,:,None].expand(b,k,19))
        features=torch.cat((z_event,z_global,meta_ev),dim=-1)
        require(features.shape==(120,8,35),
                "S78 event-specific read-only 35D mini-area ABI")
        logits=torch.stack([area(features) for area in self.areas],dim=-1)
        logits=logits.masked_fill(~top8_allowed[:,:,None],-1.e5)
        require(logits.shape==(120,8,len(AREAS))
                and torch.isfinite(logits).all(),
                "S78 independent 8 situation heads source features finite")
        return logits

    def budget(self):
        return {
            "new_FP16_trainable_parameters":N_PAID,
            "new_FP16_weight_bytes":N_PAID*2,
            "eight_separate_trainable_heads":len(self.areas),
            "original_parent_weights_modified":False,
            "all_in_bpp_CPU_NOT_certified":True,
            "output_gate":0,"candidate_accepted":False,
            "activation_allowed":False,
        }


def source_first_then_train_gold(args):
    """All pretrained H8 / original S65 reads BEFORE first label import."""
    obs=source_only_event_proposals(args)
    p=args.s75
    x=load_file(str(p),device="cpu")[
        "original_real43m_gt_word339_READONLY"].detach().float().clone()
    valid=obs["valid"].detach().clone()
    candidates,allowed=obs["ranked"]["mention_quote_inclusive"][8]
    require(x.shape[0]==120 and x.shape[-1]==339
            and candidates.shape==allowed.shape==(120,8)
            and not bool((obs["asserted"] & obs["quote"]).any())
            and bool((obs["mention"] & obs["quote"]).any()),
            "true43M H8 original + strict separate mention/factual channels")
    # Hard source/gold wall: construction labels imported ONLY once
    # genuine Calibri43M/GT/S65 source-only forwards have completed.
    from research.nra3_s20.s76_s74_newtrain120_event_relative_gold_TRAIN_LOSS_ONLY import (
        template_constructed_OLDTRAIN_event_area_gold120_for_loss_only,
    )
    gold=template_constructed_OLDTRAIN_event_area_gold120_for_loss_only()
    rows=obs["texts"]
    require([z["row_id"] for z in gold]==[z["row_id"] for z in rows]
            and len(gold)==120,
            "S74 TRAIN-only labels imported after source-only parent")
    anchor=torch.tensor(
        [z["main_event_raw_index_ONLY_in_TRAIN_loss"] for z in gold],
        dtype=torch.long)
    area=torch.tensor(
        [z["event_area_multi_hot_ONLY_in_TRAIN_loss"] for z in gold],
        dtype=torch.float32)
    matches=(candidates==anchor[:,None])&allowed
    covered=matches.any(-1)
    # Physical independent S77 experiment established 120/120; fail
    # if original weights or BPE changed. NEVER use anchor to rerank.
    require(int(covered.sum())==120,
            "S77 source-only quote-inclusive top8 must recover 120 constructed anchors")
    slot=matches.long().argmax(-1)
    train=torch.tensor([z["family"] in GROUP_TRAIN for z in rows],
                       dtype=torch.bool)
    diag=torch.tensor([z["family"] in GROUP_DIAG for z in rows],
                      dtype=torch.bool)
    require(int(train.sum())==90 and int(diag.sum())==30
            and not bool((train&diag).any())
            and area.shape==(120,8),
            "S78 grouped source family new synthetic TRAIN90/diagnostic30")
    return obs,x,valid,candidates,allowed,slot,area,train,diag


@torch.no_grad()
def scores(model,x,valid,indices,allowed,slot,area,subset,
           *,without_h8=False,without_raw_meta=False):
    source=x.clone()
    if without_h8:
        source[:,:,:320]=0
    if without_raw_meta:
        source[:,:,320:339]=0
    logits=model(source,valid,indices,allowed)[torch.arange(120),slot]
    predicted=logits.sigmoid()>=0.5
    truth=area.bool()
    active=predicted[subset];g=truth[subset]
    tp=int((active&g).sum());fp=int((active&~g).sum())
    fn=int((~active&g).sum())
    p=tp/(tp+fp) if tp+fp else 0.
    r=tp/(tp+fn) if tp+fn else 0.
    f1=2*p*r/(p+r) if p+r else 0.
    by={}
    for i,name in enumerate(AREAS):
        a=active[:,i];y=g[:,i]
        atp=int((a&y).sum());afp=int((a&~y).sum());afn=int((~a&y).sum())
        ap=atp/(atp+afp) if atp+afp else 0.
        ar=atp/(atp+afn) if atp+afn else 0.
        by[name]={
            "support":int(y.sum()),"tp":atp,"fp":afp,"fn":afn,
            "precision":ap,"recall":ar,
            "f1":2*ap*ar/(ap+ar) if ap+ar else 0.,
        }
    return {
        "n_constructed_TRAIN_event_anchor_in_source_only_mention_beam":int(subset.sum()),
        "micro_precision":p,"micro_recall":r,"micro_f1":f1,
        "exact_all8_multihot":int((active==g).all(-1).sum()),
        "per_area":by,
        "gold_event_anchor_used_ONLY_for_teacher_forced_diagnostic":True,
        "model_forward_without_gold_labels":True,
        "not_autonomous_mention_factuality_or_recipient_detection":True,
    }


def fit_and_audit(obs,x,valid,indices,allowed,slot,area,train,diag):
    torch.manual_seed(SEED)
    model=S78MentionSituationSpecialists()
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    positive=area[train].sum(0)
    weights=((int(train.sum())-positive)/
             positive.clamp_min(1)).clamp(1,6)
    traces=[]
    n=torch.arange(120)[train]
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits=model(x,valid,indices,allowed)[n,slot[train]]
        error=F.binary_cross_entropy_with_logits(
            logits,area[train],pos_weight=weights)
        require(bool(torch.isfinite(error)),"S78 source-only factor loss finite")
        error.backward()
        require(all(p.grad is not None and
                    bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "S78 each of eight mini-area experts gradient required")
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        if epoch in (0,9,29,49,EPOCHS-1):
            traces.append({"epoch":epoch,"TRAIN_only_area_loss":float(error.detach())})
    model.eval()
    # No checkpoint/threshold tuned to grouped diagnostic outcomes.
    quoted=torch.tensor([int(z["row_id"].split("_")[1]) in (2,10)
                         for z in obs["texts"]],dtype=torch.bool)
    with torch.no_grad():
        report={
            "train90_ANCHORED":scores(model,x,valid,indices,allowed,slot,area,train),
            "diagnostic30_ANCHORED":scores(model,x,valid,indices,allowed,slot,area,diag),
            "diagnostic30_no_H8_ANCHORED":scores(
                model,x,valid,indices,allowed,slot,area,diag,without_h8=True),
            "diagnostic30_no_raw19_ANCHORED":scores(
                model,x,valid,indices,allowed,slot,area,diag,without_raw_meta=True),
            "diagnostic_quoted6_ANCHORED":scores(
                model,x,valid,indices,allowed,slot,area,diag&quoted),
            "diagnostic_notquoted24_ANCHORED":scores(
                model,x,valid,indices,allowed,slot,area,diag&~quoted),
            "TRAIN_only_loss_history":traces,
            "n_original_S65_factual_event_beam8_constructed_anchor_covered":96,
            "n_new_mention_beam8_constructed_anchor_covered":120,
            "source_only_mentions_event_candidates_not_asserted":True,
            "fixed_hyperparameters":{"seed":SEED,"epochs":EPOCHS,"lr":LR,"wd":WD},
            "same_10_synthetic_templates_across_all_families":True,
            "gold_anchored_for_loss_and_diagnostic_not_event_discovery":True,
            "unanchored_event_candidate_FALSE_POSITIVES_NOT_MEASURED":True,
            "no_original_S65_logits_or_assertion_decoding_changed":True,
            "no_independent_first_read_or_production_integration":True,
        }
    return model,report


def main():
    parser=argparse.ArgumentParser()
    for name in ("s75","s75_report","s26","s57_paid","s59_paid",
                 "s61_paid","s62_paid","s63_paid","s65","s65_manifest","out"):
        parser.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    args=parser.parse_args()
    require(not args.out.exists(),"S78 never overwrite existing experimental weights")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    obs,x,valid,indices,allowed,slot,area,train,diag=source_first_then_train_gold(args)
    model,diag_report=fit_and_audit(
        obs,x,valid,indices,allowed,slot,area,train,diag)
    require(all(tensor_sha(*(p.detach() for p in m.parameters()))==hash_before
                for m,hash_before in zip(obs["parents"],obs["old_hashes"])),
            "S78 original physical parent weights never train/mutate")
    state={
        "NRA2_S78_NEW_MENTION_SPECIALISTS."+k:v.detach().half().cpu().contiguous()
        for k,v in model.state_dict().items()
    }
    require(sum(x.numel() for x in state.values())==N_PAID
            and all(torch.isfinite(x).all() for x in state.values()),
            "S78 all eight specialists physical FP16")
    args.out.mkdir(parents=True,exist_ok=False)
    file=args.out/FP16_FILE
    save_file(state,str(file))
    saved=load_file(str(file),device="cpu")
    require(set(saved)==set(state)
            and all(torch.equal(saved[k],v) for k,v in state.items()),
            "S78 full FP16 physical state readback exact")
    report={
        "stage":STAGE,
        "real43M_original_S75_616J7_H8_readonly_120_native_BPE_replayed_before_this":True,
        "S74_newtrain_source_SHA256":source_sha256(),
        "S75_source_original_native_H8_physical_SHA256":
            "83c1da7b6e79afcfad339c498dcf0f4937411fbbfb081d500d5ca59ea4798900",
        "S77_sourceonly_mentioned_vs_asserted_decoupled":True,
        "S78_no_original43M_GT_S26_S57_S59_S61_S62_S63_S65_weight_change":True,
        "n_train":90,"n_grouped_diagnostic_not_independent":30,
        "n_trainable_specialist_parameters":N_PAID,
        "fp16_new_specialist_weights_SHA256":
            hashlib.sha256(file.read_bytes()).hexdigest(),
        "eight_distinct_supervised_situation_microareas_NOT_same_head":True,
        "constructed_labels_not_independently_human_reviewed":True,
        "no_original_S65_recipients_or_factuality_CHANGED":True,
        "no_old_S66_spent32_gold_or_sealed_EVAL32_EVAL128":True,
        "no_new_candidate_J7_616_bpp_CPU_or_online_safety_certification":True,
        "evaluation_teacher_anchored_not_end_to_end":True,
        "per_group_synthetic_TRAIN_diagnostics":diag_report,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (args.out/MANIFEST).write_text(json.dumps(report,ensure_ascii=False,
                        sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("NRA2_S78_REAL43M_GT_QUOTE_MENTION_8_INDEPENDENT_SPECIALISTS",
          json.dumps({
              "TRAIN_microF1":diag_report["train90_ANCHORED"]["micro_f1"],
              "synthetic_diag30_microF1":diag_report["diagnostic30_ANCHORED"]["micro_f1"],
              "synthetic_diag_quoted6_microF1":diag_report["diagnostic_quoted6_ANCHORED"]["micro_f1"],
              "diag_noH8_microF1":diag_report["diagnostic30_no_H8_ANCHORED"]["micro_f1"],
              "quote_mention_covered_all120":True,
              "FP16_new_params":N_PAID,
              "gate":0,
          },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
