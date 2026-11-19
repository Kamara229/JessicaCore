"""NRA2 S90: REAL 43M/GT S83 TRAIN56-only MENTION main event TOP1.

S88 demonstrated source-only S59 mention TOP8 recall 48/48 on
S86 first-read synthetic but TOP1 only 30/48. S86 48 and S79 48
are SPENT; this file NEVER imports, fits, scores or retunes them.
Research exclusively S83 72 author-constructed TRAIN-only source
captured genuine original 43M+GT H8 in S84 AFTER protected 616 J7.

First load original read-only paid S26/S57/S59/S61/S62/S63/S65 and
compute source-only event candidate rankings/scores BEFORE opening
any S83 constructed target. The S90 selector trains only its 5725
new params using 56 group TRAIN author-designated *focus* positions.
Other candidates are merely 'not focus' for that synthetic request,
NOT negatives for the fact that an event may exist. After FIXED
epochs, evaluate 16 different authored-group synthetic TRAIN
diagnostics ONCE without optimizing hyperparams against this result.

Original ASSERTED S65 and all original 43M/GT weights unchanged.
Selected quoted event is only a mention, NEVER an asserted transfer.
Original protected 616 J7 preserved as historical frozen baseline,
but not tested for any new integrated active candidate (none exists).
No all-in bpp<=0.7/CPU>=90% or online causal certification, gate=0.
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

from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    gold_blind_eventfirst_features,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    semantic_features,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    source_local_scope,
)
from research.nra3_s20.s77_real43m_gt_mentionhood_vs_asserted_candidate_audit import (
    mention_only_source_mask,
)
from research.nra3_s20.s85_true43mGT_S82_event_scope_train_new72 import (
    fully_frozen_original_SOURCE_only,
    open_constructed_targets_ONLY_after_source_forward,
)
from research.nra3_s20.s89_sourceonly_event_mention_top1_ranker import (
    S89MainEventMentionTop1,PAID,
)
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72_sha256,TRAIN_GROUPS,DIAG_GROUPS,
)

STAGE="NRA2_S90_REAL43M_GT_S83_TRAIN56_S89_MENTION_TOP1_5725_FP16_SHADOW"
S83_SHA="285475efb25a1310ce87fa5ffc9e8ec99dd5f129d147b0edf4afa7195f43a62a"
S84_SHA="25c2cf61a3b3c331cd80b3c5e979eeb1badca3c3d2c8fd5125b7fc0de3fe35c4"
FP16_NAME="NRA2_S90_TRUE43M_GT_S89_TOP1_MAIN_MENTION_5725_FP16.safetensors"
REPORT_NAME="NRA2_S90_S89_TOP1_REAL43M_GT_NEWTRAIN56_SYNTHETICDIAG16.json"
SEED=901119
EPOCHS=80
LR=0.006
WD=0.01
N=72
K=8


def require(ok,why):
    if not bool(ok):
        raise ValueError("NRA2 S90 new main MENTION ranking research: "+why)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_only_old_S59_scores_BEFORE_gold(args):
    """All original true 43M paid frozen parent source-only forwards.

    Zero constructed target/gold modules imported or accessed.
    """
    src=fully_frozen_original_SOURCE_only(args)
    require(source_only_new72_sha256()==S83_SHA
            and sha(args.s84)==S84_SHA
            and len(src["source_rows"])==N
            and src["h"].shape[0]==N
            and src["candidates"].shape==src["allowed"].shape==(N,K),
            "genuine original Calibri43M GT S83 only, not old spent tests")
    original=load_file(str(args.s84),device="cpu")[
        "original_real43m_gt_word339_READONLY"].detach()
    old26,old57,old59,*_=src["parents"]
    with torch.inference_mode():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(
            original,src["valid"],src["unsafe"])
        event=gold_blind_eventfirst_features(
            original,obs,src["texts"])
        semantic=semantic_features(original,event)
        scoped=source_local_scope(semantic,src["texts"])
        old_logits=tuple(v.detach() for v in old57(event))
        scores=old59(scoped.base,old_logits)[0].detach().clone()
    mentions=mention_only_source_mask(src["valid"],src["unsafe"],
                                       obs.punctuation_mask,src["texts"])
    require(scores.shape==src["valid"].shape
            and not scores.requires_grad
            and bool(torch.isfinite(
                scores.gather(1,src["candidates"])[src["allowed"]]).all())
            and not bool((event.candidate_event&obs.quote_word_mask).any()),
            "old S59 source scoring finite and original asserted quote gate preserved")
    sorted_scores=scores.masked_fill(~mentions,float("-inf")).topk(K,dim=-1)
    expected_valid=torch.isfinite(sorted_scores.values)
    require(torch.equal(sorted_scores.indices,src["candidates"])
            and torch.equal(expected_valid,src["allowed"]),
            "S90 source-only ranking is EXACT same S85 original paid S59")
    require(all(tensor_sha(*(p.detach() for p in m.parameters()))==old_hash
                for m,old_hash in zip(src["parents"],src["old_hashes"])),
            "all old original paid parent weights physically unchanged")
    src["original_source_only_S59_word_scores"]=scores
    src["original_ASSERTED_quote_exclusion_preserved"]=True
    src["S77_MENTION_vs_asserted_source_only_preserved"]=True
    return src


def source_forward(net,src,*,zero_H8=False,zero_clause_fields=False):
    require(not zero_clause_fields,
            "S90 no engineered label-related field ablation switch allowed in inference")
    h=(torch.zeros_like(src["h"]) if zero_H8 else src["h"])
    return net(h,src["valid"],src["unsafe"],src["candidates"],
               src["allowed"],src["original_source_only_S59_word_scores"],
               src["texts"])


@torch.inference_mode()
def top1_metrics(net,src,gold,subset,*,zero_H8=False):
    require(subset.shape==(N,) and subset.dtype==torch.bool,
            "S90 exact TRAIN or synthetic group diagnostic mask")
    logits=source_forward(net,src,zero_H8=zero_H8)
    guess=logits.argmax(-1)
    focused=gold["gold_slot_POSTHOC_ONLY"]
    covered=gold["covered"]
    original_top1=(focused==0)&covered
    new_top1=(guess==focused)&covered
    rows=torch.where(subset)[0]
    require(len(rows)>0,"not an empty S83 TRAIN-only synthetic diagnostic group")
    quote_flags=[]
    for i in rows:
        # Original textual quote is source-derived, not S83 gold.
        quote_flags.append(bool(gold["truth"][i,0]))
    return {
        "n_group_rows":len(rows),
        "n_teacher_focus_anchor_in_original_MENTION_top8":int((covered&subset).sum()),
        "n_teacher_anchor_missing":int((~covered&subset).sum()),
        "n_original_S59_SOURCEONLY_MENTION_top1_matches_author_focus":
            int(original_top1[subset].sum()),
        "n_new_S89_SOURCEONLY_MENTION_top1_matches_author_focus":
            int(new_top1[subset].sum()),
        "source_S59_top1_accuracy_author_focus":float(original_top1[subset].float().mean()),
        "new_S89_top1_accuracy_author_focus":float(new_top1[subset].float().mean()),
        "n_constructed_quote_report_samples":sum(quote_flags),
        "teacher_anchored_MAIN_focus_not_event_existence_or_truth":True,
        "candidate_quoted_MENTION_is_not_asserted":True,
    }


def fixed_S83_train_ONLY(src,gold):
    require(int(gold["train"].sum())==56
            and int(gold["synthetic_diag"].sum())==16
            and int((gold["train"]&gold["covered"]).sum())==56
            and int((gold["synthetic_diag"]&gold["covered"]).sum())==16
            and not bool((gold["train"]&gold["synthetic_diag"]).any()),
            "S90 predeclared S83 56/16 structural-group TRAIN-only focus labels")
    torch.manual_seed(SEED)
    model=S89MainEventMentionTop1()
    require(model.budget()["new_trainable_parameters"]==PAID==5725
            and not model.is_trained and model.output_gate==0,
            "S89 strict distinct main-MENTION new params, not original S65")
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD)
    selected=torch.where(gold["train"])[0]
    source_focus_slot=gold["gold_slot_POSTHOC_ONLY"][selected]
    trace=[]
    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        scores=source_forward(model,src)
        loss=F.cross_entropy(scores[selected],source_focus_slot)
        require(bool(torch.isfinite(loss)),"source-only learned focus ranking loss finite")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in model.parameters()),
                "new event/H8/grammatical/source rank branch all receive TRAIN grads")
        nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        if epoch in (0,9,29,59,EPOCHS-1):
            trace.append({"epoch":epoch,"S83_AUTHOR_TRAIN56_only_focus_loss":
                          float(loss.detach())})
    model.eval()
    model.is_trained=True
    with torch.inference_mode():
        train=top1_metrics(model,src,gold,gold["train"])
        diag=top1_metrics(model,src,gold,gold["synthetic_diag"])
        no_h8=top1_metrics(model,src,gold,gold["synthetic_diag"],zero_H8=True)
    return model,{
        "S83_56_new_TRAIN_author_focus_only":True,
        "S83_16_different_group_synthetic_diagnostic_NOT_independent":True,
        "old_S79_S86_FIRSTREAD_48_each_SPENT_and_NEVER_IMPORTED":True,
        "training_negative_candidates_mean_not_author_focus_NOT_no_event":True,
        "same_source_only_original_S59_rankings_and_unmodified_S65":True,
        "fixed_hyperparameters":{"seed":SEED,"epochs":EPOCHS,"lr":LR,"wd":WD},
        "training_loss_only_trace":trace,
        "TRAIN56_focus_top1":train,
        "SYNTHETIC_diag16_focus_top1":diag,
        "SYNTHETIC_diag16_zero_H8_ablation":no_h8,
        "S85_scenario_eight_areas_NOT_UPDATED":True,
        "not_user_visible_or_event_fact_truth_measure":True,
    }


def main():
    parser=argparse.ArgumentParser()
    for n in ("s84","s84_report","s26","s57_paid","s59_paid",
              "s61_paid","s62_paid","s63_paid","s65","s65_manifest","out"):
        parser.add_argument("--"+n.replace("_","-"),required=True,type=Path)
    args=parser.parse_args()
    require(not args.out.exists(),
            "never overwrite original physically trained selector experiment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source=source_only_old_S59_scores_BEFORE_gold(args)
    gold=open_constructed_targets_ONLY_after_source_forward(source)
    net,measure=fixed_S83_train_ONLY(source,gold)
    require(all(tensor_sha(*(p.detach() for p in m.parameters()))==old_hash
                for m,old_hash in zip(source["parents"],source["old_hashes"])),
            "original S26 S57 S59 S61 S62 S63 S65 never train or mutate")
    physical={
        "NRA2_S90_NEW_SOURCEONLY_MENTION_TOP1." + k:
            v.detach().half().cpu().contiguous()
        for k,v in net.state_dict().items()
    }
    require(sum(t.numel() for t in physical.values())==PAID
            and all(bool(torch.isfinite(t).all()) for t in physical.values()),
            "physical FP16 5725 new event ranking weights valid")
    args.out.mkdir(parents=True,exist_ok=False)
    saved=args.out/FP16_NAME
    save_file(physical,str(saved))
    reread=load_file(str(saved),device="cpu")
    require(set(reread)==set(physical)
            and all(torch.equal(reread[k],physical[k]) for k in reread),
            "true new original43M derived selector FP16 readback")
    report={
        "stage":STAGE,
        "S83_newTRAIN72_SOURCE_SHA256":source_only_new72_sha256(),
        "S84_true_original43M_GT_H8_SHA256":sha(args.s84),
        "S90_new_5725_FP16_selector_SHA256":sha(saved),
        "new_selector_parameters":PAID,
        "FP16_new_selector_bytes_nominal":PAID*2,
        "S84_original_Calibri43M_GT_protected616J7_replayed_before_source":True,
        "all_old_original_S26_S57_S59_S61_S62_S63_S65_weights_unchanged":True,
        "source_only_old_S59_rank_and_scores_computed_BEFORE_S83_constructed_gold_import":True,
        "new_NRA2_S89_focus_selector_is_not_S65_asserted_fact_or_roles":True,
        "old_S65_asserted_quoted_event_exclusion_still_enforced":True,
        "old_S79_SPENT48_NEVER_reused":True,
        "old_S86_SPENT48_NEVER_reused":True,
        "S83_diag16_not_independent_firstread_or_generalization":True,
        "new_active_candidate_616J7_retention_not_measured":True,
        "all_in_bpp_CPU_online_causality_NOT_certified":True,
        "original_parent_and_S85_existing_eight_miniareas_not_modified":True,
        "S90_sourceonly_focus_training_and_diagnostics":measure,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (args.out/REPORT_NAME).write_text(
        json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf8")
    print("NRA2_S90_REAL43MGT_SOURCEONLY_NEW5725_FOCUS_TOP1_AUTHOR_TRAIN",
          json.dumps({
              "original_S59_TRAIN56_top1":
                  measure["TRAIN56_focus_top1"]["n_original_S59_SOURCEONLY_MENTION_top1_matches_author_focus"],
              "new_S89_TRAIN56_top1":
                  measure["TRAIN56_focus_top1"]["n_new_S89_SOURCEONLY_MENTION_top1_matches_author_focus"],
              "original_S59_groupDIAG16_top1":
                  measure["SYNTHETIC_diag16_focus_top1"]["n_original_S59_SOURCEONLY_MENTION_top1_matches_author_focus"],
              "new_S89_groupDIAG16_top1":
                  measure["SYNTHETIC_diag16_focus_top1"]["n_new_S89_SOURCEONLY_MENTION_top1_matches_author_focus"],
              "without_H8_groupDIAG16_top1":
                  measure["SYNTHETIC_diag16_zero_H8_ablation"]["n_new_S89_SOURCEONLY_MENTION_top1_matches_author_focus"],
              "new_FP16_params":PAID,"gate":0,
          },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
