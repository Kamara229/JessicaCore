"""NRA2 S85: REAL original frozen 43M/GT new72 H8 TRAIN-only
S82 event-centered 8 independently trained clause/scope area heads.

The prior S81 frozen S78 novel48 first-read F1 28.83% (synthetic)
was SPENT immediately. S79 original 48 source AND gold are NEVER
imported, trained/tuned/checked in this module. New SOURCE72 is S83
SHA/Git-blob precommitted and physically captured after protected
original616J7 in S84 without labels/model scores.

Source-only original S59 MENTION top8 is constructed BEFORE S83
constructed TRAIN-only loss labels are imported; event gold never
changes rank, logits or S82 forward. Only 56 grouped synthetic TRAIN
are optimized, 16 separate authored-group synthetic TRAIN diagnostics
are evaluated ONCE at fixed epoch 90 and threshold 0.5; they are
NOT an independent naturalistic first-read. No generated output or
factual transfer/recipient quality claims. Gate remains ZERO.
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
from research.nra3_s20.s73_fit_real43m_old344_microareas import (
    original_frozen_paid,
)
from research.nra3_s20.s77_real43m_gt_mentionhood_vs_asserted_candidate_audit import (
    mention_only_source_mask,
)
from research.nra3_s20.s82_predicate_centered_situation_fields import (
    S82PredicateCenteredMiniAreas,AREAS,FIELDS,
)
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,
    TRAIN_GROUPS,DIAG_GROUPS,
)

STAGE="NRA2_S85_REAL43M_GT_S83_DIVERSE72_S82_EVENT_SCOPE_8264_TRAIN_ONLY"
S83_SHA="285475efb25a1310ce87fa5ffc9e8ec99dd5f129d147b0edf4afa7195f43a62a"
S84_SHA="S84_PHYSICAL_SHA_TO_PIN_AFTER_ORIGINAL_J7_CI"
S84_TENSOR="NRA2_S84_GENUINE43M_GT_DIVERSE72_WORD339_NO_GOLD.safetensors"
S84_REPORT="NRA2_S84_GENUINE43M_GT_DIVERSE72_GOLD_BLIND_SOURCE_PROVENANCE.json"
SEED=851119
EPOCHS=90
LR=0.004
WD=0.015
N=72
K=8
PAID=8264
FILE="NRA2_S85_TRUE43M_GT_S82_EVENT_LOCAL8264_FP16.safetensors"
MANIFEST="NRA2_S85_TRUE43M_GT_DIVERSE72_S82_8AREA_TRAINONLY_REPORT.json"


def require(ok,msg):
    if not bool(ok):
        raise ValueError("NRA2 S85 real43M GT event-field source-only: "+msg)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fully_frozen_original_SOURCE_only(args):
    """TRUE original H8+paid parent scores computed BEFORE gold opened."""
    rows=source_only_new72()
    r=json.loads(args.s84_report.read_text(encoding="utf8"))
    require(len(rows)==N
            and source_only_new72_sha256()==S83_SHA
            and args.s84.name==S84_TENSOR
            and args.s84_report.name==S84_REPORT
            and sha(args.s84)==S84_SHA
            and r["stage"]=="NRA2_S84_TRUE43M_GT_FIRSTREAD_DIVERSE72_GOLD_BLIND_H8_SOURCE"
            and r["genuine_source_tensor_file_SHA256"]==S84_SHA
            and r["S83_new_diverse72_SOURCE_ONLY_SHA256"]==S83_SHA
            and r["S83_PRECOMMIT_SUCCESS_WORKFLOW_RUN_ID"]==44514452404
            and r["original_Calibri43M_protected_616J7_physically_replayed_BEFORE_training_source"]
            and r["S83_CONSTRUCTION_GOLD_NEVER_IMPORTED_BY_REAL_H8_CPU"]
            and r["S82_architecture_weights_NEVER_LOADED_BY_SOURCE_H8_CAPTURE"]
            and r["n_diverse_train_only_input_rows"]==N
            and r["new_model_training_steps"]==0
            and r["output_gate"]==0 and not r["candidate_accepted"],
            "S84 physical original43M GT source SHA protected after original616J7")
    t=load_file(str(args.s84),device="cpu")
    require(set(t)=={
        "original_real43m_gt_word339_READONLY",
        "original_native_word_valid_mask",
        "native_BPE_causally_unsafe_word_mask",
        "source_row_indices_in_frozen_S83_DIVERSE72",
    } and t["source_row_indices_in_frozen_S83_DIVERSE72"].tolist()==list(range(N)),
            "no gold in real original43M H8 original BPE tensors")
    x=t["original_real43m_gt_word339_READONLY"].detach().float().clone()
    valid=t["original_native_word_valid_mask"].detach().clone()
    unsafe=t["native_BPE_causally_unsafe_word_mask"].detach().clone()
    require(x.shape[0]==N and x.shape[-1]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and not bool((unsafe&~valid).any())
            and bool(torch.isfinite(x).all()),
            "true original native word339 H8+19 unaltered")
    parents,hashes=original_frozen_paid(args)
    old26,old57,old59,*_=parents
    texts=tuple(row["text"] for row in rows)
    with torch.no_grad():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        ev=gold_blind_eventfirst_features(x,obs,texts)
        sem=semantic_features(x,ev)
        fused=source_local_scope(sem,texts)
        oldlogits=tuple(v.detach() for v in old57(ev))
        scores=old59(fused.base,oldlogits)[0].detach()
    mention=mention_only_source_mask(
        valid,unsafe,obs.punctuation_mask,texts)
    require(scores.shape==valid.shape
            and not bool((ev.candidate_event&obs.quote_word_mask).any())
            and bool((mention&obs.quote_word_mask).any()),
            "old asserted quote exclusion, new separate MENTION includes quotes")
    sorted_scores=scores.masked_fill(~mention,float("-inf"))
    top=sorted_scores.topk(K,dim=-1)
    candidates=top.indices.detach().long().clone()
    allowed=torch.isfinite(top.values).detach().clone()
    require(candidates.shape==allowed.shape==(N,K)
            and bool(allowed.any(-1).all())
            and not bool((mention.gather(1,candidates)&~allowed).any()),
            "frozen source-only top8 old S59 score: NO constructed event GOLD")
    require(all(not p.requires_grad for x in parents for p in x.parameters()),
            "no original paid parent can be trained")
    return {
        "h":x[:,:,:320].detach().clone(),
        "valid":valid,"unsafe":unsafe,
        "texts":texts,
        "source_rows":rows,
        "candidates":candidates,"allowed":allowed,
        "parents":parents,"old_hashes":hashes,
        "n_original_asserted_eligible_words":int(ev.candidate_event.sum()),
        "n_new_sourceonly_mention_eligible_words":int(mention.sum()),
    }


def open_constructed_targets_ONLY_after_source_forward(source):
    """No S83 gold module present in original model or S82 forward."""
    from research.nra3_s20.s83_NEW_DIVERSE72_gold_TRAIN_LOSS_ONLY import (
        S83_train_loss_gold_only,precommit_manifest,
    )
    gold=S83_train_loss_gold_only()
    spec=precommit_manifest()
    require(len(gold)==N
            and [z["row_id"] for z in gold]
               ==[z["row_id"] for z in source["source_rows"]]
            and spec["source_only_sha256"]==S83_SHA
            and spec["n_train_rows"]==56
            and spec["n_synthetic_diag_rows"]==16,
            "S83 original TRAIN source/annotation pairing remains locked")
    truth=torch.tensor(
        [z["eight_area_multi_hot_ONLY_loss"] for z in gold],
        dtype=torch.float32)
    indices=torch.tensor(
        [z["event_raw_word_index_ONLY_loss_or_posthoc"] for z in gold],
        dtype=torch.long)
    source_idx=source["candidates"]
    available=source["allowed"]
    matches=(source_idx==indices[:,None])&available
    covered=matches.any(-1)
    slot=matches.long().argmax(-1)
    train=torch.tensor([z["group"] in TRAIN_GROUPS
                        for z in source["source_rows"]],dtype=torch.bool)
    diag=torch.tensor([z["group"] in DIAG_GROUPS
                       for z in source["source_rows"]],dtype=torch.bool)
    require(int(train.sum())==56 and int(diag.sum())==16
            and not bool((train&diag).any())
            and truth.shape==(N,8)
            and int((train&covered).sum())>=28
            and int((diag&covered).sum())>=6,
            "S85 original source-only real 43M mention event coverage insufficient")
    return {
        "truth":truth,"event_anchor_posthoc":indices,
        "gold_slot_POSTHOC_ONLY":slot,
        "covered":covered,"train":train,"synthetic_diag":diag,
    }


def forward(net,src,*,without_H8=False):
    h=src["h"]
    if without_H8:
        h=torch.zeros_like(h)
    return net(h,src["valid"],src["unsafe"],
               src["candidates"],src["allowed"],src["texts"])


@torch.inference_mode()
def metrics(net,src,gold,mask,*,without_H8=False):
    y=gold["truth"]
    covered=gold["covered"]
    actual=mask&covered
    slots=gold["gold_slot_POSTHOC_ONLY"]
    out=forward(net,src,without_H8=without_H8)
    idx=torch.where(actual)[0]
    require(len(idx)>0,"S85 no exact source-only candidate event in diagnostic")
    logits=out[idx,slots[idx]]
    pred=logits.sigmoid().ge(0.5)
    target=y[idx].bool()
    tp=int((pred&target).sum())
    fp=int((pred&~target).sum())
    fn=int((~pred&target).sum())
    precision=tp/(tp+fp) if tp+fp else 0.
    recall=tp/(tp+fn) if tp+fn else 0.
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    per={}
    for i,name in enumerate(AREAS):
        p=pred[:,i];g=target[:,i]
        a=int((p&g).sum());b=int((p&~g).sum());c=int((~p&g).sum())
        ap=a/(a+b) if a+b else 0.
        ar=a/(a+c) if a+c else 0.
        per[name]={"support":int(g.sum()),
                   "tp":a,"fp":b,"fn":c,
                   "precision":ap,"recall":ar,
                   "F1":2*ap*ar/(ap+ar) if ap+ar else 0.}
    return {
        "n_group_input":int(mask.sum()),
        "n_event_anchor_IN_source_only_MENTION_top8":len(idx),
        "n_event_anchor_MISSING":int((mask&~covered).sum()),
        "micro_precision":precision,"micro_recall":recall,"micro_F1":f1,
        "exact_all8_on_covered":int(pred.eq(target).all(-1).sum()),
        "exact_all8_including_uncovered_failures":int(pred.eq(target).all(-1).sum()),
        "teacher_event_anchor_used_ONLY_for_loss_diagnostics":True,
        "mention_precision_and_unanchored_non_event_FALSE_POSITIVES_unmeasured":True,
        "factor_model_frozen_old_43M_and_old_paid_S65_unchanged":True,
        "per_area":per,
    }


def fit_new_model_only(src,gold):
    torch.manual_seed(SEED)
    net=S82PredicateCenteredMiniAreas()
    require(net.budget()["new_neural_parameters"]==PAID
            and net.budget()["within_10000_new_params_budget"]
            and not net.is_trained and not net.activation_allowed,
            "new S82 small 8264 parameter research-only candidate")
    mask=gold["train"]&gold["covered"]
    idx=torch.where(mask)[0]
    target=gold["truth"][idx]
    pos=target.sum(0)
    counts=len(idx)
    weights=((counts-pos)/pos.clamp_min(1)).clamp(1,8)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
    trace=[]
    for epoch in range(EPOCHS):
        net.train()
        opt.zero_grad(set_to_none=True)
        out=forward(net,src)
        logits=out[idx,gold["gold_slot_POSTHOC_ONLY"][idx]]
        loss=F.binary_cross_entropy_with_logits(
            logits,target,pos_weight=weights)
        require(bool(torch.isfinite(loss)),"S85 original native 43M train factor loss nonfinite")
        loss.backward()
        require(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in net.parameters()),
                "all new H8 projections and eight situation microhead gradients")
        nn.utils.clip_grad_norm_(net.parameters(),5.)
        opt.step()
        if epoch in (0,9,29,59,EPOCHS-1):
            trace.append({"epoch":epoch,"TRAIN_only_loss":float(loss.detach())})
    net.eval()
    net.is_trained=True
    with torch.inference_mode():
        train=metrics(net,src,gold,gold["train"])
        diag=metrics(net,src,gold,gold["synthetic_diag"])
        without=metrics(net,src,gold,gold["synthetic_diag"],without_H8=True)
    return net,{
        "fixed_training_seed":SEED,
        "fixed_epochs":EPOCHS,
        "fixed_lr":LR,"fixed_weight_decay":WD,
        "only_new_s83_SOURCE72_labels_used_for_training":True,
        "not_old_S79_NOVEL48_gold_or_text":True,
        "n_precommitted_new_train_rows":56,
        "n_grouped_synthetic_diagnostic_rows":16,
        "training_covered_original_S59_MENTION":int(mask.sum()),
        "synthetic_diagnostic_teacher_anchor_and_same_patterns_NOT_independent":True,
        "S77_mentionhood_top8_not_asserted_fact":True,
        "not_factuality_or_recipient_or_event_selection_accuracy":True,
        "training_trace_only":trace,
        "new_train56_teacher_anchored":train,
        "synthetic_diag16_teacher_anchored":diag,
        "synthetic_diag16_ZERO_H8_ablation":without,
    }


def main():
    parser=argparse.ArgumentParser()
    for name in ("s84","s84_report","s26","s57_paid","s59_paid",
                 "s61_paid","s62_paid","s63_paid","s65","s65_manifest","out"):
        parser.add_argument("--"+name.replace("_","-"),required=True,type=Path)
    a=parser.parse_args()
    require(not a.out.exists(),"never overwrite first real S85 trained experiment")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source=fully_frozen_original_SOURCE_only(a)
    gold=open_constructed_targets_ONLY_after_source_forward(source)
    net,stats=fit_new_model_only(source,gold)
    require(all(tensor_sha(*(p.detach() for p in m.parameters()))==before
                for m,before in zip(source["parents"],source["old_hashes"])),
            "S85 original old 43M-derived S26/S57/S59/S61/S62/S63/S65 physically immutable")
    weights={"NRA2_S85_S82_EVENT_LOCAL."+k:v.detach().half().cpu().contiguous()
             for k,v in net.state_dict().items()}
    require(sum(z.numel() for z in weights.values())==PAID
            and all(torch.isfinite(z).all() for z in weights.values()),
            "S85 8264 real FP16 from S82 eight mini-area source model")
    a.out.mkdir(parents=True,exist_ok=False)
    file=a.out/FILE
    save_file(weights,str(file))
    reread=load_file(str(file),device="cpu")
    require(set(weights)==set(reread)
            and all(torch.equal(weights[k],reread[k]) for k in weights),
            "S85 actual physical FP16 all 8 new experts serialized")
    output={
        "stage":STAGE,
        "source_S83_SHA256":source_only_new72_sha256(),
        "true_real43M_GT_S84_native_H8_SHA256":sha(a.s84),
        "new_S82_event_centered_8areas_S85_FP16_SHA256":sha(file),
        "new_8264_trainable_FP16_params":PAID,
        "new_8_expert_64BIT_training_does_NOT_touch_original_weight":True,
        "original_616_J7_protected_parent_replayed_before_S84_H8":True,
        "all_original_S26_S57_S59_S61_S62_S63_S65_weights_unchanged":True,
        "source_only_candidates_computed_before_S83_constructed_gold_open":True,
        "S83_diverse_72_source_only_before_train_loss_labels":True,
        "n_original_S57_asserted_candidate_tokens":
            source["n_original_asserted_eligible_words"],
        "n_original_S59_quote_inclusive_MENTION_candidate_tokens":
            source["n_new_sourceonly_mention_eligible_words"],
        "S79_spent48_NOVEL_TEST_NEVER_train_reused_or_retuned":True,
        "synthetic_diagnostic_group_NOT_independent_or_prospective_EVAL":True,
        "full_new_candidate_616_J7_NOT_measured":True,
        "all_in_bpp_and_CPU_NOT_measured":True,
        "online_causal_generation_NOT_certified":True,
        "not_end_to_end_factuality_or_recipient_evaluation":True,
        "training_and_diagnostics":stats,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (a.out/MANIFEST).write_text(json.dumps(
        output,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print("NRA2_S85_REAL43M_GT_NEWTRAIN56_EVENT_CENTERED_EIGHT_AREAS",
          json.dumps({
              "train_F1":stats["new_train56_teacher_anchored"]["micro_F1"],
              "diag_F1":stats["synthetic_diag16_teacher_anchored"]["micro_F1"],
              "diag_zeroH8_F1":stats["synthetic_diag16_ZERO_H8_ablation"]["micro_F1"],
              "train_coverage":stats["new_train56_teacher_anchored"]["n_event_anchor_IN_source_only_MENTION_top8"],
              "diag_coverage":stats["synthetic_diag16_teacher_anchored"]["n_event_anchor_IN_source_only_MENTION_top8"],
              "FP16_params":PAID,"output_gate":0,
          },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
