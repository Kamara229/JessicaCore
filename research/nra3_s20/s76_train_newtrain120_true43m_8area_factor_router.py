"""NRA2 S76: physically fit only 4,232 router+projection parameters on
S75 authentic frozen original43M/GT native H8 S74 NEW TRAIN source90.

S75 original 616 J7, 120×339D CPU native BPE original frozen physical
source from first-read CI is SHA locked. Original S26/S57/S59/S61/S62/
S63/S65 trained adapters are SHA pinned and NEVER trained. All S74
constructed TRAIN event labels are loaded strictly AFTER all source-only
parent inference. Event anchors and multi-hot factors enter ONLY the
TRAIN loss; not the frozen 43M, GreenTunnel, parent source feature
routing, or S72 forward.

S76 trains only the situation ROUTER & source word projection, NOT
the expert effect heads. This tests whether source embeddings encode
event-relative areas. 30 new TRAIN diagnostic lexical-family cases
are observed only once after a fixed 50-epoch fit on other 90 TRAIN;
all ten surface templates are shared, so diagnostic is NOT an
independent natural-language generalization test. Inference event
identity is teacher-anchored only for scoring; missing original S65
candidate slots are counted as a coverage failure, never silently
inferred from gold to the model's forward. No original S65 prediction
can change: all eight expert output corrections remain exactly zero.

No protected S66 spent32, sealed EVAL32/EVAL128, user chat tests,
LIVE, original model update, decoder phase change, or acceptance.
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
from research.nra3_s20.s59_native320_contrastive_event_polarity import semantic_features
from research.nra3_s20.s60_native43m_local_triclass_model import source_local_scope
from research.nra3_s20.s62_joint_event_target_beam2166 import source_only_frozen_beam
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import quote_overlay
from research.nra3_s20.s65_factuality_attributed_event259 import discourse_features
from research.nra3_s20.s73_fit_real43m_old344_microareas import original_frozen_paid
from research.nra3_s20.s72_nra2_situation_microareas import SituationMicroAreas,AREAS
from research.nra3_s20.s74_newtrain120_source_ONLY import (
    new_source_only_train120,source_sha256,GROUP_TRAIN,GROUP_DIAG,
)
# Re-declare exact immutable S75 physical ABI strings rather than import
# its CPU model-capture implementation (which imports transformers and may
# otherwise couple the TRAIN-only reader to original model runtime).
S75_STAGE="NRA2_S75_REAL_ORIGINAL43M_GT_PROTECTED616J7_NEW_TRAIN120_NATIVE_BPE_H8_SOURCE_ONLY"
EXPECTED_S74_SOURCE_SHA="5ff7f5681b0130d4c230fe5b3513038e4d6f3487a3b71964a7ad79e53f50b351"
TENSOR_NAME="NRA2_S75_GENUINE43M_GT_NEWTRAIN120_WORD339_AND_UNSAFE.safetensors"
REPORT_NAME="NRA2_S75_GENUINE43M_GT_NEWTRAIN120_SOURCE_PROVENANCE.json"

STAGE="NRA2_S76_TRUE43M_GT_S74_NEW_TRAIN90_EVENT_RELATIVE_SITUATION_ROUTER"
SOURCE75_SHA="83c1da7b6e79afcfad339c498dcf0f4937411fbbfb081d500d5ca59ea4798900"
SEED=761119
EPOCHS=50
LR=0.008
WD=0.01
TRAIN_WEIGHTS=4232
PHYSICAL="NRA2_S76_TRUE43M_GT_NEWTRAIN90_8AREA_ROUTER9920_FP16.safetensors"
MANIFEST="NRA2_S76_NEWTRAIN120_GROUPED8AREA_ROUTER_MANIFEST.json"


def require(ok,why):
    if not bool(ok):raise ValueError("NRA2 S76 true43M 8area shadow: "+why)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_only_original_parent_features(args):
    """Every source feature computed without importing S74 target module."""
    t=load_file(str(args.s75),device="cpu")
    r=json.loads(args.s75_report.read_text(encoding="utf-8"))
    rows=new_source_only_train120()
    require(args.s75.name==TENSOR_NAME and args.s75_report.name==REPORT_NAME
            and sha(args.s75)==SOURCE75_SHA
            and r["stage"]==S75_STAGE
            and r["physical_source_tensor_SHA256"]==SOURCE75_SHA
            and r["original_protected_616_real_J7_replayed_before_new_S74_capture"]
            and r["S74_SOURCE_ONLY_SHA256"]==source_sha256()==EXPECTED_S74_SOURCE_SHA
            and r["no_S74_construction_labels_imported_or_used_during_any_native_H8"]
            and r["n_new_S74_TRAIN_source_rows"]==len(rows)==120
            and r["output_gate"]==0 and not r["candidate_accepted"],
            "pin real original616 J7, goldblind S75 source120 artifact")
    expected={"original_real43m_gt_word339_READONLY",
              "original_native_word_valid_mask",
              "native_BPE_causally_unsafe_word_mask",
              "source_row_indices_in_frozen_S74_NEW_TRAIN120"}
    require(set(t)==expected
            and t["source_row_indices_in_frozen_S74_NEW_TRAIN120"].tolist()==list(range(120)),
            "no source row rearrangement or extra gold tensor")
    x=t["original_real43m_gt_word339_READONLY"]
    valid=t["original_native_word_valid_mask"]
    unsafe=t["native_BPE_causally_unsafe_word_mask"]
    require(x.ndim==3 and x.shape[0]==120 and x.shape[-1]==339
            and valid.shape==unsafe.shape==x.shape[:2]
            and not bool((unsafe&~valid).any())
            and bool(torch.isfinite(x).all())
            and not x.requires_grad,
            "S75 physically original 43M H8+19 and BPE masks")
    parents,identities=original_frozen_paid(args)
    old26,old57,old59,old61,old62,old63,old65=parents
    texts=tuple(z["text"] for z in rows)
    with torch.no_grad():
        obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
        ev=gold_blind_eventfirst_features(x,obs,texts)
        sem=semantic_features(x,ev)
        fused=source_local_scope(sem,texts)
        logits=tuple(z.detach() for z in old57(ev))
        beam=source_only_frozen_beam(fused,logits,old59,old61)
        overlay=quote_overlay(beam,texts,unsafe)
        state=discourse_features(overlay,texts)
        old_target,old_polar,old_cls=(p.detach().float().clone()
                                      for p in old65(state,old62,old63))
    original=state.older.original
    require(x.shape[1]==old_polar.shape[-1]
            and old_target.shape==(120,original.sources.shape[1],x.shape[1]+1)
            and old_polar.shape==(120,original.sources.shape[1],x.shape[1])
            and old_cls.shape==(120,original.sources.shape[1],3)
            and state.factual_source.shape==(120,original.sources.shape[1],36),
            "S76 original frozen S65 source/features native339 event×target ABI")
    source={
        "word320":x[:,:,:320].detach().clone().half(),
        "source36":state.factual_source.detach().float().clone(),
        "event_indices":original.sources.detach().clone().long(),
        "event_allowed":original.allowed.detach().clone(),
        "target_safe":original.target_safe.detach().clone(),
        "old_target":old_target,
        "old_rejected":old_polar,
        "old_factuality":old_cls,
    }
    require(all(not x.requires_grad for x in source.values())
            and all(torch.isfinite(x).all() for x in source.values()
                    if x.dtype.is_floating_point),
            "NRA2 S76 source-only immutable old 43M+S65 inputs")
    return source,parents,identities,rows


def model_shadow(net,source,*,without_h8=False,without_source36=False):
    w=source["word320"]
    c=source["source36"]
    if without_h8:
        w=torch.zeros_like(w)
    if without_source36:
        c=torch.zeros_like(c)
    return net(c,w,source["event_indices"],
               source["target_safe"],source["event_allowed"],
               source["old_target"],source["old_rejected"],
               source["old_factuality"])


def loss_only_training_anchors(source,rows):
    """ONLY here load construction targets AFTER source parent forwards."""
    from research.nra3_s20.s76_s74_newtrain120_event_relative_gold_TRAIN_LOSS_ONLY import (
        template_constructed_OLDTRAIN_event_area_gold120_for_loss_only,
        preregistration,
    )
    labels=template_constructed_OLDTRAIN_event_area_gold120_for_loss_only()
    require(len(labels)==len(rows)==120
            and [z["row_id"] for z in labels]==[z["row_id"] for z in rows]
            and preregistration()["S74_source_sha256"]==source_sha256(),
            "S76 original source features sealed before any TRAIN label opening")
    gold_events=torch.tensor(
        [z["main_event_raw_index_ONLY_in_TRAIN_loss"] for z in labels],
        dtype=torch.long)
    gold_multi=torch.tensor(
        [z["event_area_multi_hot_ONLY_in_TRAIN_loss"] for z in labels],
        dtype=torch.float32)
    src=source["event_indices"]
    allowed=source["event_allowed"]
    present=(src==gold_events[:,None]) & allowed
    covered=present.any(-1)
    slots=present.long().argmax(-1)
    in_train=torch.tensor([r["family"] in GROUP_TRAIN for r in rows],
                          dtype=torch.bool)
    in_dev=torch.tensor([r["family"] in GROUP_DIAG for r in rows],
                        dtype=torch.bool)
    require(int(in_train.sum())==90 and int(in_dev.sum())==30
            and not bool((in_train&in_dev).any())
            and gold_multi.shape==(120,8)
            and all(z["constructed_not_independent_human_gold"] for z in labels),
            "S76 source-group disjoint 90 train 30 old TRAIN diagnostic")
    return gold_multi,slots,covered,in_train,in_dev


@torch.no_grad()
def area_metrics(router:torch.Tensor,labels:torch.Tensor,slots:torch.Tensor,
                 included:torch.Tensor):
    require(router.ndim==3 and router.shape[0]==120 and router.shape[-1]==8,
            "S76 120x8 original event area output")
    n=torch.where(included)[0]
    require(len(n)>0,"S76 need nonempty actual native event coverage")
    y=labels[n]
    logit=router[n,slots[n]]
    pred=(logit.sigmoid()>=0.5).float()
    tp=int((pred*y).sum())
    fp=int((pred*(1-y)).sum())
    fn=int(((1-pred)*y).sum())
    prec=tp/(tp+fp) if tp+fp else 0.
    rec=tp/(tp+fn) if tp+fn else 0.
    f1=2*prec*rec/(prec+rec) if prec+rec else 0.
    per=[]
    for a in range(8):
        p=pred[:,a];target=y[:,a]
        atp=int((p*target).sum());afp=int((p*(1-target)).sum())
        afn=int(((1-p)*target).sum())
        ap=atp/(atp+afp) if atp+afp else 0.
        ar=atp/(atp+afn) if atp+afn else 0.
        af=2*ap*ar/(ap+ar) if ap+ar else 0.
        per.append({
            "name":AREAS[a],
            "tp":atp,"fp":afp,"fn":afn,
            "f1":af,
        })
    return {
        "scored_teacher_event_source_rows":len(n),
        "micro_precision":prec,"micro_recall":rec,"micro_f1":f1,
        "exact_multilabel_vector_correct":int(pred.eq(y).all(-1).sum()),
        "per_area":per,
        "forward_uses_no_gold_features":True,
        "scored_oracle_event_anchor_not_model_discovery":True,
    }


def learn_only_router(source,gold,slots,covered,train,dev):
    torch.manual_seed(SEED)
    net=SituationMicroAreas()
    require(net.parameter_budget()["new_architecture_parameters"]==9920
            and all(not x.requires_grad for x in source.values()),
            "original 43M/GT/S65 frozen, 9,920 candidate shadow only")
    # Zero-initialized event/pair experts are left untrained on purpose.
    # Only NEW word projection + 8 factor multi-label router learn here.
    for key,p in net.named_parameters():
        p.requires_grad_(key.startswith("word_projection.")
                         or key.startswith("router."))
    trainable=sum(p.numel() for p in net.parameters() if p.requires_grad)
    require(trainable==TRAIN_WEIGHTS,"S76 4232 exclusively trainable router")
    train_rows=train&covered
    diagnostic_rows=dev&covered
    require(int(train_rows.sum())>=40
            and int(diagnostic_rows.sum())>=10,
            "S76 original S65 event beam has inadequate native source coverage")
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),
                          lr=LR,weight_decay=WD)
    pos=gold[train_rows].sum(0)
    n=int(train_rows.sum())
    pos_weight=((n-pos)/pos.clamp_min(1)).clamp(1,8)
    # Deterministic, no checkpoint/threshold searching on dev label rows.
    losses=[]
    for epoch in range(EPOCHS):
        net.train()
        opt.zero_grad(set_to_none=True)
        output=model_shadow(net,source)
        index=torch.where(train_rows)[0]
        logits=output.router_logits[index,slots[index]]
        err=F.binary_cross_entropy_with_logits(
            logits,gold[index],pos_weight=pos_weight)
        require(bool(torch.isfinite(err)),"S76 source-specific factor BCE finite")
        err.backward()
        require(net.router.weight.grad is not None
                and net.word_projection.weight.grad is not None,
                "S76 factor router and H8 projection gradients required")
        nn.utils.clip_grad_norm_(
            (p for p in net.parameters() if p.requires_grad),5.)
        opt.step()
        if epoch in (0,9,24,EPOCHS-1):
            losses.append({"epoch":epoch,"old_train_only_loss":float(err.detach())})
    net.eval()
    with torch.no_grad():
        s=model_shadow(net,source)
        one=area_metrics(s.router_logits,gold,slots,train_rows)
        d=area_metrics(s.router_logits,gold,slots,diagnostic_rows)
        no_h8=area_metrics(
            model_shadow(net,source,without_h8=True).router_logits,
            gold,slots,diagnostic_rows)
        no36=area_metrics(
            model_shadow(net,source,without_source36=True).router_logits,
            gold,slots,diagnostic_rows)
        # Critical: this study never trains experts, so old frozen S65
        # event×target/REJECTED/factuality outputs MUST be exactly
        # unchanged even after router optimization and model save.
        require(torch.equal(s.target_or_none,source["old_target"])
                and torch.equal(s.rejected_edge,source["old_rejected"])
                and torch.equal(s.event_factuality,source["old_factuality"]),
                "S76 router-only phase must preserve original S65 scores EXACT")
        for p in net.areas.parameters():
            require(not p.requires_grad and p.grad is None,
                    "S76 expert correction weights must remain untrained")
    return net,{
        "trainable_router_and_originalH8_projection_parameters":trainable,
        "all_new_serialized_microarea_parameters":9920,
        "fixed_seed":SEED,
        "fixed_hyperparameters":{"epochs":EPOCHS,"lr":LR,"weight_decay":WD},
        "original_source_train90_lexically_grouped_rows":90,
        "original_source_dev30_lexically_grouped_rows":30,
        "train_teacher_event_covered":int(train_rows.sum()),
        "dev_teacher_event_covered":int(diagnostic_rows.sum()),
        "train_event_missing_from_S65_candidate_beam":int((train&~covered).sum()),
        "dev_event_missing_from_S65_candidate_beam":int((dev&~covered).sum()),
        "diagnostic_teacher_forced_event_NOT_end_to_end":True,
        "diagnostic_synthetic_family_holdout_NOT_prospective_EVAL":True,
        "old_train_only_loss_curve":losses,
        "train_router_multilabel":one,
        "old_dev_router_multilabel":d,
        "old_dev_no_true43M_H8_ablation":no_h8,
        "old_dev_no_S65_source36_ablation":no36,
        "original_S65_joint_rejected_factuality_bit_identical":True,
        "old_hypothesis_expert_corrections_unchanged_zero":True,
        "functional_candidate_improvement_NOT_CLAIMED":True,
    }


def main():
    p=argparse.ArgumentParser()
    for key in ("s75","s75_report","s26","s57_paid","s59_paid",
                "s61_paid","s62_paid","s63_paid","s65","s65_manifest","out"):
        p.add_argument("--"+key.replace("_","-"),type=Path,required=True)
    args=p.parse_args()
    require(not args.out.exists(),"refuse overwrite S76 physical sidecar")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source,parents,identity,rows=source_only_original_parent_features(args)
    gold,slots,covered,train,dev=loss_only_training_anchors(source,rows)
    model,diagnostic=learn_only_router(source,gold,slots,covered,train,dev)
    require(all(tensor_sha(*(p.detach() for p in m.parameters()))==digest
                for m,digest in zip(parents,identity)),
            "S76 S26/S57/S59/S61/S62/S63/S65 original frozen parent mutated")
    state={("NRA2_S76_SITUATION."+k):v.detach().half().cpu().contiguous()
           for k,v in model.state_dict().items()}
    require(sum(z.numel() for z in state.values())==9920
            and all(torch.isfinite(z).all() for z in state.values()),
            "S76 FP16 physical router+untrained expert serialization exact")
    args.out.mkdir(parents=True,exist_ok=False)
    artifact=args.out/PHYSICAL
    save_file(state,str(artifact))
    loaded=load_file(str(artifact),device="cpu")
    require(set(loaded)==set(state)
            and all(torch.equal(loaded[k],v) for k,v in state.items()),
            "S76 physical FP16 readback changed native 8area router")
    manifest={
        "stage":STAGE,
        "S74_source_only_SHA":source_sha256(),
        "S75_true43M_native_H8_SHA":sha(args.s75),
        "S76_new_9920_FP16_SHA":sha(artifact),
        "S76_new_synthetic_target_human_independent_verification":False,
        "old_true43M_GT_616J7_original_baseline_previously_physically_replayed":True,
        "no_original_Calibri43M_GT_or_old_S65_weights_changed":True,
        "new_constructed_EVENT_gold_only_training_loss_not_source_forward":True,
        "no_EVAL128_EVAL32_no_S66_SPENT32_gold":True,
        "S72_router_training_ONLY_no_expert_correction_effects":True,
        "old_S65_model_outputs_bit_identical_after_S76":True,
        "not_full_616_J7_S76_candidate_acceptance":True,
        "no_all_in_bpp_CPU_certification":True,
        "not_online_causality_certified":True,
        "diagnostics":diagnostic,
        "output_gate":0,"candidate_accepted":False,"activation_allowed":False,
    }
    (args.out/MANIFEST).write_text(json.dumps(
        manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
        encoding="utf-8")
    print("NRA2_S76_REAL43M_GT_NEWTRAIN90_SITUATION_FACTOR_ROUTER_SHADOW",
          json.dumps({
              "train_covered":diagnostic["train_teacher_event_covered"],
              "dev_covered":diagnostic["dev_teacher_event_covered"],
              "train_micro_F1":diagnostic["train_router_multilabel"]["micro_f1"],
              "dev_micro_F1":diagnostic["old_dev_router_multilabel"]["micro_f1"],
              "dev_noH8_micro_F1":diagnostic["old_dev_no_true43M_H8_ablation"]["micro_f1"],
              "dev_no36_micro_F1":diagnostic["old_dev_no_S65_source36_ablation"]["micro_f1"],
              "original_S65_unchanged":True,
              "gate":0,
          },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
