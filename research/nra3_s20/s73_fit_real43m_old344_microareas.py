"""NRA2 S73: first REAL original43M GT OLD344 S72 micro-area shadow fit.

This is a source-only research TRAIN diagnostic, not an independent
evaluation. Previously-scored S53..S64 OLD344 real original43M/GT
native339 immutable archives and prior trained S65 259FP16 are reused.
All original parent modules are reconstructed physically with SHA.
Nine complete OLD cohorts train (S53..S61, 248 samples); three complete
OLD cohorts (S62..S64, 96 samples) are OLD-TRAIN diagnostics only.
They were used to fit frozen S65, so NOT clean heldout evidence.

Only the 9920 new S72 micro-area parameters are trainable. Router
specialization to its names has NOT yet been proven: there is NO per-area
human TRAIN gold in the old corpus; S73 deliberately does NOT invent
such labels. Gradient-trained shared supervised event×target and rejected
loss is a bootstrap, not proof of mastered compositional situations.
No gold enters inference, all old parent weights remain frozen, no new
held data/QA/EVAL/online phase, no model acceptance.
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

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,tensor_sha,
)
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import (
    S57EventFirstRecipient,gold_blind_eventfirst_features,
)
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    S59NativeEventSemantic,semantic_features,
)
from research.nra3_s20.s60_native43m_local_triclass_model import source_local_scope
from research.nra3_s20.s61_native_pair_null_polarity_2752 import S61NativePairNull
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    S62JointEventRecipient,source_only_frozen_beam,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    S63QuoteOuterSafe,quote_overlay,
)
from research.nra3_s20.s65_factuality_attributed_event259 import (
    S65FactualityJoint,discourse_features,PAID,
)
from research.nra3_s20.s65_fit_old344_original43m_gt_factual259 import (
    FILES,old_spent344,word_idx,
    S26,S57,S59,S61,S62,S63,
    S65_WEIGHT,S65_MANIFEST,
)
from research.nra3_s20.s72_nra2_situation_microareas import (
    SituationMicroAreas,AREAS,
)

STAGE="NRA2_S73_REAL_ORIGINAL43M_GT_FROZEN_S65_OLD344_SITUATION_MICROAREAS_BOOTSTRAP_TRAIN"
GROUPS=tuple(x[0] for x in FILES)
TRAIN=GROUPS[:9]
OLD_DIAG=GROUPS[9:]
assert len(GROUPS)==12 and len(TRAIN)==9 and len(OLD_DIAG)==3
SEED=731119
EPOCHS=42
LR=0.003
WD=0.015
SAVE_KEY_PREFIX="NRA2_S73_MICROAREA."
S65_SHA="4de84bf5ffcffc08c815f13c376b17cf66901370b2c96816d12c92e5bdefc35a"

def require(ok,reason):
    if not bool(ok):
        raise ValueError("NRA2 S73 old original43M GT shadow fit: "+reason)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def sources_and_gold_only_prior_spent():
    groups,gold=old_spent344()
    require(len(groups)==len(gold)==12
            and [len(z) for z in groups]==[24]*5+[32]*7
            and [len(z) for z in gold]==[24]*5+[32]*7,
            "entire S53..S64 old graded-only TRAIN sources")
    return groups,gold

def original_frozen_paid(args):
    files={
        "s26":(S26,EventGroundedInversionGraph.from_physical),
        "s57_paid":(S57,None),"s59_paid":(S59,None),
        "s61_paid":(S61,None),"s62_paid":(S62,None),
        "s63_paid":(S63,None),"s65":(S65_WEIGHT,None),
    }
    for label,(filename,_) in files.items():
        p=getattr(args,label)
        require(p.is_file() and p.name==filename,
                "frozen original paid ancestry "+label)
    old26=EventGroundedInversionGraph.from_physical(args.s26.read_bytes())
    def fp(label,key,cls):
        a=load_file(str(getattr(args,label)),device="cpu")
        require(set(a)=={key}, "unchanged ancestor physical FP16 keys "+label)
        return cls.from_fp16(a[key])
    old57=S57EventFirstRecipient.from_physical_fp16(load_file(
        str(args.s57_paid),device="cpu")["S57_EVENTFIRST_PAID231_FP16"])
    old59=fp("s59_paid","S59_NATIVE320_SEMANTIC2046_FP16",S59NativeEventSemantic)
    old61=fp("s61_paid","S61_REAL43M_GT_PAIR_NULL_POLAR2752_FP16",S61NativePairNull)
    old62=fp("s62_paid","S62_JOINT_EVENT_TARGET2166_FP16",S62JointEventRecipient)
    old63=fp("s63_paid","S63_QUOTE_OUTER_STRICTBPE471_FP16",S63QuoteOuterSafe)
    paid=load_file(str(args.s65),device="cpu")
    require(sha(args.s65)==S65_SHA
            and set(paid)=={"S65_ATTRIBUTED_3WAY259_FP16"}
            and paid["S65_ATTRIBUTED_3WAY259_FP16"].shape==(PAID,),
            "exact genuine original S65 frozen trained259 not replaced")
    old65=S65FactualityJoint.from_fp16(paid["S65_ATTRIBUTED_3WAY259_FP16"])
    manifest=json.loads(args.s65_manifest.read_text())
    require(args.s65_manifest.name==S65_MANIFEST
            and manifest["new_S65_physically_fitted259FP16_SHA256"]==S65_SHA
            and manifest["new_S65_NEW32_raw_and_gold_NEVER_IMPORTED_IN_OLD344_fitting"]
            and [manifest[k]==sha(getattr(args,name)) for name,k in (
                ("s26","frozen_S26_paid_SHA256"),
                ("s57_paid","frozen_S57_paid_SHA256"),
                ("s59_paid","frozen_S59_paid_SHA256"),
                ("s61_paid","frozen_S61_paid_SHA256"),
                ("s62_paid","frozen_S62_paid_SHA256"),
                ("s63_paid","frozen_S63_paid_SHA256"))]==[True]*6,
            "original physical SHA-pinned S65 43M paid ancestry")
    parents=(old26,old57,old59,old61,old62,old63,old65)
    require(all(not p.requires_grad for m in parents for p in m.parameters()),
            "original physically reloaded parents may never train")
    for m in parents:
        m.eval()
    require(all(not m.training for m in parents),
            "entire original paid parent modules frozen eval")
    identity=[tensor_sha(*(p.detach() for p in m.parameters()))
              for m in parents]
    return parents,identity

def reconstruct_train_groups(args,parents):
    old26,old57,old59,old61,old62,old63,old65=parents
    raw,gold=sources_and_gold_only_prior_spent()
    grouped=[]
    for (name,filename,idx_key),source_rows,manual in zip(FILES,raw,gold):
        p=getattr(args,name.lower())
        require(p.name==filename and p.is_file(),
                "old graded real native339 source pinned "+name)
        archive=load_file(str(p),device="cpu")
        require(set(archive)=={
            "original_real43m_gt_word339_READONLY",
            "original_native_word_valid_mask",
            "native_BPE_causally_unsafe_word_mask",idx_key},
            "real GT source must contain NO GOLD or annotations")
        x=archive["original_real43m_gt_word339_READONLY"].detach().clone()
        valid=archive["original_native_word_valid_mask"].detach().clone()
        unsafe=archive["native_BPE_causally_unsafe_word_mask"].detach().clone()
        n=len(source_rows);w=x.shape[1]
        require(x.shape==(n,w,339)
                and valid.shape==unsafe.shape==(n,w)
                and archive[idx_key].tolist()==list(range(n))
                and torch.isfinite(x).all(),
                "original CPU 320+19 native word ABI")
        physical_x=tensor_sha(x,valid,unsafe)
        with torch.no_grad():
            obs=S52QuoteAwareFrozenPaidGraphBridge(old26).observe(x,valid,unsafe)
            ev=gold_blind_eventfirst_features(
                x,obs,tuple(z[2] for z in source_rows))
            sem=semantic_features(x,ev)
            fused=source_local_scope(sem,tuple(z[2] for z in source_rows))
            oldlogits=tuple(t.detach() for t in old57(ev))
            beam=source_only_frozen_beam(fused,oldlogits,old59,old61)
            overlay=quote_overlay(beam,tuple(z[2] for z in source_rows),unsafe)
            state=discourse_features(overlay,tuple(z[2] for z in source_rows))
            joint,polar,classes=(z.detach().float().contiguous()
                                 for z in old65(state,old62,old63))
        original=state.older.original
        require(joint.shape==(n,original.sources.shape[1],w+1)
                and polar.shape==(n,original.sources.shape[1],w)
                and classes.shape==(n,original.sources.shape[1],3),
                "original trained S65 true native joint/polar/class ABI")
        labels_event=torch.full((n,),-1,dtype=torch.long)
        labels_target=torch.full((n,),w,dtype=torch.long)
        labels_rejected=torch.zeros((n,),dtype=torch.bool)
        for i,(record,g) in enumerate(zip(source_rows,manual)):
            require(record[0]==g.row_id,"old human annotation must match source")
            tokens=raw_surface_grammar(record[2])
            require(len(tokens)==int(valid[i].sum()) and len(tokens)<=w,
                    "original GT raw Unicode words physically aligned")
            source=(g.predicate if name in ("S53","S54") else g.main_predicate)
            dest=(g.recipient if name in ("S53","S54")
                  else g.transferred_recipient)
            polar_gold=(g.recipient_polarity if name in ("S53","S54")
                        else g.transfer_polarity if name=="S55"
                        else g.directed_transfer_polarity)
            src=word_idx(tokens,source)
            require(bool(ev.candidate_event[i,src]),"old gold predicate native")
            labels_event[i]=src
            if dest is not None:
                target=word_idx(tokens,dest)
                require(bool(ev.valid_pair[i,src,target]),
                        "old gold native GT transfer target")
                labels_target[i]=target
                labels_rejected[i]=(polar_gold=="REJECTED")
        src=original.sources
        matching=(src==labels_event[:,None])&original.allowed
        has=matching.any(-1)
        slots=matching.long().argmax(-1)
        safe=has&~state.strict_native_or_quote_unsafe
        present=labels_target!=w
        rows=torch.where(safe&present)[0]
        if len(rows):
            require(bool(original.target_safe[
                rows,slots[rows],labels_target[rows]].all()),
                "safe old gold target must be in native BPE candidate space")
        require(tensor_sha(x,valid,unsafe)==physical_x,
                "immutable true original43M GT CPU source changed")
        # Old gold is held only in this separate immutable loss-target
        # wrapper, NEVER in the model's forward feature arguments.
        grouped.append({
            "group":name, "n":n,
            "features":state.factual_source.detach().clone().float(),
            "word320":x[:,:,:320].detach().clone().half(),
            "source_indices":src.detach().clone(),
            "target_safe":original.target_safe.detach().clone(),
            "event_allowed":original.allowed.detach().clone(),
            "old_joint":joint,
            "old_polar":polar,
            "old_cls":classes,
            "gold_slot":slots,
            "gold_recipient":labels_target,
            "gold_rejected":labels_rejected,
            "safe_rows":safe,
            "gold_event_in_beam":int(has.sum()),
            "source_physical_sha256":sha(p),
            "human_gold_used_ONLY_loss":True,
        })
    require(sum(z["n"] for z in grouped)==344
            and [z["group"] for z in grouped]==list(GROUPS)
            and sum(x["safe_rows"].sum().item() for x in grouped)>=200,
            "S73 genuine original43M OLD344 source groups incomplete")
    return grouped

def shadow_forward(net,row):
    return net(row["features"],row["word320"],row["source_indices"],
               row["target_safe"],row["event_allowed"],
               row["old_joint"],row["old_polar"],row["old_cls"])

def loss_on_group(net,row):
    s=shadow_forward(net,row)
    indices=torch.where(row["safe_rows"])[0]
    require(len(indices)>0,"old group has no original BPE-safe gold event")
    b,k,w1=s.target_or_none.shape
    w=w1-1
    slot=row["gold_slot"][indices]
    target=row["gold_recipient"][indices]
    pair_label=slot*w1+target
    flat=s.target_or_none[indices].reshape(-1,k*w1)
    joint=F.cross_entropy(flat,pair_label)
    cls_truth=torch.where(target==w,torch.zeros_like(target),
              torch.where(row["gold_rejected"][indices],
                          torch.full_like(target,2),torch.ones_like(target)))
    class_loss=F.cross_entropy(
        s.event_factuality[indices,slot],cls_truth)
    real=indices[target!=w]
    if len(real):
        negative=F.binary_cross_entropy_with_logits(
            s.rejected_edge[real,row["gold_slot"][real],
                            row["gold_recipient"][real]],
            row["gold_rejected"][real].float())
    else:
        negative=joint*0
    require(bool(torch.isfinite(joint+class_loss+negative)),
            "old source loss numerically finite")
    return joint+.4*class_loss+.5*negative,len(indices)

@torch.no_grad()
def confusion(rows,net=None):
    # Exact same frozen old native real43M S65 logits and S72 inputs;
    # only OLD TRAIN group diagnostics; no independent validation.
    scores={"safe":0,"joint_exact":0,"pred_edges":0,"gold_edges":0,
            "matched_edges":0,"pred_rejected":0,"gold_rejected":0,
            "matched_rejected":0,"nontransfer_false_edges":0,
            "nontransfer_gold":0}
    for r in rows:
        if net is None:
            y,p=r["old_joint"],r["old_polar"]
        else:
            q=shadow_forward(net,r)
            y,p=q.target_or_none,q.rejected_edge
        n,k,w1=y.shape;w=w1-1
        safe=torch.where(r["safe_rows"])[0]
        if not len(safe):continue
        flatten=y[safe].reshape(-1,k*w1).argmax(-1)
        slot=flatten//w1;dest=flatten%w1
        src=r["source_indices"][safe,slot]
        gold_src=r["source_indices"][safe,r["gold_slot"][safe]]
        gold_dest=r["gold_recipient"][safe]
        gold_have=gold_dest!=w
        pred_have=dest!=w
        matches=pred_have&gold_have&(src==gold_src)&(dest==gold_dest)
        neg=torch.zeros_like(pred_have)
        nonnull=torch.where(pred_have)[0]
        if len(nonnull):
            neg[nonnull]=p[safe[nonnull],slot[nonnull],
                            dest[nonnull]]>0
        gold_neg=r["gold_rejected"][safe]&gold_have
        scores["safe"]+=len(safe)
        scores["joint_exact"]+=int(((src==gold_src)&(dest==gold_dest)).sum())
        scores["pred_edges"]+=int(pred_have.sum())
        scores["gold_edges"]+=int(gold_have.sum())
        scores["matched_edges"]+=int(matches.sum())
        scores["pred_rejected"]+=int((pred_have&neg).sum())
        scores["gold_rejected"]+=int(gold_neg.sum())
        scores["matched_rejected"]+=int((matches&neg&gold_neg).sum())
        scores["nontransfer_false_edges"]+=int((pred_have&~gold_have).sum())
        scores["nontransfer_gold"]+=int((~gold_have).sum())
    def ratio(hit,pred,gold):
        prec=hit/pred if pred else 0.
        rec=hit/gold if gold else 0.
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.
        return {"precision":prec,"recall":rec,"f1":f1}
    scores["recipient"]=ratio(scores["matched_edges"],
                               scores["pred_edges"],scores["gold_edges"])
    scores["rejected"]=ratio(scores["matched_rejected"],
                              scores["pred_rejected"],scores["gold_rejected"])
    return scores

def fit(net,groups):
    torch.manual_seed(SEED)
    opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
    history=[]
    train=[r for r in groups if r["group"] in TRAIN]
    dev=[r for r in groups if r["group"] in OLD_DIAG]
    require(len(train)==9 and len(dev)==3
            and sum(x["n"] for x in train)==248
            and sum(x["n"] for x in dev)==96,
            "S73 fixed disjoint old cohort training vs diagnostic")
    before_train=confusion(train)
    before_dev=confusion(dev)
    for epoch in range(EPOCHS):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss=None
        n_safes=0
        # Weighted across OLD TRAIN groups only; never backprop old
        # diagnostic group labels. Fixed epochs, no checkpoint search.
        for g in train:
            v,n=loss_on_group(net,g)
            loss=v*n if loss is None else loss+v*n
            n_safes+=n
        require(loss is not None and n_safes>0,
                "old TRAIN cannot fit with no real native safe labels")
        loss=loss/n_safes
        loss=loss+2.e-6*sum(p.square().sum() for p in net.parameters())
        require(bool(torch.isfinite(loss)),"S73 loss nonfinite")
        loss.backward()
        require(any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                    for p in net.parameters()),"S73 micro-area gradient absent")
        nn.utils.clip_grad_norm_(net.parameters(),max_norm=5.)
        opt.step()
        if epoch in (0,6,13,27,EPOCHS-1):
            history.append({"epoch":epoch,"train_loss":float(loss.detach())})
    net.eval()
    final_train=confusion(train,net)
    final_dev=confusion(dev,net)
    return net,{
        "old_group_TRAIN":list(TRAIN),
        "old_group_TRAIN_rows":248,
        "old_group_DIAGNOSTIC_NOT_INDEPENDENT":list(OLD_DIAG),
        "old_group_DIAGNOSTIC_rows":96,
        "prior_S65_had_seen_ALL_old344_before_S73":True,
        "per_microarea_human_gold_labels_NOT_AVAILABLE":True,
        "named_area_specialization_NOT_CERTIFIED":True,
        "fixed_seed":SEED,
        "frozen_hyperparameters":{"epochs":EPOCHS,"lr":LR,"weight_decay":WD},
        "old_train_unchanged_parent_S65":before_train,
        "old_train_S72_shadow_bootstrap":final_train,
        "old_dev_unchanged_parent_S65":before_dev,
        "old_dev_S72_shadow_bootstrap":final_dev,
        "old_train_loss_trace_only":history,
    }

def main():
    p=argparse.ArgumentParser()
    for name in [s[0].lower() for s in FILES]+[
        "s26","s57_paid","s59_paid","s61_paid","s62_paid","s63_paid",
        "s65","s65_manifest","out"]:
        p.add_argument("--"+name.replace("_","-"),type=Path,required=True)
    a=p.parse_args()
    require(not a.out.exists(),"never overwrite previous S73 archive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parents,identities=original_frozen_paid(a)
    grouped=reconstruct_train_groups(a,parents)
    torch.manual_seed(SEED)
    net=SituationMicroAreas()
    budget=net.parameter_budget()
    require(budget["new_architecture_parameters"]==9920
            and not net.activation_allowed,"S72 exact shadow source ABI")
    net,stats=fit(net,grouped)
    require(all(tensor_sha(*(p.detach() for p in m.parameters()))==h
                for m,h in zip(parents,identities)),
            "parent original Calibri GT S65 trained weights changed")
    with torch.no_grad():
        tensors={SAVE_KEY_PREFIX+k:v.detach().cpu().half().contiguous()
                 for k,v in net.state_dict().items()}
    require(sum(z.numel() for z in tensors.values())==9920
            and all(z.dtype==torch.float16 and torch.isfinite(z).all()
                    for z in tensors.values()),
            "S73 serialization includes every trainable weight")
    a.out.mkdir(parents=True,exist_ok=False)
    paid=a.out/"NRA2_S73_REAL43M_GT_OLD344_SITUATION_MICROAREA9920_FP16.safetensors"
    save_file(tensors,str(paid))
    loaded=load_file(str(paid))
    require(set(loaded)==set(tensors)
            and all(torch.equal(loaded[k],v) for k,v in tensors.items()),
            "S73 actual physical FP16 state readback changed")
    out={
        "stage":STAGE,
        "n_old_physically_captured_training_sources":344,
        "all_old_native_source_S53_S64_SHAs":{
            g["group"]:g["source_physical_sha256"] for g in grouped},
        "frozen_real43M_GT_parent_paid_SHA256":{
            x:sha(getattr(a,x)) for x in (
                "s26","s57_paid","s59_paid","s61_paid","s62_paid",
                "s63_paid","s65")},
        "prior_S65_fit_ALL_old344_and_NEW_S66_spent_NO_REUSE":True,
        "S72_diagnostics":stats,
        "micro_areas":list(AREAS),
        "new_trainable_parameters_exact":9920,
        "physical_fp16_SHA256":sha(paid),
        "fp16_all_new_weights_serialized_exact":True,
        "not_independent_held_or_prospective_EVAL":True,
        "true_J7_integrated_candidate_NOT_measured":True,
        "combined_all_in_bpp_and_CPU_NOT_measured":True,
        "online_causal_generation_NOT_CERTIFIED":True,
        "output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
    }
    manifest=a.out/"NRA2_S73_REAL43M_GT_OLD344_SITUATION_MICROAREA9920_MANIFEST.json"
    manifest.write_text(json.dumps(out,ensure_ascii=False,
                          indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("NRA2_S73_REAL43M_GT_OLD344_TRAIN_ONLY_SHADOW",
          json.dumps({
              "train_rows":248,"old_dev_rows":96,
              "old_dev_parent_recipient_F1":stats["old_dev_unchanged_parent_S65"]["recipient"]["f1"],
              "old_dev_s73_recipient_F1":stats["old_dev_S72_shadow_bootstrap"]["recipient"]["f1"],
              "old_dev_parent_false_NONE":stats["old_dev_unchanged_parent_S65"]["nontransfer_false_edges"],
              "old_dev_s73_false_NONE":stats["old_dev_S72_shadow_bootstrap"]["nontransfer_false_edges"],
              "trained_params":9920,
              "gate":0,
          },sort_keys=True),flush=True)

if __name__=="__main__":
    main()
