"""S3D: freeze real 43M Calibri and pretrained NRA, train only graph bridge.

Use immutable S3C 48 real CPU layer12 hidden320 / independently grounded graph
pairs. Exactly first 32 4/5-node TRAIN are visible during 240 predeclared
optimizer steps; final 16 6-node OOD controls are touched only ONCE after
physical FP16 save+reload. No model selection on holdout, no QA/TEST.
No decoder changes, zero gate, no LIVE/Render/Supabase/APK/main.
Fail-closed: falling TRAIN loss cannot certify semantic generalization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file, load_file

from research.nra2_standalone.model import RelationalNRA
from research.nra2_bridge.stage3b_relational_adapter import (
    physical_frozen_nra, zero_gate_passthrough,
)

SEED=2026111304
STEPS=240
TRAIN=32
HELD=16
WIDTH=320
N=6


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class Hidden320GraphBridge(nn.Module):
    """One prompt vector -> 6x6 directed edge logits + endpoint logits.

    24-wide bottleneck makes a small but still potentially overfitting
    320D/48D supervised network. Do NOT infer generalization from TRAIN.
    """
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.LayerNorm(WIDTH),
            nn.Linear(WIDTH,24),
            nn.GELU(),
            nn.Linear(24,N*N+N+N),
        )

    def forward(self,hidden):
        if (hidden.ndim!=2 or hidden.shape[1]!=WIDTH
                or not torch.isfinite(hidden).all()):
            raise AssertionError("S3D requires finite real hidden320")
        raw=self.net(hidden)
        return (raw[:,:N*N].reshape(-1,N,N),
                raw[:,N*N:N*N+N],
                raw[:,N*N+N:])


def normalized_hidden(features,mean,scale):
    if not (torch.isfinite(mean).all() and torch.isfinite(scale).all()
            and torch.all(scale>0)):
        raise AssertionError("TRAIN-only normalizer invalid")
    return (features.float()-mean)/scale


def logits_from_bridge(bridge,area,hidden):
    edge_logits,source_logits,target_logits=bridge(hidden)
    mask=~torch.eye(N,dtype=torch.bool,device=hidden.device)
    soft_edges=torch.sigmoid(edge_logits).masked_fill(~mask[None],0.)
    src_prob=F.softmax(source_logits,dim=1)
    dst_prob=F.softmax(target_logits,dim=1)
    area_logits=area(soft_edges,src_prob,dst_prob)
    return edge_logits,source_logits,target_logits,area_logits


def original_train_loss(bridge,area,hidden,labels):
    """Multi-task loss uses exactly 32 TRAIN labels, never heldout."""
    edge,src,dst,nra=logits_from_bridge(bridge,area,hidden)
    mask=~torch.eye(N,dtype=torch.bool,device=hidden.device)
    edges=labels["adjacency"][:TRAIN].float()
    src_gold=labels["source"][:TRAIN].argmax(dim=1)
    dst_gold=labels["target"][:TRAIN].argmax(dim=1)
    y=labels["reachability"][:TRAIN].float()
    return (
        1.2*F.binary_cross_entropy_with_logits(
            edge[:,mask],edges[:,mask],pos_weight=torch.tensor(4.))
        +0.7*F.cross_entropy(src,src_gold)
        +0.7*F.cross_entropy(dst,dst_gold)
        +0.18*F.binary_cross_entropy_with_logits(nra,y)
    )


def metrics_for_split(bridge,area,h,labels,start):
    """Gold labels enter POSTHOC scoring only, not bridge forward."""
    with torch.inference_mode():
        a,s,d,r=logits_from_bridge(bridge,area,h)
        offdiag=~torch.eye(N,dtype=torch.bool)
        e=(torch.sigmoid(a)>0.5).masked_fill(~offdiag[None],False)
        truth=labels["adjacency"][start:start+len(h)].bool()
        src_gt=labels["source"][start:start+len(h)].argmax(dim=1)
        dst_gt=labels["target"][start:start+len(h)].argmax(dim=1)
        reach=labels["reachability"][start:start+len(h)].float()
        tp=int((e&truth).sum())
        fp=int((e&~truth).sum())
        fn=int((~e&truth).sum())
        return {
            "count":len(h),
            "directed_edges_true_positive":tp,
            "directed_edges_false_positive":fp,
            "directed_edges_false_negative":fn,
            "directed_edge_f1":(2*tp)/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "exact_graphs":int((e==truth).flatten(1).all(dim=1).sum()),
            "source_identity_correct":int((s.argmax(dim=1)==src_gt).sum()),
            "target_identity_correct":int((d.argmax(dim=1)==dst_gt).sum()),
            "frozen_NRA_soft_graph_reachability_correct":int(
                ((r>0).float()==reach).sum()),
        }


def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: python -m research.nra2_bridge.stage3d_graph_bridge "
                         "S3C_ARTIFACT_DIRECTORY NRA_STANDALONE_DIRECTORY OUTPUT_DIR")
    src,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3D must never overwrite research result")
    audit_path=src/"NRA2_S3C_GRAPH48_REAL_CPU_PAIRS_AUDIT.json"
    feat_path=src/"NRA2_S3C_REAL_43M_GRAPH48_HIDDEN320_FP16.safetensors"
    target_path=src/"NRA2_S3C_GRAPH48_STRUCTURAL_LABELS.safetensors"
    report=json.loads(audit_path.read_text())
    pre=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    frozen_nra=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    if (
        report.get("stage")!="NRA2_S3C_REAL_CPU_FROZEN_48_GRAPH_STRUCTURAL_PAIRS_ONLY"
        or report.get("all_616_token_digest_matches_S194") is not True
        or report.get("original_456_full_EOS")!=456
        or report.get("original_new128_full_EOS")!=128
        or report.get("new32_full_EOS")!=0
        or report.get("nra_output_gate")!=0.
        or report.get("real_hidden320_shape")!=[48,320]
        or report.get("real_native_CPU_43M_layer12_forward") is not True
        or report.get("heldout_topologies_and_prompt_templates_disjoint") is not True
        or report.get("physical_reload_bitwise_equal") is not True
        or report.get("train_count")!=TRAIN
        or report.get("novel_structural_heldout_count")!=HELD
        or report.get("physical_feature_fp16_sha256")!=sha_file(feat_path)
        or report.get("physical_graph_labels_sha256")!=sha_file(target_path)
        or report.get("frozen_independent_NRA_sha256")!=sha_file(frozen_nra)
        or pre.get("fp16_checkpoint_sha256")!=sha_file(frozen_nra)
        or len(report.get("train_prompt_sha256",[]))!=TRAIN
        or len(report.get("heldout_prompt_sha256",[]))!=HELD
        or set(report["train_prompt_sha256"])&set(report["heldout_prompt_sha256"])
    ):
        raise AssertionError("S3D pinned real S3C J7 + fully separated OOD source invalid")

    features=load_file(str(feat_path),device="cpu")["hidden320"]
    labels=load_file(str(target_path),device="cpu")
    if (features.shape!=(TRAIN+HELD,WIDTH)
        or features.dtype!=torch.float16
        or not torch.isfinite(features).all()
        or set(labels)!={"adjacency","source","target","reachability"}
        or labels["adjacency"].shape!=(TRAIN+HELD,N,N)
        or labels["source"].shape!=(TRAIN+HELD,N)
        or labels["target"].shape!=(TRAIN+HELD,N)
        or labels["reachability"].shape!=(TRAIN+HELD,)
        or labels["adjacency"].dtype!=torch.float16
        or not torch.all(labels["adjacency"].diagonal(dim1=-2,dim2=-1)==0)):
        raise AssertionError("S3D physically loaded 48 pair tensors invalid")

    # Everything below through optimizer step uses first 32 TRAIN only.
    train_features=features[:TRAIN].float()
    mean=train_features.mean(dim=0)
    scale=train_features.std(dim=0,unbiased=False).clamp(min=0.05)
    train=normalized_hidden(train_features,mean,scale)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    bridge=Hidden320GraphBridge().cpu().train()
    area=physical_frozen_nra(frozen_nra,pre["fp16_checkpoint_sha256"]).cpu().eval()
    frozen_snapshot={k:v.detach().clone() for k,v in area.state_dict().items()}
    optimizer=torch.optim.AdamW(bridge.parameters(),lr=.002,weight_decay=.02)
    with torch.no_grad():
        before=float(original_train_loss(bridge,area,train,labels))
    track=[]
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss=original_train_loss(bridge,area,train,labels)
        if not torch.isfinite(loss):
            raise AssertionError("S3D nonfinite real graph TRAIN loss")
        loss.backward()
        if any(p.grad is not None for p in area.parameters()):
            raise AssertionError("S3D frozen NRA received gradients")
        torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.0)
        optimizer.step()
        if step+1 in (1,40,80,160,240):
            track.append({"step":step+1,"train_loss":float(loss.detach())})
    with torch.no_grad():
        after=float(original_train_loss(bridge,area,train,labels))
    if (after>=before
        or any(not torch.equal(v,frozen_snapshot[k])
               for k,v in area.state_dict().items())
        or not all(torch.isfinite(v).all() for v in bridge.state_dict().values())):
        raise AssertionError("S3D real graph training or frozen NRA guard failed")

    # Physical artifact written before looking at any 16 OOD labels/scores.
    out.mkdir(parents=True,exist_ok=True)
    weights=out/"NRA2_S3D_REAL_HIDDEN320_GRAPH48_TRAIN32_FP16.safetensors"
    persist={"bridge."+k:v.detach().cpu().to(torch.float16).contiguous()
             for k,v in bridge.state_dict().items()}
    persist["train_mean320"]=mean.to(torch.float16).contiguous()
    persist["train_scale320"]=scale.to(torch.float16).contiguous()
    save_file(persist,str(weights))
    reloaded=load_file(str(weights),device="cpu")
    if set(reloaded)!=set(persist) or any(
        not torch.equal(reloaded[k],v) for k,v in persist.items()
    ):
        raise AssertionError("S3D physical FP16 bridge reload not exact")

    # Evaluate physically reloaded trained model, NOT in-memory FP32 clone.
    physical=Hidden320GraphBridge().eval().cpu()
    physical.load_state_dict({k.removeprefix("bridge."):v
                              for k,v in reloaded.items()
                              if k.startswith("bridge.")},strict=True)
    physical.requires_grad_(False)
    scale32=reloaded["train_scale320"].float()
    mean32=reloaded["train_mean320"].float()
    if (not torch.all(scale32>0) or any(
        not torch.equal(v,frozen_snapshot[k])
        for k,v in area.state_dict().items())
    ):
        raise AssertionError("Frozen pretrained graph core changed or bad FP16 scale")
    hold=normalized_hidden(features[TRAIN:].float(),mean32,scale32)
    train_check=normalized_hidden(features[:TRAIN].float(),mean32,scale32)
    train_metrics=metrics_for_split(physical,area,train_check,labels,0)
    # FIRST and ONLY heldout evaluation: no optimizer or hyperparam selection.
    held_metrics=metrics_for_split(physical,area,hold,labels,TRAIN)
    with torch.inference_mode():
        oracle_logits=area(labels["adjacency"][TRAIN:].float(),
                           labels["source"][TRAIN:].float(),
                           labels["target"][TRAIN:].float())
        oracle_correct=int(
            ((oracle_logits>0).float()==labels["reachability"][TRAIN:].float()).sum())
        original=torch.arange(7,dtype=torch.float32)
        if zero_gate_passthrough(original,torch.zeros(7)) is not original:
            raise AssertionError("S3D absolute original output identity changed")

    result={
        "stage":"NRA2_S3D_REAL_HIDDEN320_GRAPH48_BRIDGE_TRAIN32_OOD16_SHADOW_ONLY",
        "real_calibri_cpu_hidden320_inputs":True,
        "train32_count":TRAIN,"untouched_6_node_structural_heldout16_count":HELD,
        "train_only_normalization":True,
        "no_heldout_in_optimizer_or_model_selection":True,
        "fixed_steps_before_any_heldout_scoring":STEPS,
        "real_training_loss_before":before,"real_training_loss_after":after,
        "train_loss_trace":track,
        "physically_reloaded_FP16_before_holdout":True,
        "frozen_2241p_NRA_unchanged":True,
        "calibri_43M_and_Green_Tunnel_frozen_unloaded":True,
        "output_gate":0.0,
        "original_S3C_584_unchanged_J7_pinned":True,
        "original_S3C_all_616_identical_to_S194_pinned":True,
        "S3D_replayed_original_616_decoder":False,
        "train_metrics":train_metrics,
        "heldout_metrics":held_metrics,
        "frozen_graph_NRA_oracle_heldout_correct":oracle_correct,
        "naive_heldout_majority_class_accuracy":0.5,
        "semantic_generalization_confirmed":(
             held_metrics["exact_graphs"]==HELD
             and held_metrics["source_identity_correct"]==HELD
             and held_metrics["target_identity_correct"]==HELD
             and held_metrics["frozen_NRA_soft_graph_reachability_correct"]==HELD
             and oracle_correct==HELD),
        "candidate_model_accepted":False,
        "new_full_EOS_gains_tested":False,
        "all_in_physical_bpp_tested":False,
        "same_CPU_speed_tested":False,
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
        "fp16_adapter_bytes":weights.stat().st_size,
        "fp16_adapter_sha256":sha_file(weights),
        "original_S3C_feature_sha256":sha_file(feat_path),
        "original_S3C_graph_label_sha256":sha_file(target_path),
        "original_NRA_frozen_checkpoint_sha256":sha_file(frozen_nra),
    }
    (out/"NRA2_S3D_REAL_GRAPH_BRIDGE_TRAIN32_OOD16_AUDIT.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="train_loss_trace"},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
