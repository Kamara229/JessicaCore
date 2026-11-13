"""CALIBRI NRA2 S3F: first preregistered full-token attention bridge.

Input: S3E physically measured genuine frozen CPU layer12 contextual tokens
[T,320] from 48 synthetic graph prompts. Train first 32, evaluate unchanged
16 six-node novel structural prompts ONCE, after physical FP16 reload. 
NRA2 itself frozen (2241 parameters), zero output gate, no Calibri training.
The reused S3D OOD16 has already been examined and is EXPLORATORY ONLY.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file,save_file

from research.nra2_bridge.stage3b_relational_adapter import (
    physical_frozen_nra,zero_gate_passthrough,
)

SEED=2026111401
STEPS=240
WIDTH=320
H=32
N=6
TRAIN=32
HELD=16


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class TokenAttentionToGraph(nn.Module):
    """Six learned node queries cross-attend to every real contextual token."""
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.LayerNorm(WIDTH),nn.Linear(WIDTH,H))
        self.node_queries=nn.Parameter(torch.zeros(N,H))
        self.attn=nn.MultiheadAttention(H,num_heads=4,dropout=0.,batch_first=True)
        self.node_norm=nn.LayerNorm(H)
        self.edge_bilinear=nn.Parameter(torch.empty(H,H))
        self.endpoint=nn.Linear(H,2)
        nn.init.normal_(self.node_queries,mean=0.,std=.02)
        nn.init.xavier_uniform_(self.edge_bilinear)

    def forward(self,tokens,mask):
        if (tokens.ndim!=3 or tokens.shape[-1]!=WIDTH
                or mask.shape!=tokens.shape[:2] or mask.dtype!=torch.bool
                or not bool(mask.any(dim=1).all())
                or not torch.isfinite(tokens).all()):
            raise AssertionError("S3F must receive masked genuine [B,T,320] token features")
        b,t,_=tokens.shape
        keys=self.project(tokens)
        # Deterministic generalizable position encoding, never labels/answer.
        pos=torch.arange(t,dtype=torch.float32,device=tokens.device)
        freq=torch.exp(-torch.arange(H//2,device=tokens.device).float()
                       * (9.210340371976184/(H//2)))
        theta=pos[:,None]*freq[None,:]
        position=torch.stack((torch.sin(theta),torch.cos(theta)),
                             dim=-1).flatten(1)
        keys=keys+0.12*position.unsqueeze(0)
        q=self.node_queries.unsqueeze(0).expand(b,-1,-1)
        att,_=self.attn(q,keys,keys,key_padding_mask=~mask,need_weights=False)
        nodes=self.node_norm(q+att)
        edge=torch.einsum("bid,df,bjf->bij",nodes,
                          self.edge_bilinear,nodes)/(H**0.5)
        diagonal=torch.eye(N,device=tokens.device,dtype=torch.bool)
        edge=edge.masked_fill(diagonal.unsqueeze(0),-14.)
        endpoints=self.endpoint(nodes)
        return edge,endpoints[:,:,0],endpoints[:,:,1]


def graph_forward(adapter,nra,tokens,mask):
    edges,src,dst=adapter(tokens,mask)
    soft=torch.sigmoid(edges)
    src_prob=F.softmax(src,dim=1)
    dst_prob=F.softmax(dst,dim=1)
    reach=nra(soft,src_prob,dst_prob)
    return edges,src,dst,reach


def split_metrics(adapter,nra,tokens,mask,targets,offset):
    """One final physically-reloaded evaluation, not optimizer feedback."""
    with torch.inference_mode():
        edge,src,dst,reach=graph_forward(adapter,nra,tokens,mask)
        truth=targets["adjacency"][offset:offset+len(tokens)].bool()
        p=(torch.sigmoid(edge)>.5)
        diagonal=torch.eye(N,dtype=torch.bool).unsqueeze(0)
        p=p.masked_fill(diagonal,False)
        tp=int((p&truth).sum())
        fp=int((p&~truth).sum())
        fn=int((~p&truth).sum())
        y=targets["reachability"][offset:offset+len(tokens)].float()
        src_gold=targets["source"][offset:offset+len(tokens)].argmax(dim=1)
        dst_gold=targets["target"][offset:offset+len(tokens)].argmax(dim=1)
        return {
            "count":len(tokens),
            "exact_graphs":int((p==truth).flatten(1).all(dim=1).sum()),
            "source_correct":int((src.argmax(1)==src_gold).sum()),
            "target_correct":int((dst.argmax(1)==dst_gold).sum()),
            "frozen_NRA_reachability_correct":int(((reach>0).float()==y).sum()),
            "edge_F1":float((2*tp)/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else 0.,
            "edge_TP":tp,"edge_FP":fp,"edge_FN":fn,
        }


def training_loss(adapter,nra,tokens,mask,targets):
    """Only indexes [:32] are passed, even though source file has 48 rows."""
    edge,src,dst,reach=graph_forward(adapter,nra,tokens,mask)
    truth=targets["adjacency"][:TRAIN].float()
    src_y=targets["source"][:TRAIN].argmax(dim=1)
    dst_y=targets["target"][:TRAIN].argmax(dim=1)
    reach_y=targets["reachability"][:TRAIN].float()
    offdiag=~torch.eye(N,dtype=torch.bool)
    return (
        1.2*F.binary_cross_entropy_with_logits(
            edge[:,offdiag],truth[:,offdiag],pos_weight=torch.tensor(4.))
        +0.7*F.cross_entropy(src,src_y)
        +0.7*F.cross_entropy(dst,dst_y)
        +0.18*F.binary_cross_entropy_with_logits(reach,reach_y)
    )


def normalize(tokens,mean,scale,mask):
    if (not torch.isfinite(mean).all() or not torch.isfinite(scale).all()
            or not bool((scale>0).all())):
        raise AssertionError("S3F normalized input invalid")
    return ((tokens.float()-mean)/scale).masked_fill(~mask.unsqueeze(-1),0.)


def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: python -m research.nra2_bridge.stage3f_token_attention "
                         "S3E_TOKEN_ARTIFACT_DIR NRA_STANDALONE_DIR OUTPUT_DIR")
    src,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3F research evidence cannot overwrite prior results")
    audit_path=src/"NRA2_S3E_FULL_TOKEN_CPU_GRAPH48_AUDIT.json"
    seq_path=src/"NRA2_S3E_REAL_CPU_GRAPH48_ALL_TOKENS_FP16.safetensors"
    final_path=src/"NRA2_S3E_REAL_43M_GRAPH48_HIDDEN320_FP16.safetensors"
    label_path=src/"NRA2_S3E_GRAPH48_STRUCTURAL_LABELS.safetensors"
    audit=json.loads(audit_path.read_text())
    pretrained=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    frozen_path=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    if (
        audit.get("stage")!="NRA2_S3E_REAL_CPU_FROZEN_FULL_TOKEN_GRAPH48_CAPTURE_ONLY"
        or audit.get("full_token_fp16_sha256")!=sha_file(seq_path)
        or audit.get("physical_feature_fp16_sha256")!=sha_file(final_path)
        or audit.get("physical_graph_labels_sha256")!=sha_file(label_path)
        or audit.get("frozen_independent_NRA_sha256")!=sha_file(frozen_path)
        or pretrained.get("fp16_checkpoint_sha256")!=sha_file(frozen_path)
        or audit.get("all_616_token_digest_matches_S194") is not True
        or audit.get("same_48_final_token_state_as_corrected_S3C") is not True
        or audit.get("physical_reload_bitwise_equal") is not True
        or audit.get("nra_output_gate")!=0.
        or audit.get("train_count")!=TRAIN
        or audit.get("novel_structural_heldout_count")!=HELD
        or len(audit.get("train_prompt_sha256",[]))!=TRAIN
        or len(audit.get("heldout_prompt_sha256",[]))!=HELD
        or set(audit["train_prompt_sha256"])&set(audit["heldout_prompt_sha256"])
    ):
        raise AssertionError("S3F immutable real S3E CPU full token / 616 J7 provenance invalid")
    seq=load_file(str(seq_path),device="cpu")
    labels=load_file(str(label_path),device="cpu")
    final=load_file(str(final_path),device="cpu")["hidden320"]
    tokens=seq["token_hidden320"]
    mask=seq["token_mask"]
    lengths=seq["token_lengths"]
    if (tokens.ndim!=3 or tokens.shape[0]!=TRAIN+HELD
        or tokens.shape[2]!=WIDTH or tokens.dtype!=torch.float16
        or mask.shape!=tokens.shape[:2] or mask.dtype!=torch.bool
        or lengths.shape!=(TRAIN+HELD,)
        or lengths.dtype!=torch.int32
        or list(tokens.shape)!=audit["full_token_feature_shape"]
        or not torch.equal(mask.sum(1).to(torch.int32),lengths)
        or not torch.equal(tokens[torch.arange(TRAIN+HELD),lengths.long()-1],final)
        or not torch.isfinite(tokens).all()
        or set(labels)!={"adjacency","source","target","reachability"}
        or labels["adjacency"].shape!=(TRAIN+HELD,N,N)):
        raise AssertionError("S3F physically reloaded token mask/last hidden/graph ABI invalid")

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    adapter=TokenAttentionToGraph()
    frozen=physical_frozen_nra(frozen_path,pretrained["fp16_checkpoint_sha256"]).eval()
    frozen_snapshot={k:v.clone() for k,v in frozen.state_dict().items()}
    # Per-feature mean and variance ONLY across valid TRAIN tokens.
    valid=tokens[:TRAIN][mask[:TRAIN]].float()
    mean=valid.mean(dim=0)
    scale=valid.std(dim=0,unbiased=False).clamp(min=.05)
    tr=normalize(tokens[:TRAIN],mean,scale,mask[:TRAIN])
    opt=torch.optim.AdamW(adapter.parameters(),lr=.002,weight_decay=.02)
    with torch.inference_mode():
        before=float(training_loss(adapter,frozen,tr,mask[:TRAIN],labels))
    trace=[]
    for step in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss=training_loss(adapter,frozen,tr,mask[:TRAIN],labels)
        if not torch.isfinite(loss):
            raise AssertionError("S3F nonfinite TRAIN loss")
        loss.backward()
        if any(p.grad is not None for p in frozen.parameters()):
            raise AssertionError("S3F mistakenly trained frozen NRA")
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
        opt.step()
        if step+1 in (1,40,80,160,240):
            trace.append({"step":step+1,"train_loss":float(loss.detach())})
    with torch.inference_mode():
        after=float(training_loss(adapter,frozen,tr,mask[:TRAIN],labels))
    if (after>=before or any(not torch.equal(v,frozen_snapshot[k])
                              for k,v in frozen.state_dict().items())):
        raise AssertionError("S3F train failed or pretrained NRA mutated")

    # PRECOMMIT model and TRAIN-only normalization before any OOD scores.
    out.mkdir(parents=True,exist_ok=True)
    weights=out/"NRA2_S3F_REAL_TOKEN_ATTENTION_TRAIN32_FP16.safetensors"
    packed={"bridge."+k:v.detach().cpu().half().contiguous()
            for k,v in adapter.state_dict().items()}
    packed["train_mean320"]=mean.half().contiguous()
    packed["train_scale320"]=scale.half().contiguous()
    save_file(packed,str(weights))
    physical=load_file(str(weights),device="cpu")
    if set(physical)!=set(packed) or any(
        not torch.equal(physical[k],v) for k,v in packed.items()):
        raise AssertionError("S3F physical FP16 reload mismatch")
    reloaded=TokenAttentionToGraph().eval()
    reloaded.load_state_dict({k.removeprefix("bridge."):v
                              for k,v in physical.items() if k.startswith("bridge.")})
    reloaded.requires_grad_(False)
    normalized_train=normalize(tokens[:TRAIN],physical["train_mean320"].float(),
                               physical["train_scale320"].float(),mask[:TRAIN])
    normalized_held=normalize(tokens[TRAIN:],physical["train_mean320"].float(),
                              physical["train_scale320"].float(),mask[TRAIN:])
    train_metrics=split_metrics(reloaded,frozen,normalized_train,mask[:TRAIN],labels,0)
    # ONE exploratory OOD16 evaluation, after frozen FP16 commit.
    held_metrics=split_metrics(reloaded,frozen,normalized_held,mask[TRAIN:],labels,TRAIN)
    with torch.inference_mode():
        original=torch.arange(9,dtype=torch.float32)
        if zero_gate_passthrough(original,torch.zeros(9)) is not original:
            raise AssertionError("S3F zero gate changed original logits")
        oracle=frozen(labels["adjacency"][TRAIN:].float(),
                      labels["source"][TRAIN:].float(),
                      labels["target"][TRAIN:].float())
        oracle_correct=int(((oracle>0).float()==
                            labels["reachability"][TRAIN:].float()).sum())
    if any(not torch.equal(v,frozen_snapshot[k])
           for k,v in frozen.state_dict().items()):
        raise AssertionError("S3F frozen NRA mutated during heldout")
    report={
        "stage":"NRA2_S3F_REAL_TOKEN320_ATTENTION_GRAPH32_OOD16_EXPLORATORY_ONLY",
        "full_real_token_hidden_input":True,
        "S3E_source_run_checked":True,
        "train32":TRAIN,"novel_graph_heldout16":HELD,
        "optimizer_steps":STEPS,
        "train_loss_before":before,"train_loss_after":after,
        "train_loss_trace":trace,
        "train_normalization_only":True,
        "heldout_in_training_or_model_selection":False,
        "reused_S3D_heldout_is_fully_blind":False,
        "heldout_evaluated_once_after_FP16_commit":True,
        "old_S3D_reused_OOD16_exact_graphs":0,
        "old_S3D_reused_OOD16_reachability_correct":8,
        "train_metrics":train_metrics,"heldout_metrics":held_metrics,
        "frozen_NRA_oracle_heldout_correct":oracle_correct,
        "output_gate":0.,
        "original_584_and_616_protected_by_S3E_J7":True,
        "replayed_Calibri_616_in_S3F":False,
        "Calibri_and_Green_Tunnel_unloaded_and_unchanged":True,
        "pretrained_NRA_frozen_unchanged":True,
        "candidate_accepted":False,
        "genuine_new_full_EOS_gains_tested":False,
        "all_in_physical_bpp_tested":False,
        "same_CPU_speed_tested":False,
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
        "fp16_adapter_sha256":sha_file(weights),
        "fp16_adapter_bytes":weights.stat().st_size,
        "physical_reload_bitwise_equal":True,
        "S3E_input_token_sha256":sha_file(seq_path),
        "NRA_standalone_fp16_sha256":sha_file(frozen_path),
    }
    (out/"NRA2_S3F_TOKEN_ATTENTION_32TRAIN_16OOD_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="train_loss_trace"},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
