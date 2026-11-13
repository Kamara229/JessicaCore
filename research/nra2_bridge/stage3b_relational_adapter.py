"""NRA2 S3B: independent, TRAIN-only frozen-core compatibility experiment.

The 32 S190 FP16 real hidden vectors (S3A) are split *before training*
into 24 TRAIN and 8 held-out (6/2 per prompt family).

This is a deliberately narrow binary relational *proxy*: whether the
prompt's declared requested owner is B (rather than A), or whether the
numbers in a numeric comparison are in descending order. It is NOT a
hidden-to-text or general graph parser and cannot claim new full-EOS answers.
The separately pretrained graph NRA is frozen; only the newly added hidden320
to 2-edge adapter learns. Output gate is always exactly zero. LIVE untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from research.nra2_standalone.model import RelationalNRA

NUMERIC_PAIRS = ((412,409),(300,301),(1001,999),(78,79),
                 (101,100),(23,25),(999,2),(2,900))
FAMILIES = ("owner_A_paraphrase","owner_B_paraphrase",
            "reordered_owner_paraphrase","numeric_new_wording")
HIDDEN_WIDTH = 320


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def partition_proxy_targets(audit: dict, s190: dict):
    """Use only the frozen S190 TRAIN family and numeric source construction.

    Proxy labels are metadata-derived and not read by inference or the
    model's forward. Group-balanced 24/8 SHA split prevents selection using
    held-out errors, and proxy results are explicitly NOT linguistic ability.
    """
    expected = audit["dataset_question_sha256_sorted"]
    if len(expected) != 32 or len(set(expected)) != 32 or expected != sorted(expected):
        raise AssertionError("S3A exactly 32 distinct sorted TRAIN prompt SHA required")
    by_hash = {}
    for row in s190["hashed_per_prompt_audit"]:
        key, group = row["prompt_sha256"], row["group"]
        if key in by_hash or group not in FAMILIES:
            raise AssertionError("S190 frozen SHA/group collision or unexpected family")
        by_hash[key] = group
    if set(by_hash) != set(expected):
        raise AssertionError("S190 real TRAIN prompt SHA drift from S3A FP16 archive")

    numeric = {sha_text(f"Answer Yes or No: does {a} exceed {b}?"):int(a>b)
               for a,b in NUMERIC_PAIRS}
    if len(numeric) != 8:
        raise AssertionError("Numeric source identity collision")
    if {k for k,v in by_hash.items() if v == "numeric_new_wording"} != set(numeric):
        raise AssertionError("Numeric 8 inputs no longer match pinned S190 source")
    groups = {g: sorted(k for k,v in by_hash.items() if v == g) for g in FAMILIES}
    if any(len(keys) != 8 for keys in groups.values()):
        raise AssertionError("Expected exactly eight TRAIN probes per family")

    train_sha = [k for g in FAMILIES for k in groups[g][:6]]
    held_sha = [k for g in FAMILIES for k in groups[g][6:]]
    if len(train_sha) != 24 or len(held_sha) != 8 or set(train_sha)&set(held_sha):
        raise AssertionError("S3B required disjoint balanced 24/8 split")
    label = {k:(numeric[k] if by_hash[k]=="numeric_new_wording"
                else int(by_hash[k]=="owner_B_paraphrase")) for k in expected}
    return train_sha, held_sha, label


class Hidden320ToRelationalEdges(nn.Module):
    """Actual differentiable [B,320] -> 2 directed soft edges for frozen NRA.

    N=2 is a constrained ABI smoke proxy: the graph-trained area originally
    learned 5-hop chains, NOT natural language item/owner understanding.
    """
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(HIDDEN_WIDTH),
                               nn.Linear(HIDDEN_WIDTH,32),nn.GELU(),
                               nn.Linear(32,2))

    def forward(self, h: torch.Tensor):
        if h.ndim!=2 or h.shape[1]!=HIDDEN_WIDTH or not torch.isfinite(h).all():
            raise AssertionError("S3B actual finite hidden320 required")
        raw=self.net(h)
        edges=torch.zeros((h.shape[0],2,2),dtype=h.dtype,device=h.device)
        edges[:,0,1]=torch.sigmoid(raw[:,0])
        edges[:,1,0]=torch.sigmoid(raw[:,1])
        source=torch.zeros((h.shape[0],2),dtype=h.dtype,device=h.device)
        target=torch.zeros_like(source)
        source[:,0]=1.
        target[:,1]=1.
        return edges,source,target,raw


def zero_gate_passthrough(original_logits:torch.Tensor, nra_signal:torch.Tensor,gate:float=0.):
    """Never modify the Calibri output until a separately accepted active gate."""
    if gate!=0.0:
        raise AssertionError("S3B is shadow-only; output gate cannot be active")
    if not torch.isfinite(nra_signal).all():
        raise AssertionError("Invalid NRA shadow prediction")
    return original_logits


def physical_frozen_nra(checkpoint:Path,expected_sha:str)->RelationalNRA:
    if sha_file(checkpoint)!=expected_sha:
        raise AssertionError("Independent NRA frozen FP16 physical SHA drift")
    state=torch.load(checkpoint,map_location="cpu",weights_only=True)
    area=RelationalNRA(width=16,hops=5).to("cpu").eval()
    if set(state)!=set(area.state_dict()) or any(
        x.dtype!=torch.float16 or not torch.isfinite(x).all() for x in state.values()
    ):
        raise AssertionError("NRA2 physical FP16 weights invalid")
    area.load_state_dict(state,strict=True)
    area.requires_grad_(False)
    return area


def frozen_proxy_forward(adapter:Hidden320ToRelationalEdges,area:RelationalNRA,
                         features:torch.Tensor):
    edges,source,target,raw=adapter(features)
    return raw[:,0]-raw[:,1],area(edges,source,target)


def main():
    if len(sys.argv)!=5:
        raise SystemExit("usage: python -m research.nra2_bridge.stage3b_relational_adapter "
                         "S3A_ARTIFACT_DIRECTORY NRA_STANDALONE_DIRECTORY OUTPUT_DIRECTORY")
    src,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Research output cannot overwrite existing evidence")
    audit=json.loads((src/"NRA2_S3A_REAL_CPU_32_TRAIN_FEATURE_AUDIT.json").read_text())
    s190=json.loads((src/"S190_REAL_CPU_32_NEW_TRAIN_PARAPHRASE_PROBES.json").read_text())
    standalone=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    frozen_fp16=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    features_path=src/"NRA2_S3A_REAL_CPU_32_TRAIN_HIDDEN320_FP16.safetensors"
    if (audit["stage"]!="NRA2_S3A_REAL_43M_CPU_TRAIN32_LAST_LAYER_FEATURES_ONLY"
            or audit["feature_sha256"]!=sha_file(features_path)
            or audit["feature_reload_bitwise_equal"] is not True
            or audit["feature_shape"]!=[32,320]
            or audit["existing_456_full_EOS"]!=456
            or audit["additional_128_full_EOS"]!=128
            or audit["protected_584_digest_matches_S194_and_S2"] is not True
            or audit["new32_digest_matches_S194_and_S2"] is not True
            or audit["NRA2_gate"]!=0.
            or standalone["fp16_checkpoint_sha256"]!=audit["standalone_NRA2_trained_FP16_sha256"]):
        raise AssertionError("S3B source provenance/true CPU J7 frozen gate failure")
    all_h=load_file(str(features_path),device="cpu")["final_hidden_320"]
    if (all_h.shape!=(32,320) or all_h.dtype!=torch.float16
            or not torch.isfinite(all_h).all()):
        raise AssertionError("S3A physical real CPU features invalid")
    train_sha,held_sha,labels=partition_proxy_targets(audit,s190)
    ix={key:i for i,key in enumerate(audit["dataset_question_sha256_sorted"])}
    train=torch.stack([all_h[ix[k]].float() for k in train_sha])
    held=torch.stack([all_h[ix[k]].float() for k in held_sha])
    train_y=torch.tensor([labels[k] for k in train_sha],dtype=torch.float32)
    held_y=torch.tensor([labels[k] for k in held_sha],dtype=torch.float32)
    # Derive all normalization exclusively from 24 TRAIN, not held-out.
    mean=train.mean(0)
    scale=train.std(0,unbiased=False).clamp(min=1e-3)
    train=(train-mean)/scale
    held=(held-mean)/scale

    torch.manual_seed(20261113)
    torch.set_num_threads(2)
    adapter=Hidden320ToRelationalEdges().to("cpu")
    area=physical_frozen_nra(frozen_fp16,standalone["fp16_checkpoint_sha256"])
    frozen_snapshot={k:v.detach().clone() for k,v in area.state_dict().items()}
    opt=torch.optim.AdamW(adapter.parameters(),lr=0.003,weight_decay=0.001)
    def losses(h,y):
        proxy,neural=frozen_proxy_forward(adapter,area,h)
        return F.binary_cross_entropy_with_logits(proxy,y)+0.25*F.binary_cross_entropy_with_logits(neural,y)
    with torch.no_grad():
        before=float(losses(train,train_y))
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss=losses(train,train_y)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite proxy training loss")
        loss.backward()
        if any(p.grad is not None for p in area.parameters()):
            raise AssertionError("Frozen pretrained graph NRA received gradients")
        opt.step()
    with torch.no_grad():
        after=float(losses(train,train_y))
        validation=float(losses(held,held_y))  # evaluate ONCE; never tune on held8
        proxy,_=frozen_proxy_forward(adapter,area,held)
        held_correct=int(((proxy>0).float()==held_y).sum())
        baseline=torch.randn(2,320)
        neural_probe=torch.zeros(2)
        if zero_gate_passthrough(baseline,neural_probe) is not baseline:
            raise AssertionError("Zero-gate original outputs lost identity")
    if after>=before or any(not torch.equal(v,frozen_snapshot[k])
                            for k,v in area.state_dict().items()):
        raise AssertionError("No learned sidecar improvement or frozen NRA mutated")
    if any(not torch.isfinite(v).all() for v in adapter.state_dict().values()):
        raise AssertionError("NRA2 candidate adapter nonfinite")
    out.mkdir(parents=True,exist_ok=True)
    weights=out/"NRA2_S3B_HIDDEN320_TO_GRAPH_PROXY_TRAIN24_FP16.safetensors"
    serialized={k:v.detach().cpu().to(torch.float16).contiguous()
                for k,v in adapter.state_dict().items()}
    save_file(serialized,str(weights))
    reload=load_file(str(weights),device="cpu")
    if set(reload)!=set(serialized) or any(not torch.equal(reload[k],v)
                                           for k,v in serialized.items()):
        raise AssertionError("S3B physical FP16 adapter reload mismatch")
    report={
        "stage":"NRA2_S3B_INDEPENDENT_HIDDEN320_GRAPH_ABI_PROXY_TRAIN24_HOLDOUT8",
        "newly_trained_adapter":True,
        "semantic_relational_graph_alignment_confirmed":False,
        "training_objective":"owner-B-vs-A and descending-numeric two-node directed-edge proxy ONLY",
        "actual_43M_hidden_states":True,
        "train24_count":24,"heldout8_count":8,
        "heldout_split_selected_before_training":True,
        "no_heldout_in_optimizer_or_normalizer":True,
        "train24_hashes":train_sha,"heldout8_hashes":held_sha,
        "train_loss_before":before,"train_loss_after":after,
        "heldout_proxy_loss":validation,
        "heldout_direct_edge_accuracy":held_correct,
        "pretrained_nra_frozen_unchanged":True,
        "underlying_calibri_43M_frozen_and_unloaded":True,
        "output_gate":0.0,
        "original_584_replayed_in_S3B":False,
        "original_S3A_584_unchanged_evidence_pinned":True,
        "source_32_new_full_EOS":audit["new32_full_EOS"],
        "S3B_real_CPU_new32_or_candidate_bpp_tested":False,
        "candidate_model_accepted":False,
        "original_s3a_features_sha256":audit["feature_sha256"],
        "original_standalone_fp16_sha256":standalone["fp16_checkpoint_sha256"],
        "sidecar_physical_fp16_sha256":sha_file(weights),
        "sidecar_physical_fp16_bytes":weights.stat().st_size,
        "sidecar_fp16_reload_verified":True,
        "only_research_no_LIVE_Render_Supabase_APK":True,
    }
    (out/"NRA2_S3B_24TRAIN_8HELDOUT_PROXY_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("train24_hashes","heldout8_hashes")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
