"""NRA2 S3I: fixed-architecture S3F bridge, 64 TRAIN, 4x16 controlled shifts.

Intentionally NOT a new larger architecture. Distinguishes unseen topology,
wording, graph length and combined shift without using the four controls to
train or select hyperparameters. Freeze NRA; no Calibri/GT model loaded here.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from research.nra2_bridge.stage3f_token_attention import (
    TokenAttentionToGraph,graph_forward,split_metrics,normalize,
)
from research.nra2_bridge.stage3b_relational_adapter import (
    physical_frozen_nra,zero_gate_passthrough,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import GROUP_SIZES

SEED=2026111401  # same seed as S3F, to isolate data change
STEPS=240         # same as S3F
TRAIN=64
TOTAL=128
NODES=6
WIDTH=320


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_loss(model,nra,tokens,mask,gold):
    """Caller supplies ONLY the 64 training rows of gold and token states."""
    if tokens.shape[0]!=TRAIN or mask.shape[0]!=TRAIN or gold["adjacency"].shape[0]!=TRAIN:
        raise AssertionError("S3I cannot compute loss on any control example")
    edge,src,dst,reach=graph_forward(model,nra,tokens,mask)
    truth=gold["adjacency"].float()
    src_y=gold["source"].argmax(dim=1)
    dst_y=gold["target"].argmax(dim=1)
    reach_y=gold["reachability"].float()
    offdiag=~torch.eye(NODES,dtype=torch.bool)
    return (
        1.2*F.binary_cross_entropy_with_logits(
            edge[:,offdiag],truth[:,offdiag],pos_weight=torch.tensor(4.))
        +0.7*F.cross_entropy(src,src_y)
        +0.7*F.cross_entropy(dst,dst_y)
        +0.18*F.binary_cross_entropy_with_logits(reach,reach_y)
    )


def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: python -m research.nra2_bridge.stage3i_controlled_attention "
                         "S3H_REAL_TOKEN_DIR NRA_STANDALONE_DIR OUTPUT_DIR")
    src,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3I immutable research results cannot be overwritten")
    audit_path=src/"NRA2_S3H_128_FACTORIAL_REAL_CPU_AUDIT.json"
    seq_path=src/"NRA2_S3H_REAL_CPU_GRAPH128_ALL_TOKENS_FP16.safetensors"
    hidden_path=src/"NRA2_S3H_REAL_43M_GRAPH128_HIDDEN320_FP16.safetensors"
    labels_path=src/"NRA2_S3H_GRAPH128_STRUCTURAL_LABELS.safetensors"
    checkpoint=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    audit=json.loads(audit_path.read_text())
    pretrained=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    groups=audit.get("split_groups",{})
    if (audit.get("stage")!="NRA2_S3H_REAL_CPU_FROZEN_128_FACTORIAL_TOKEN_CAPTURE_ONLY"
        or audit.get("S3E_successful_parent_run")!=43512049506
        or audit.get("full_token_fp16_sha256")!=sha_file(seq_path)
        or audit.get("physical_feature_fp16_sha256")!=sha_file(hidden_path)
        or audit.get("physical_graph_labels_sha256")!=sha_file(labels_path)
        or audit.get("frozen_independent_NRA_sha256")!=sha_file(checkpoint)
        or pretrained.get("fp16_checkpoint_sha256")!=sha_file(checkpoint)
        or audit.get("all_616_token_digest_matches_S194") is not True
        or audit.get("physical_reload_bitwise_equal") is not True
        or audit.get("labels_materialized_after_native_CPU") is not True
        or audit.get("control_groups_scored") is not False
        or audit.get("adapter_trained") is not False
        or audit.get("nra_output_gate")!=0.
        or audit.get("train_count")!=TRAIN
        or audit.get("group_sizes")!=GROUP_SIZES
        or set(groups)!=set(GROUP_SIZES)
        or audit.get("original_456_full_EOS")!=456
        or audit.get("original_new128_full_EOS")!=128
        or audit.get("candidate_accepted") is not False):
        raise AssertionError("S3I S3H immutable real full-token source, groups or J7 provenance invalid")

    index=0
    all_sha=[]
    for name,count in GROUP_SIZES.items():
        block=groups[name]
        if (block.get("start")!=index or block.get("count")!=count
            or block.get("positive")!=count//2
            or len(block.get("prompt_sha256",[]))!=count):
            raise AssertionError("S3I split group ordering, balance or SHA invalid")
        all_sha.extend(block["prompt_sha256"])
        index+=count
    if index!=TOTAL or len(set(all_sha))!=TOTAL:
        raise AssertionError("S3I duplicate prompts across TRAIN/CTRL")

    saved=load_file(str(seq_path),device="cpu")
    tokens=saved["token_hidden320"]
    mask=saved["token_mask"]
    lengths=saved["token_lengths"]
    hidden=load_file(str(hidden_path),device="cpu")["hidden320"]
    if (tokens.ndim!=3 or tokens.shape[0]!=TOTAL or tokens.shape[-1]!=WIDTH
        or tokens.dtype!=torch.float16 or mask.dtype!=torch.bool
        or mask.shape!=tokens.shape[:2] or lengths.shape!=(TOTAL,)
        or lengths.dtype!=torch.int32
        or list(tokens.shape)!=audit["full_token_feature_shape"]
        or not torch.equal(lengths,mask.sum(dim=1).to(torch.int32))
        or not torch.equal(hidden,tokens[torch.arange(TOTAL),lengths.long()-1])
        or not torch.isfinite(tokens).all()):
        raise AssertionError("S3I physical token mask/last hidden ABI invalid")

    # PRE-TRAIN: safetensors loads the complete immutable physical label
    # archive, but ONLY the [:64] slice is retained/passed to the optimizer.
    # The four control label slices are never referenced or scored until
    # AFTER the bridge parameters and normalizer are physically committed.
    # File-level loading is not equivalent to information-theoretic sealing.
    gold_all=load_file(str(labels_path),device="cpu")
    if (set(gold_all)!={"adjacency","source","target","reachability"}
        or gold_all["adjacency"].shape!=(TOTAL,NODES,NODES)):
        raise AssertionError("S3I graph label physical schema mismatch")
    train_gold={k:v[:TRAIN].clone() for k,v in gold_all.items()}
    del gold_all

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model=TokenAttentionToGraph()
    frozen=physical_frozen_nra(checkpoint,pretrained["fp16_checkpoint_sha256"]).eval()
    snapshot={k:v.clone() for k,v in frozen.state_dict().items()}
    valid=tokens[:TRAIN][mask[:TRAIN]].float()
    mean=valid.mean(dim=0)
    scale=valid.std(dim=0,unbiased=False).clamp(min=.05)
    norm_train=normalize(tokens[:TRAIN],mean,scale,mask[:TRAIN])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.02)

    with torch.no_grad():
        before=float(train_loss(model,frozen,norm_train,mask[:TRAIN],train_gold))
    trace=[]
    for i in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss=train_loss(model,frozen,norm_train,mask[:TRAIN],train_gold)
        if not torch.isfinite(loss):
            raise AssertionError("S3I nonfinite TRAIN-only loss")
        loss.backward()
        if any(p.grad is not None for p in frozen.parameters()):
            raise AssertionError("S3I learned pretrained frozen NRA weights")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if (i+1) in (1,40,80,160,240):
            trace.append({"step":i+1,"train_loss":float(loss.detach())})
    with torch.no_grad():
        after=float(train_loss(model,frozen,norm_train,mask[:TRAIN],train_gold))
    if (after>=before or any(not torch.equal(v,snapshot[k])
                              for k,v in frozen.state_dict().items())):
        raise AssertionError("S3I training did not improve TRAIN or altered NRA")

    # Only now physically commit model+normalizer BEFORE any CTRL gold load.
    out.mkdir(parents=True,exist_ok=True)
    weights=out/"NRA2_S3I_FACTORIZED_TRAIN64_TOKEN_ATTENTION_FP16.safetensors"
    packed={"bridge."+k:v.detach().cpu().half().contiguous()
            for k,v in model.state_dict().items()}
    packed["train_mean320"]=mean.half().contiguous()
    packed["train_scale320"]=scale.half().contiguous()
    save_file(packed,str(weights))
    physical=load_file(str(weights),device="cpu")
    if set(physical)!=set(packed) or any(not torch.equal(physical[k],v)
                                           for k,v in packed.items()):
        raise AssertionError("S3I FP16 physical checkpoint reload invalid")
    committed_sha=sha_file(weights)
    model=TokenAttentionToGraph().eval()
    model.load_state_dict({k.removeprefix("bridge."):v
                           for k,v in physical.items() if k.startswith("bridge.")})
    model.requires_grad_(False)
    saved_mean=physical["train_mean320"].float()
    saved_scale=physical["train_scale320"].float()

    # First USE of control gold for scoring after physical commit.
    # No optimization or model selection afterward; one evaluation per split.
    final_gold=load_file(str(labels_path),device="cpu")
    metrics={}
    oracle={}
    for name,count in GROUP_SIZES.items():
        start=groups[name]["start"]
        part=normalize(tokens[start:start+count],saved_mean,saved_scale,
                       mask[start:start+count])
        metrics[name]=split_metrics(model,frozen,part,
                                    mask[start:start+count],final_gold,start)
        with torch.inference_mode():
            exact=frozen(final_gold["adjacency"][start:start+count].float(),
                         final_gold["source"][start:start+count].float(),
                         final_gold["target"][start:start+count].float())
            reference=final_gold["reachability"][start:start+count].float()
            oracle[name]=int(((exact>0).float()==reference).sum())
    with torch.inference_mode():
        baseline=torch.arange(7,dtype=torch.float32)
        if zero_gate_passthrough(baseline,torch.zeros(7)) is not baseline:
            raise AssertionError("S3I output gate not zero")
    if any(not torch.equal(v,snapshot[k])
           for k,v in frozen.state_dict().items()):
        raise AssertionError("S3I pretrained NRA changed during CTRL evaluation")
    if sha_file(weights)!=committed_sha:
        raise AssertionError("S3I weights changed after any control observations")
    report={
        "stage":"NRA2_S3I_FACTORIZED_64TRAIN_FOUR_CTRL16_EXPLORATORY_ONLY",
        "source_S3H_native_CPU_provenance":True,
        "source_616_token_digest_unchanged":True,
        "train64":TRAIN,"control_groups":dict(GROUP_SIZES),
        "optimizer_steps":STEPS,"seed":SEED,
        "train_loss_before":before,"train_loss_after":after,
        "train_loss_trace":trace,
        "train_normalization_only":True,
        "controls_never_in_optimizer":True,
        "control_gold_not_used_by_optimizer":True,
        "complete_label_file_loaded_precommit_for_train_slice":True,
        "control_gold_first_scored_after_FP16_commit":True,
        "all_controls_scored_once":True,
        "independent_frozen_NRA_oracle_by_group":oracle,
        "per_group_metrics":metrics,
        "control_metrics_exploratory_not_independent_final_blind":True,
        "candidate_accepted":False,
        "candidate_real_full_EOS_gain_tested":False,
        "candidate_all_in_bpp_tested":False,
        "same_CPU_speed_tested":False,
        "calibri_green_tunnel_unloaded_and_unchanged":True,
        "pretrained_NRA_weights_unchanged":True,
        "original_584_and_616_protected_by_S3H":True,
        "output_gate":0.0,
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
        "physical_FP16_reload_bitwise_equal":True,
        "committed_adapter_FP16_sha256":committed_sha,
        "committed_adapter_FP16_bytes":weights.stat().st_size,
        "S3H_input_token_sha256":sha_file(seq_path),
        "NRA_standalone_fp16_sha256":sha_file(checkpoint),
    }
    (out/"NRA2_S3I_64TRAIN_FOUR_CTRL16_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="train_loss_trace"},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
