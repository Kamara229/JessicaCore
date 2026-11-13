"""S3O: real frozen CPU neural source/destination roles + S3K graph bridge.

TRAIN128 new S3M counterbalanced real token prompts; physically commit
FP16 before ONE topology-disjoint DEV64 score. Reused old S3K edge
weights, original normalizer, and frozen independent 2241p NRA.
This is research-only SHADOW; no native decoder full+EOS acceptance,
no LIVE, no main, no QA/TEST, no GT or Calibri weight changes.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import save_file,load_file

from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    build_rows,validate,
)
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3k_neural_entity_bridge import bind_mentions_to_context
from research.nra2_bridge.stage3o_neural_query_roles import (
    build_query_text_mask,S3KGraphWithLearnedRoles,graph_forward,
)
from research.nra2_bridge.stage3k_train_real_entity_binding import (
    exact_original_tokenizer,sha_file,
)
from research.nra2_bridge.stage3f_token_attention import normalize
from research.nra2_bridge.stage3b_relational_adapter import (
    physical_frozen_nra,zero_gate_passthrough,
)

SEED=2026111413
STEPS=240
TRAIN=128
DEV=64
TOTAL=192
NODES=6
WIDTH=320
HEAD_LR=0.002
GRAPH_LR=0.0003


def training_loss(model,nra,tokens,mask,binding,query,gold):
    if (tokens.shape[0]!=TRAIN or mask.shape[0]!=TRAIN
        or query.query_mask.shape[0]!=TRAIN
        or binding.mention_mask.shape[0]!=TRAIN
        or gold["adjacency"].shape[0]!=TRAIN):
        raise AssertionError("S3O optimizer must use TRAIN128 only")
    edges,src,dst,reach=graph_forward(model,nra,tokens,mask,binding,query)
    truth=gold["adjacency"].float()
    offdiag=~torch.eye(NODES,dtype=torch.bool)
    return (1.2*F.binary_cross_entropy_with_logits(
            edges[:,offdiag],truth[:,offdiag],pos_weight=torch.tensor(4.))
            +0.7*F.cross_entropy(src,gold["source"].argmax(dim=1))
            +0.7*F.cross_entropy(dst,gold["target"].argmax(dim=1))
            +0.18*F.binary_cross_entropy_with_logits(
                reach,gold["reachability"].float()))


def metrics(model,nra,tokens,mask,binding,query,gold):
    with torch.inference_mode():
        edges,src,dst,reach=graph_forward(model,nra,tokens,mask,binding,query)
        predicted=torch.sigmoid(edges)>.5
        diag=torch.eye(NODES,dtype=torch.bool)[None,:,:]
        predicted=predicted.masked_fill(diag,False)
        truth=gold["adjacency"].bool()
        tp=int((predicted&truth).sum())
        fp=int((predicted&~truth).sum())
        fn=int((~predicted&truth).sum())
        answers=gold["reachability"].float()
        return {
            "count":tokens.shape[0],
            "exact_graphs":int((predicted==truth).flatten(1).all(dim=1).sum()),
            "edge_F1":float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
            "edge_TP":tp,"edge_FP":fp,"edge_FN":fn,
            "source_correct":int((src.argmax(1)==gold["source"].argmax(1)).sum()),
            "target_correct":int((dst.argmax(1)==gold["target"].argmax(1)).sum()),
            "NRA_predicted_graph_reachability_correct":
                int(((reach>0).float()==answers).sum()),
        }


def main():
    if len(sys.argv)!=5:
        raise SystemExit(
            "usage: python -m research.nra2_bridge.stage3o_train_neural_roles "
            "S3N_TRUE_CPU_DIR S3K_FP16_DIR NRA_FROZEN_DIR OUTPUT_DIR")
    src,old,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3O immutable evidence cannot overwrite")
    s3n=json.loads((src/"NRA2_S3N_192_ROLEBALANCED_REAL_CPU_AUDIT.json").read_text())
    s3k=json.loads((old/"NRA2_S3K_ENTITY_BOUND_REAL_CPU_TRAIN64_AUDIT.json").read_text())
    standalone=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    tf=src/"NRA2_S3N_REAL_CPU_GRAPH192_ALL_TOKENS_FP16.safetensors"
    hf=src/"NRA2_S3N_REAL_43M_GRAPH192_HIDDEN320_FP16.safetensors"
    lf=src/"NRA2_S3N_GRAPH192_STRUCTURAL_LABELS.safetensors"
    wf=old/"NRA2_S3K_REAL_CPU_ENTITY_BOUND_TRAIN64_FP16.safetensors"
    nf=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    if (
        s3n.get("stage")!="NRA2_S3N_REAL_CPU_FROZEN_192_ROLEBALANCED_TOKEN_CAPTURE_ONLY"
        or s3n.get("S3H_successful_CPU_run")!=43517070838
        or s3n.get("S3M_independent_paired_topologies")!=48
        or s3n.get("all_616_token_digest_matches_S194") is not True
        or s3n.get("original_456_full_EOS")!=456
        or s3n.get("original_new128_full_EOS")!=128
        or s3n.get("graph_prefill_count")!=TOTAL
        or s3n.get("train_count")!=TRAIN
        or s3n.get("group_sizes")!={"TRAIN128":TRAIN,"DEV64":DEV}
        or s3n.get("nra_output_gate")!=0.
        or s3n.get("physical_reload_bitwise_equal") is not True
        or s3n.get("adapter_trained") is not False
        or s3n.get("control_groups_scored") is not False
        or s3n.get("candidate_accepted") is not False
        or s3n.get("full_token_fp16_sha256")!=sha_file(tf)
        or s3n.get("physical_feature_fp16_sha256")!=sha_file(hf)
        or s3n.get("physical_graph_labels_sha256")!=sha_file(lf)
        or s3k.get("stage")!="NRA2_S3K_NATIVE_ENTITY_BOUND_TRAIN64_ONLY_EXPLORATORY"
        or s3k.get("physical_FP16_adapter_sha256")!=sha_file(wf)
        or s3k.get("trained_only_first64") is not True
        or s3k.get("candidate_accepted") is not False
        or s3k.get("NRA_fp16_SHA256")!=sha_file(nf)
        or standalone.get("fp16_checkpoint_sha256")!=sha_file(nf)
    ):
        raise AssertionError("S3O source S3N native CPU, original J7 or frozen S3K/NRA invalid")
    seq=load_file(str(tf),device="cpu")
    tokens=seq["token_hidden320"]
    mask=seq["token_mask"]
    lengths=seq["token_lengths"]
    final=load_file(str(hf),device="cpu")["hidden320"]
    if (tokens.ndim!=3 or tokens.shape[0]!=TOTAL or tokens.shape[2]!=WIDTH
        or tokens.dtype!=torch.float16
        or mask.shape!=tokens.shape[:2] or mask.dtype!=torch.bool
        or lengths.shape!=(TOTAL,) or lengths.dtype!=torch.int32
        or not torch.equal(mask.sum(1).to(torch.int32),lengths)
        or not torch.equal(tokens[torch.arange(TOTAL),lengths.long()-1],final)
        or not torch.isfinite(tokens).all()):
        raise AssertionError("S3O actual S3N native CPU full-token mask, tail and SHA mismatch")

    rows=build_rows()
    if len(rows)!=TOTAL or not validate(rows):
        raise AssertionError("S3O S3M counterbalanced 192 prompts drift")
    offset=0
    for group,size in (("TRAIN128",TRAIN),("DEV64",DEV)):
        index=s3n["split_groups"][group]
        if (index["start"]!=offset or index["count"]!=size
            or index["prompt_sha256"]!=[r.prompt_sha256 for r in rows[offset:offset+size]]
            or index["positive"]!=size//2):
            raise AssertionError("S3O S3N immutable prompt order SHA and true balance drift")
        offset+=size
    tokenizer=exact_original_tokenizer()
    # All 64 unseen DEV prompt text intentionally UNREAD until FP16
    # is physically trained, committed and its SHA fixed.
    train_text=[r.question for r in rows[:TRAIN]]
    train_aligned=validate_without_gold(tokenizer,train_text)
    if any(a["tokens"]!=int(lengths[i]) for i,a in enumerate(train_aligned)):
        raise AssertionError("S3O authentic S160 native TRAIN tokenizer prefill ids/length drift")
    btrain=bind_mentions_to_context(train_aligned,int(tokens.shape[1]))
    qtrain=build_query_text_mask(tokenizer,train_text,train_aligned,int(tokens.shape[1]))
    del train_text,train_aligned

    gold_file=load_file(str(lf),device="cpu")
    if (set(gold_file)!={"adjacency","source","target","reachability"}
        or gold_file["adjacency"].shape!=(TOTAL,NODES,NODES)):
        raise AssertionError("S3O physical gold schema drift")
    gold_train={k:v[:TRAIN].clone() for k,v in gold_file.items()}
    del gold_file

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model=S3KGraphWithLearnedRoles()
    source_fp16=load_file(str(wf),device="cpu")
    source_graph={k.removeprefix("bridge."):v
                  for k,v in source_fp16.items() if k.startswith("bridge.")}
    if (set(source_graph)!=set(model.graph.state_dict())
        or source_fp16.get("train_mean320") is None
        or source_fp16.get("train_scale320") is None):
        raise AssertionError("S3O pretrained S3K graph model/normalization incompatible")
    model.graph.load_state_dict(source_graph)
    frozen=physical_frozen_nra(nf,standalone["fp16_checkpoint_sha256"]).eval()
    frozen.requires_grad_(False)
    snapshot={k:v.clone() for k,v in frozen.state_dict().items()}
    mean=source_fp16["train_mean320"].float()
    scale=source_fp16["train_scale320"].float()
    ntrain=normalize(tokens[:TRAIN],mean,scale,mask[:TRAIN])

    optimizer=torch.optim.AdamW([
        {"params":model.graph.parameters(),"lr":GRAPH_LR},
        {"params":model.query.parameters(),"lr":HEAD_LR},
    ],weight_decay=.02)
    with torch.no_grad():
        before=float(training_loss(model,frozen,ntrain,mask[:TRAIN],btrain,qtrain,gold_train))
    trace=[]
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss=training_loss(model,frozen,ntrain,mask[:TRAIN],btrain,qtrain,gold_train)
        if not torch.isfinite(loss):
            raise AssertionError("S3O nonfinite TRAIN-only loss")
        loss.backward()
        if any(p.grad is not None for p in frozen.parameters()):
            raise AssertionError("S3O changed frozen independent pretrained NRA")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step+1 in (1,40,80,160,240):
            trace.append({"step":step+1,"train_loss":float(loss.detach())})
    with torch.no_grad():
        after=float(training_loss(model,frozen,ntrain,mask[:TRAIN],btrain,qtrain,gold_train))
    if after>=before or any(not torch.equal(snapshot[k],v)
                             for k,v in frozen.state_dict().items()):
        raise AssertionError("S3O TRAIN loss did not improve or frozen NRA mutated")

    # Full physical FP16 model commitment before ANY DEV input/score.
    out.mkdir(parents=True,exist_ok=True)
    weights_path=out/"NRA2_S3O_NATIVE_CPU_LEARNED_ROLES_TRAIN128_FP16.safetensors"
    packed={"bridge."+k:v.detach().cpu().half().contiguous()
            for k,v in model.state_dict().items()}
    packed["normalizer.mean"]=mean.half().contiguous()
    packed["normalizer.scale"]=scale.half().contiguous()
    save_file(packed,str(weights_path))
    physical=load_file(str(weights_path),device="cpu")
    if set(physical)!=set(packed) or any(not torch.equal(physical[k],v)
                                           for k,v in packed.items()):
        raise AssertionError("S3O physical FP16 checkpoint reload mismatch")
    committed_sha=sha_file(weights_path)
    reloaded=S3KGraphWithLearnedRoles().eval()
    reloaded.load_state_dict({k.removeprefix("bridge."):v
                              for k,v in physical.items() if k.startswith("bridge.")})
    reloaded.requires_grad_(False)
    norm_mean=physical["normalizer.mean"].float()
    norm_scale=physical["normalizer.scale"].float()
    train_metrics=metrics(reloaded,frozen,normalize(tokens[:TRAIN],
                          norm_mean,norm_scale,mask[:TRAIN]),mask[:TRAIN],
                          btrain,qtrain,gold_train)

    # Exactly one DEV64 exploratory score. DEV had no influence on
    # architecture, optimizer, checkpoint or its SHA.
    dev_text=[r.question for r in rows[TRAIN:]]
    dev_aligned=validate_without_gold(tokenizer,dev_text)
    if any(a["tokens"]!=int(lengths[TRAIN+i]) for i,a in enumerate(dev_aligned)):
        raise AssertionError("S3O DEV tokenizer native CPU id/length drift")
    bdev=bind_mentions_to_context(dev_aligned,int(tokens.shape[1]))
    qdev=build_query_text_mask(tokenizer,dev_text,dev_aligned,int(tokens.shape[1]))
    del dev_text,dev_aligned,tokenizer
    # File-level load is not a cryptographic embargo; CTRL gold is
    # first USED for scoring only after physical FP16 commit.
    all_gold=load_file(str(lf),device="cpu")
    dev_gold={k:v[TRAIN:].clone() for k,v in all_gold.items()}
    dev_metrics=metrics(reloaded,frozen,normalize(tokens[TRAIN:],
                        norm_mean,norm_scale,mask[TRAIN:]),mask[TRAIN:],
                        bdev,qdev,dev_gold)
    with torch.inference_mode():
        expected=frozen(dev_gold["adjacency"].float(),dev_gold["source"].float(),
                        dev_gold["target"].float())
        oracle=int(((expected>0).float()==
                    dev_gold["reachability"].float()).sum())
        x=torch.arange(8,dtype=torch.float32)
        if zero_gate_passthrough(x,torch.zeros(8)) is not x:
            raise AssertionError("S3O original logits output gate no longer zero")
    if (sha_file(weights_path)!=committed_sha
        or any(not torch.equal(snapshot[k],v) for k,v in frozen.state_dict().items())):
        raise AssertionError("S3O checkpoint selection after DEV or frozen NRA changed")
    report={
        "stage":"NRA2_S3O_NATIVE_TOKEN_SEMANTIC_ROLES_TRAIN128_DEV64_EXPLORATORY",
        "S3N_successful_frozen_CPU_run":43522555326,
        "S3K_successful_train64_run":43519730567,
        "real_native_43M_cpu_frozen_full_token_inputs":True,
        "original_616_S194_generated_token_hash_preserved_by_S3N":True,
        "protected_584_control_answers_unchanged":True,
        "train_count":TRAIN,"dev_count":DEV,
        "optimizer_steps":STEPS,"seed":SEED,
        "role_lr":HEAD_LR,"graph_lr":GRAPH_LR,
        "source_graph_S3K_FP16_sha256":sha_file(wf),
        "source_S3N_token_FP16_sha256":sha_file(tf),
        "train_loss_before":before,"train_loss_after":after,
        "train_loss_trace":trace,
        "train_metrics":train_metrics,"dev_metrics":dev_metrics,
        "frozen_NRA_DEV_oracle_correct":oracle,
        "DEV_topologies_disjoint_from_TRAIN":True,
        "DEV_never_in_optimizer":True,
        "DEV_only_scored_after_physical_FP16_commit":True,
        "DEV_is_untouched_final_blind":False,
        "independent_final_blind_tested":False,
        "FP16_checkpoint_physical_reload_bitwise_equal":True,
        "FP16_checkpoint_sha256":committed_sha,
        "FP16_checkpoint_bytes":weights_path.stat().st_size,
        "candidate_accepted":False,
        "full_decoder_new_EOS_gains_tested":False,
        "all_in_physical_bpp_tested":False,
        "same_CPU_speed_tested":False,
        "Calibri_GreenTunnel_NRA_weights_unchanged":True,
        "output_gate":0.0,
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
    }
    (out/"NRA2_S3O_NEURAL_SEMANTIC_ROLES_128TRAIN_64DEV_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="train_loss_trace"},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
