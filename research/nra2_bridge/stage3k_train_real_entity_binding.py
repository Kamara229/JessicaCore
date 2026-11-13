"""NRA2 S3K: physically fit an explicit TEXT-to-token entity-bound bridge.

Authentic frozen Calibri 43M CPU [128,T,320] from S3H; no model weights
or decoder loaded. Use first 64 TRAIN only. Do NOT score the four
previously exposed control groups: they are NOT blind future evidence.
FP16 real physical reload, frozen 2241-parameter NRA, output gate=0.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file
from transformers import AutoTokenizer

from research.nra2_bridge.stage3g_factorized_graph_controls import build_rows,GROUP_SIZES
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3k_neural_entity_bridge import (
    NodeSpanBoundGraph,bind_mentions_to_context,graph_forward,NODES,WIDTH,
)
from research.nra2_bridge.stage3f_token_attention import normalize
from research.nra2_bridge.stage3b_relational_adapter import (
    physical_frozen_nra,zero_gate_passthrough,
)

SEED=2026111404
STEPS=240
TRAIN=64
TOTAL=128
MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_SHA="12fd25f77366fa6b3b4b768ec3050bf629380bac"


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_original_tokenizer():
    """Do not silently change the native S160 tokenizer SHA."""
    source=ast.parse(Path("research/calibri/binary_geometry_v64.py").read_text())
    values={}
    for stmt in source.body:
        if (isinstance(stmt,ast.Assign) and len(stmt.targets)==1
            and isinstance(stmt.targets[0],ast.Name)
            and stmt.targets[0].id in ("MODEL","MODEL_SHA")):
            values[stmt.targets[0].id]=ast.literal_eval(stmt.value)
    if values!={"MODEL":MODEL,"MODEL_SHA":MODEL_SHA}:
        raise AssertionError("S3K frozen original model/tokenizer revision drift")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA,use_fast=True)
    if not tok.is_fast:
        raise AssertionError("S3K node-token offset requires authentic FAST tokenizer")
    return tok


def training_loss(model,nra,tokens,mask,binding,train_gold):
    if (tokens.shape[0]!=TRAIN or mask.shape[0]!=TRAIN
        or binding.mention_mask.shape[0]!=TRAIN
        or train_gold["adjacency"].shape[0]!=TRAIN):
        raise AssertionError("S3K optimizer may receive ONLY 64 TRAIN examples")
    edge,src,dst,reach=graph_forward(model,nra,tokens,mask,binding)
    truth=train_gold["adjacency"].float()
    source=train_gold["source"].argmax(dim=1)
    target=train_gold["target"].argmax(dim=1)
    result=train_gold["reachability"].float()
    offdiag=~torch.eye(NODES,dtype=torch.bool)
    return (
        1.2*F.binary_cross_entropy_with_logits(
            edge[:,offdiag],truth[:,offdiag],pos_weight=torch.tensor(4.))
        +.7*F.cross_entropy(src,source)
        +.7*F.cross_entropy(dst,target)
        +.18*F.binary_cross_entropy_with_logits(reach,result)
    )


def score_train_only(model,nra,normalized,mask,binding,gold):
    with torch.inference_mode():
        edge,src,dst,reach=graph_forward(model,nra,normalized,mask,binding)
        binary=(torch.sigmoid(edge)>.5)
        truth=gold["adjacency"].bool()
        offdiag=torch.eye(NODES,dtype=torch.bool)
        binary=binary.masked_fill(offdiag[None,:,:],False)
        tp=int((binary&truth).sum())
        fp=int((binary&~truth).sum())
        fn=int((~binary&truth).sum())
        return {
            "count":TRAIN,
            "exact_graphs":int((binary==truth).flatten(1).all(dim=1).sum()),
            "edge_F1":float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
            "edge_TP":tp,"edge_FP":fp,"edge_FN":fn,
            "source_correct":int((src.argmax(1)==gold["source"].argmax(1)).sum()),
            "target_correct":int((dst.argmax(1)==gold["target"].argmax(1)).sum()),
            "frozen_NRA_reachability_correct":
                int(((reach>0).float()==gold["reachability"].float()).sum()),
        }


def main():
    if len(sys.argv)!=4:
        raise SystemExit("usage: python -m research.nra2_bridge.stage3k_train_real_entity_binding S3H_DIR NRA_DIR OUTPUT_DIR")
    source,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3K evidence cannot be overwritten")
    audit_path=source/"NRA2_S3H_128_FACTORIAL_REAL_CPU_AUDIT.json"
    token_file=source/"NRA2_S3H_REAL_CPU_GRAPH128_ALL_TOKENS_FP16.safetensors"
    hidden_file=source/"NRA2_S3H_REAL_43M_GRAPH128_HIDDEN320_FP16.safetensors"
    label_file=source/"NRA2_S3H_GRAPH128_STRUCTURAL_LABELS.safetensors"
    frozen_file=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    audit=json.loads(audit_path.read_text())
    pretrained=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    if (
        audit.get("stage")!="NRA2_S3H_REAL_CPU_FROZEN_128_FACTORIAL_TOKEN_CAPTURE_ONLY"
        or audit.get("S3E_successful_parent_run")!=43512049506
        or audit.get("full_token_fp16_sha256")!=sha_file(token_file)
        or audit.get("physical_feature_fp16_sha256")!=sha_file(hidden_file)
        or audit.get("physical_graph_labels_sha256")!=sha_file(label_file)
        or audit.get("frozen_independent_NRA_sha256")!=sha_file(frozen_file)
        or pretrained.get("fp16_checkpoint_sha256")!=sha_file(frozen_file)
        or audit.get("all_616_token_digest_matches_S194") is not True
        or audit.get("original_456_full_EOS")!=456
        or audit.get("original_new128_full_EOS")!=128
        or audit.get("train_count")!=TRAIN
        or audit.get("graph_prefill_count")!=TOTAL
        or audit.get("nra_output_gate")!=0.
        or audit.get("group_sizes")!=GROUP_SIZES
        or audit.get("control_groups_scored") is not False
        or audit.get("physical_reload_bitwise_equal") is not True
        or audit.get("candidate_accepted") is not False
    ):
        raise AssertionError("S3K source S3H real CPU or NRA physical J7 provenance invalid")
    seq=load_file(str(token_file),device="cpu")
    tokens=seq["token_hidden320"]
    mask=seq["token_mask"]
    lengths=seq["token_lengths"]
    final=load_file(str(hidden_file),device="cpu")["hidden320"]
    if (
        tokens.ndim!=3 or tokens.shape[0]!=TOTAL or tokens.shape[2]!=WIDTH
        or tokens.dtype!=torch.float16 or mask.shape!=tokens.shape[:2]
        or mask.dtype!=torch.bool or lengths.shape!=(TOTAL,)
        or lengths.dtype!=torch.int32
        or not torch.equal(mask.sum(1).to(torch.int32),lengths)
        or not torch.equal(tokens[torch.arange(TOTAL),lengths.long()-1],final)
        or not torch.isfinite(tokens).all()
    ):
        raise AssertionError("S3K authentic physical [128,T,320] token source drift")

    rows=build_rows()
    if len(rows)!=TOTAL:
        raise AssertionError("S3K unique preregistered 128 prompt cohort drift")
    hashes=[r.prompt_sha256 for r in rows]
    offset=0
    for group,count in GROUP_SIZES.items():
        cfg=audit["split_groups"][group]
        if (cfg["start"]!=offset or cfg["count"]!=count
            or cfg["prompt_sha256"]!=hashes[offset:offset+count]):
            raise AssertionError("S3K read-only exact S3G S3H cohort SHA drift")
        offset+=count
    tokenizer=exact_original_tokenizer()
    # Critically: tokenize only the 64 training text prompts. The 64
    # control texts, gold graphs and metrics never enter this optimizer.
    aligned=validate_without_gold(tokenizer,[r.question for r in rows[:TRAIN]])
    if any(a["tokens"]!=int(lengths[i]) for i,a in enumerate(aligned)):
        raise AssertionError("S3K real CPU hidden and tokenizer token lengths disagree")
    binding=bind_mentions_to_context(aligned,int(tokens.shape[1]))
    del tokenizer,aligned,rows

    # Whole physical safetensors label archive is readable as one file;
    # explicitly keep only TRAIN64 labels and do not score any CTRL16.
    labels=load_file(str(label_file),device="cpu")
    if (set(labels)!={"adjacency","source","target","reachability"}
        or labels["adjacency"].shape!=(TOTAL,NODES,NODES)):
        raise AssertionError("S3K graph target ABI invalid")
    train_gold={k:v[:TRAIN].clone() for k,v in labels.items()}
    del labels

    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model=NodeSpanBoundGraph()
    frozen=physical_frozen_nra(frozen_file,pretrained["fp16_checkpoint_sha256"]).eval()
    frozen_snapshot={k:v.detach().clone() for k,v in frozen.state_dict().items()}
    valid_tokens=tokens[:TRAIN][mask[:TRAIN]].float()
    mean=valid_tokens.mean(dim=0)
    scale=valid_tokens.std(dim=0,unbiased=False).clamp(min=.05)
    normalized=normalize(tokens[:TRAIN],mean,scale,mask[:TRAIN])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.02)
    with torch.no_grad():
        before=float(training_loss(model,frozen,normalized,mask[:TRAIN],binding,train_gold))
    trace=[]
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss=training_loss(model,frozen,normalized,mask[:TRAIN],binding,train_gold)
        if not torch.isfinite(loss):
            raise AssertionError("S3K nonfinite train-only loss")
        loss.backward()
        if any(p.grad is not None for p in frozen.parameters()):
            raise AssertionError("S3K changed frozen independent NRA")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step+1 in (1,40,80,160,240):
            trace.append({"step":step+1,"train_loss":float(loss.detach())})
    with torch.no_grad():
        after=float(training_loss(model,frozen,normalized,mask[:TRAIN],binding,train_gold))
    if (after>=before or any(not torch.equal(frozen_snapshot[k],v)
                              for k,v in frozen.state_dict().items())):
        raise AssertionError("S3K training failed or pretrained NRA changed")

    # Physically commit FP16 before even TRAIN scoring; NO OOD scoring.
    out.mkdir(parents=True,exist_ok=True)
    weight_file=out/"NRA2_S3K_REAL_CPU_ENTITY_BOUND_TRAIN64_FP16.safetensors"
    packed={"bridge."+k:v.detach().cpu().to(torch.float16).contiguous()
            for k,v in model.state_dict().items()}
    packed["train_mean320"]=mean.half().contiguous()
    packed["train_scale320"]=scale.half().contiguous()
    save_file(packed,str(weight_file))
    physical=load_file(str(weight_file),device="cpu")
    if set(physical)!=set(packed) or any(not torch.equal(physical[k],v)
                                           for k,v in packed.items()):
        raise AssertionError("S3K persisted bridge FP16 mismatch")
    model=NodeSpanBoundGraph().eval()
    model.load_state_dict({k.removeprefix("bridge."):v
                          for k,v in physical.items() if k.startswith("bridge.")})
    model.requires_grad_(False)
    norm=normalize(tokens[:TRAIN],physical["train_mean320"].float(),
                   physical["train_scale320"].float(),mask[:TRAIN])
    metrics=score_train_only(model,frozen,norm,mask[:TRAIN],binding,train_gold)
    with torch.inference_mode():
        oracle=frozen(train_gold["adjacency"].float(),train_gold["source"].float(),
                      train_gold["target"].float())
        oracle_correct=int(((oracle>0).float()==
                             train_gold["reachability"].float()).sum())
        control=torch.arange(7,dtype=torch.float32)
        if zero_gate_passthrough(control,torch.zeros(7)) is not control:
            raise AssertionError("S3K output gate nonzero")
    if (sha_file(weight_file)!=sha_file(weight_file)
        or any(not torch.equal(frozen_snapshot[k],v)
               for k,v in frozen.state_dict().items())):
        raise AssertionError("S3K frozen NRA or physical weights drift")
    report={
        "stage":"NRA2_S3K_NATIVE_ENTITY_BOUND_TRAIN64_ONLY_EXPLORATORY",
        "S3H_successful_native_CPU_run":43517070838,
        "S3J_successful_original_tokenizer_preflight_run":43518808282,
        "real_frozen_43M_layer12_input":True,
        "real_128_source_tensor_shape":list(tokens.shape),
        "source_616_original_cpu_J7_SHA_protected":True,
        "source_584_protected_original_answers":True,
        "trained_only_first64":True,"train_count":TRAIN,
        "optimizer_steps":STEPS,"seed":SEED,
        "native_tokenizer_sha_pinned":MODEL_SHA,
        "text_token_entity_alignment":True,
        "node_mention_masks_only_from_text":True,
        "edge_witness_candidates_not_oracle_graph":True,
        "query_last_mention_order_from_text_not_gold":True,
        "train_loss_before":before,"train_loss_after":after,
        "training_loss_trace":trace,
        "train_metrics_after_FP16_commit":metrics,
        "frozen_graph_NRA_train_oracle_correct":oracle_correct,
        "control64_labels_used_to_optimize":False,
        "control64_metrics_scored":False,
        "fresh_independent_heldout_tested":False,
        "candidate_real_decoder_new_full_EOS_tested":False,
        "candidate_all_in_physical_bpp_tested":False,
        "paired_native_CPU_speed_tested":False,
        "candidate_accepted":False,
        "calibri_green_tunnel_unchanged":True,
        "pretrained_NRA_frozen_unchanged":True,
        "output_gate":0.0,
        "physical_FP16_reload_bitwise_equal":True,
        "physical_FP16_adapter_bytes":weight_file.stat().st_size,
        "physical_FP16_adapter_sha256":sha_file(weight_file),
        "S3H_input_token_sha256":sha_file(token_file),
        "NRA_fp16_SHA256":sha_file(frozen_file),
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
    }
    (out/"NRA2_S3K_ENTITY_BOUND_REAL_CPU_TRAIN64_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="training_loss_trace"},
                     sort_keys=True),flush=True)


if __name__=="__main__":
    main()
