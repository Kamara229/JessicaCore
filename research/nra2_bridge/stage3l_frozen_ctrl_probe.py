"""NRA2 S3L: ONE post-commit read-only pass on S3K's REUSED controls.

This is an exploratory negative/positive diagnostic, not a blind test:
the same S3G controls were previously scored by S3I. No optimizer,
training, hyperparameter search, new decoder tokens, or LIVE changes.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from research.nra2_bridge.stage3k_train_real_entity_binding import (
    sha_file, exact_original_tokenizer,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import build_rows,GROUP_SIZES
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3k_neural_entity_bridge import (
    NodeSpanBoundGraph,bind_mentions_to_context,graph_forward,NODES,WIDTH,
)
from research.nra2_bridge.stage3f_token_attention import normalize
from research.nra2_bridge.stage3b_relational_adapter import physical_frozen_nra


def metrics(model,nra,tokens,mask,binding,gold):
    count=tokens.shape[0]
    with torch.inference_mode():
        edge,src,dst,reach=graph_forward(model,nra,tokens,mask,binding)
        binary=torch.sigmoid(edge)>.5
        truth=gold["adjacency"].bool()
        diagonal=torch.eye(NODES,dtype=torch.bool)[None,:,:]
        binary=binary.masked_fill(diagonal,False)
        tp=int((binary&truth).sum())
        fp=int((binary&~truth).sum())
        fn=int((~binary&truth).sum())
        return {
            "count":count,
            "exact_graphs":int((binary==truth).flatten(1).all(dim=1).sum()),
            "edge_F1":float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
            "edge_TP":tp,"edge_FP":fp,"edge_FN":fn,
            "source_correct":int((src.argmax(1)==gold["source"].argmax(1)).sum()),
            "target_correct":int((dst.argmax(1)==gold["target"].argmax(1)).sum()),
            "frozen_NRA_reachability_correct":
                int(((reach>0).float()==gold["reachability"].float()).sum())
        }


def main():
    if len(sys.argv)!=5:
        raise SystemExit("usage: python -m research.nra2_bridge.stage3l_frozen_ctrl_probe S3H S3K NRA OUTPUT")
    src,trained,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3L diagnostic evidence cannot be overwritten")
    audit=json.loads((src/"NRA2_S3H_128_FACTORIAL_REAL_CPU_AUDIT.json").read_text())
    s3k=json.loads((trained/"NRA2_S3K_ENTITY_BOUND_REAL_CPU_TRAIN64_AUDIT.json").read_text())
    pretrained=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    fpt=src/"NRA2_S3H_REAL_CPU_GRAPH128_ALL_TOKENS_FP16.safetensors"
    graph_file=src/"NRA2_S3H_GRAPH128_STRUCTURAL_LABELS.safetensors"
    learned=trained/"NRA2_S3K_REAL_CPU_ENTITY_BOUND_TRAIN64_FP16.safetensors"
    frozen_file=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    if (
        audit.get("stage")!="NRA2_S3H_REAL_CPU_FROZEN_128_FACTORIAL_TOKEN_CAPTURE_ONLY"
        or audit.get("full_token_fp16_sha256")!=sha_file(fpt)
        or audit.get("physical_graph_labels_sha256")!=sha_file(graph_file)
        or audit.get("all_616_token_digest_matches_S194") is not True
        or audit.get("nra_output_gate")!=0.
        or audit.get("group_sizes")!=GROUP_SIZES
        or s3k.get("stage")!="NRA2_S3K_NATIVE_ENTITY_BOUND_TRAIN64_ONLY_EXPLORATORY"
        or s3k.get("physical_FP16_adapter_sha256")!=sha_file(learned)
        or s3k.get("S3H_input_token_sha256")!=sha_file(fpt)
        or s3k.get("NRA_fp16_SHA256")!=sha_file(frozen_file)
        or s3k.get("trained_only_first64") is not True
        or s3k.get("control64_metrics_scored") is not False
        or s3k.get("physical_FP16_reload_bitwise_equal") is not True
        or s3k.get("candidate_accepted") is not False
        or pretrained.get("fp16_checkpoint_sha256")!=sha_file(frozen_file)
    ):
        raise AssertionError("S3L S3H/S3K physically immutable provenance incorrect")
    weights=load_file(str(learned),device="cpu")
    model=NodeSpanBoundGraph().eval()
    if set(weights)!={"bridge."+k for k in model.state_dict()}|{"train_mean320","train_scale320"}:
        raise AssertionError("S3L physically pinned S3K checkpoint keys drift")
    model.load_state_dict({k.removeprefix("bridge."):v for k,v in weights.items()
                           if k.startswith("bridge.")})
    model.requires_grad_(False)
    nra=physical_frozen_nra(frozen_file,pretrained["fp16_checkpoint_sha256"]).eval()
    nra.requires_grad_(False)
    nra_before={k:v.clone() for k,v in nra.state_dict().items()}
    seq=load_file(str(fpt),device="cpu")
    tokens=seq["token_hidden320"]
    mask=seq["token_mask"]
    lengths=seq["token_lengths"]
    if (tokens.shape[0]!=128 or tokens.shape[2]!=WIDTH or mask.shape!=tokens.shape[:2]
        or lengths.shape!=(128,) or not torch.equal(mask.sum(1).to(torch.int32),lengths)
        or weights["train_mean320"].shape!=(WIDTH,) or weights["train_scale320"].shape!=(WIDTH,)):
        raise AssertionError("S3L real native token ABI or physical normalized width mismatch")
    rows=build_rows()
    tok=exact_original_tokenizer()
    groups={}
    oracle={}
    total_examined=0
    # Skip the original TRAIN64: S3L evaluates only the four ALREADY
    # exposed controls with the FROZEN precommitted S3K weights.
    for group,count in list(GROUP_SIZES.items())[1:]:
        section=audit["split_groups"][group]
        offset=section["start"]
        if (count!=16 or offset<64 or offset+count>128
            or section["prompt_sha256"]!=[r.prompt_sha256 for r in rows[offset:offset+count]]):
            raise AssertionError("S3L group cohort leakage or indexing mismatch")
        aligned=validate_without_gold(tok,[r.question for r in rows[offset:offset+count]])
        if any(a["tokens"]!=int(lengths[offset+i]) for i,a in enumerate(aligned)):
            raise AssertionError("S3L real tokenizer vs native 43M hidden positions drift")
        binding=bind_mentions_to_context(aligned,int(tokens.shape[1]))
        normalized=normalize(tokens[offset:offset+count],
                             weights["train_mean320"].float(),
                             weights["train_scale320"].float(),
                             mask[offset:offset+count])
        # Immutable original S3H labels for final one-pass score ONLY.
        all_gold=load_file(str(graph_file),device="cpu")
        gold={k:v[offset:offset+count] for k,v in all_gold.items()}
        groups[group]=metrics(model,nra,normalized,mask[offset:offset+count],binding,gold)
        with torch.inference_mode():
            correct=nra(gold["adjacency"].float(),gold["source"].float(),
                        gold["target"].float())
            oracle[group]=int(((correct>0).float()==gold["reachability"].float()).sum())
        total_examined+=count
    if (total_examined!=64 or len(groups)!=4
        or any(not torch.equal(v,nra_before[k]) for k,v in nra.state_dict().items())
        or sha_file(learned)!=s3k["physical_FP16_adapter_sha256"]):
        raise AssertionError("S3L frozen neural network, provenance or scoring drift")
    report={
        "stage":"NRA2_S3L_FROZEN_S3K_REUSED_FOUR_CTRL16_EXPLORATORY",
        "original_S3H_616_J7_preserved":True,
        "S3K_frozen_checkpoint_sha256":sha_file(learned),
        "S3H_cpu_tokens_sha256":sha_file(fpt),
        "control_groups":groups,
        "frozen_NRA_oracle_by_group":oracle,
        "unseen_topologies_within_original_S3G":True,
        "prior_S3I_examined_exact_same_control_labels":True,
        "is_untouched_independent_blind":False,
        "total_scored":total_examined,
        "S3K_model_frozen_after_physical_FP16_commit":True,
        "control_data_never_used_for_training_or_model_selection":True,
        "model_accepted":False,
        "new_full_EOS_gain_tested":False,
        "all_in_bpp_tested":False,
        "paired_cpu_speed_tested":False,
        "calibri_GT_NRA_weights_unchanged":True,
        "output_gate":0.0,
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"NRA2_S3L_REUSED_FOUR_CTRL16_READONLY_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
