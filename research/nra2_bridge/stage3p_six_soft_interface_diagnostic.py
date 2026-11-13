"""S3P: diagnose six S3O frozen NRA mistakes without fitting or DEV tuning.

Same authentic 43M CPU token sequence, same physically committed S3O FP16
learned bridge (no gradients), same independent frozen NRA. Inspect a
predeclared 2x2x2 soft-vs-hard edge/source/target factorial and four
label-controlled oracle checks to isolate probability-interface effects.
All DEV64 were already exposed by S3O, not independent final blind.
No test in ordinary Jessica; zero output gate, no LIVE changes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from research.nra2_bridge.stage3m_role_balanced_graph_data import build_rows, validate
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3k_neural_entity_bridge import bind_mentions_to_context
from research.nra2_bridge.stage3o_neural_query_roles import (
    S3KGraphWithLearnedRoles, build_query_text_mask,
)
from research.nra2_bridge.stage3k_train_real_entity_binding import (
    exact_original_tokenizer, sha_file,
)
from research.nra2_bridge.stage3f_token_attention import normalize
from research.nra2_bridge.stage3b_relational_adapter import (
    physical_frozen_nra, zero_gate_passthrough,
)

S3O_SHA="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3N_SHA="b8a375982e556d67dfd5f741050c18c5460249184da9c3fcebc3d174d80db81b"
S3N_RUN=43522555326
S3O_RUN=43522663612
DEV=64
TOTAL=192
TRAIN=128
WIDTH=320
NODES=6


def assess(oracle_nra,edges,src,dst,answer):
    with torch.inference_mode():
        logits=oracle_nra(edges.float(),src.float(),dst.float())
    if logits.shape!=(DEV,) or not bool(torch.isfinite(logits).all()):
        raise AssertionError("S3P wrong frozen NRA binary-logit ABI")
    good=((logits>0)==answer.bool())
    return {
        "correct":int(good.sum()),
        "error_indices":torch.where(~good)[0].tolist(),
        "logits":[round(float(x),6) for x in logits],
        "correct_mask":[bool(x) for x in good],
    }


def main():
    if len(sys.argv)!=5:
        raise SystemExit("S3P S3N_CPU_DIR S3O_FP16_DIR FROZEN_NRA_DIR OUTPUT_DIR")
    s3n_dir,s3o_dir,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3P independent immutable diagnosis cannot overwrite")

    s3n=json.loads((s3n_dir/"NRA2_S3N_192_ROLEBALANCED_REAL_CPU_AUDIT.json").read_text())
    s3o=json.loads((s3o_dir/"NRA2_S3O_NEURAL_SEMANTIC_ROLES_128TRAIN_64DEV_AUDIT.json").read_text())
    standalone=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    fp=s3o_dir/"NRA2_S3O_NATIVE_CPU_LEARNED_ROLES_TRAIN128_FP16.safetensors"
    tokens_fp=s3n_dir/"NRA2_S3N_REAL_CPU_GRAPH192_ALL_TOKENS_FP16.safetensors"
    labels_fp=s3n_dir/"NRA2_S3N_GRAPH192_STRUCTURAL_LABELS.safetensors"
    nra_fp=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    if (
        s3n.get("stage")!="NRA2_S3N_REAL_CPU_FROZEN_192_ROLEBALANCED_TOKEN_CAPTURE_ONLY"
        or s3n.get("graph_prefill_count")!=TOTAL
        or s3n.get("train_count")!=TRAIN
        or s3n.get("nra_output_gate")!=0.
        or s3n.get("all_616_token_digest_matches_S194") is not True
        or s3n.get("original_456_full_EOS")!=456
        or s3n.get("original_new128_full_EOS")!=128
        or s3n.get("physical_reload_bitwise_equal") is not True
        or s3n.get("full_token_fp16_sha256")!=sha_file(tokens_fp)
        or s3n.get("physical_graph_labels_sha256")!=sha_file(labels_fp)
        or s3o.get("stage")!="NRA2_S3O_NATIVE_TOKEN_SEMANTIC_ROLES_TRAIN128_DEV64_EXPLORATORY"
        or s3o.get("S3N_successful_frozen_CPU_run")!=S3N_RUN
        or s3o.get("source_S3N_token_FP16_sha256")!=S3N_SHA
        or s3o.get("S3K_successful_train64_run")!=43519730567
        or s3o.get("original_616_S194_generated_token_hash_preserved_by_S3N") is not True
        or s3o.get("protected_584_control_answers_unchanged") is not True
        or s3o.get("train_count")!=TRAIN
        or s3o.get("dev_count")!=DEV
        or s3o.get("dev_metrics",{}).get("exact_graphs")!=DEV
        or s3o.get("dev_metrics",{}).get("source_correct")!=DEV
        or s3o.get("dev_metrics",{}).get("target_correct")!=DEV
        or s3o.get("dev_metrics",{}).get("NRA_predicted_graph_reachability_correct")!=58
        or s3o.get("frozen_NRA_DEV_oracle_correct")!=DEV
        or s3o.get("output_gate")!=0.
        or s3o.get("FP16_checkpoint_physical_reload_bitwise_equal") is not True
        or s3o.get("FP16_checkpoint_sha256")!=S3O_SHA
        or sha_file(fp)!=S3O_SHA
        or sha_file(tokens_fp)!=S3N_SHA
        or s3o.get("candidate_accepted") is not False
        or standalone.get("fp16_checkpoint_sha256")!=sha_file(nra_fp)
    ):
        raise AssertionError("S3P immutable CPU/S3O/NRA/J7 research provenance failure")
    rows=build_rows()
    if not validate(rows) or len(rows)!=TOTAL:
        raise AssertionError("S3P S3M public synthetic graph data drift")
    for split,start,size in (("TRAIN128",0,TRAIN),("DEV64",TRAIN,DEV)):
        index=s3n["split_groups"][split]
        if (index["start"]!=start or index["count"]!=size
            or index["prompt_sha256"]!=[r.prompt_sha256 for r in rows[start:start+size]]):
            raise AssertionError("S3P S3N native input prompt SHA mismatch")

    seq=load_file(str(tokens_fp),device="cpu")
    all_hidden=seq["token_hidden320"]
    all_mask=seq["token_mask"]
    lengths=seq["token_lengths"]
    if (all_hidden.shape[0]!=TOTAL or all_hidden.shape[2]!=WIDTH
        or all_mask.shape!=all_hidden.shape[:2]
        or lengths.shape!=(TOTAL,)
        or not torch.equal(all_mask.sum(1).int(),lengths.int())):
        raise AssertionError("S3P all-real CPU token lengths/masks invalid")
    gold=load_file(str(labels_fp),device="cpu")
    assert set(gold)=={"adjacency","source","target","reachability"}
    gold={k:v[TRAIN:].float() for k,v in gold.items()}
    if (gold["adjacency"].shape!=(DEV,NODES,NODES)
        or gold["source"].shape!=(DEV,NODES)
        or gold["reachability"].shape!=(DEV,)):
        raise AssertionError("S3P gold graph ABI invalid")
    weights=load_file(str(fp),device="cpu")
    bridge=S3KGraphWithLearnedRoles().eval()
    bridge.load_state_dict(
        {k.removeprefix("bridge."):v
         for k,v in weights.items() if k.startswith("bridge.")},
        strict=True,
    )
    bridge.requires_grad_(False)
    nra=physical_frozen_nra(nra_fp,standalone["fp16_checkpoint_sha256"]).eval()
    nra.requires_grad_(False)
    old_nra_sha=sha_file(nra_fp)
    old_bridge_sha=sha_file(fp)
    tok=exact_original_tokenizer()
    dev_rows=rows[TRAIN:]
    aligned=validate_without_gold(tok,[r.question for r in dev_rows])
    if any(a["tokens"]!=int(lengths[TRAIN+i]) for i,a in enumerate(aligned)):
        raise AssertionError("S3P actual frozen CPU tokenizer IDs/lengths changed")
    binding=bind_mentions_to_context(aligned,int(all_hidden.shape[1]))
    query=build_query_text_mask(tok,[r.question for r in dev_rows],aligned,int(all_hidden.shape[1]))
    states=normalize(all_hidden[TRAIN:],weights["normalizer.mean"].float(),
                     weights["normalizer.scale"].float(),all_mask[TRAIN:])
    with torch.inference_mode():
        edge_logits,source_logits,target_logits=bridge(
            states,all_mask[TRAIN:],binding,query
        )
        e_soft=torch.sigmoid(edge_logits)
        s_soft=F.softmax(source_logits,dim=-1)
        t_soft=F.softmax(target_logits,dim=-1)
        e_hard=(e_soft>0.5).float()
        s_hard=F.one_hot(source_logits.argmax(dim=-1),NODES).float()
        t_hard=F.one_hot(target_logits.argmax(dim=-1),NODES).float()
        if (not torch.equal(e_hard,gold["adjacency"])
            or not torch.equal(s_hard,gold["source"])
            or not torch.equal(t_hard,gold["target"])):
            raise AssertionError("S3P cannot attribute 6 failures purely to soft interface")
        idx=torch.arange(DEV)
        gold_a=gold["adjacency"].bool()
        positive_conf=e_soft[gold_a]
        negative_conf=e_soft[~gold_a]
        margin=torch.min(e_soft.masked_fill(~gold_a,1.),dim=1).values
        margin=torch.min(margin,dim=1).values
        best_false=e_soft.masked_fill(gold_a,0.).flatten(1).max(dim=1).values
        gold_src=gold["source"].argmax(1)
        gold_dst=gold["target"].argmax(1)

    outcome={}
    # These 8 feed frozen NRA with ONLY predicted continuous/hard
    # probabilities; none ever uses labels.
    for e_name,e in (("soft",e_soft),("hard",e_hard)):
        for s_name,s in (("soft",s_soft),("hard",s_hard)):
            for t_name,t in (("soft",t_soft),("hard",t_hard)):
                name=f"edge_{e_name}_src_{s_name}_dst_{t_name}"
                outcome[name]=assess(nra,e,s,t,gold["reachability"])

    # Label interventions strictly for DIAGNOSIS, never inference/selection.
    outcome["oracle_edges_oracle_roles"]=assess(
        nra,gold["adjacency"],gold["source"],gold["target"],gold["reachability"]
    )
    outcome["oracle_edges_predicted_soft_roles"]=assess(
        nra,gold["adjacency"],s_soft,t_soft,gold["reachability"]
    )
    outcome["predicted_soft_edges_oracle_roles"]=assess(
        nra,e_soft,gold["source"],gold["target"],gold["reachability"]
    )

    raw=outcome["edge_soft_src_soft_dst_soft"]
    hard=outcome["edge_hard_src_hard_dst_hard"]
    if (raw["correct"]!=58 or len(raw["error_indices"])!=6
        or hard["correct"]!=64
        or outcome["oracle_edges_oracle_roles"]["correct"]!=64
        or outcome["oracle_edges_predicted_soft_roles"]["correct"]!=
            outcome["edge_hard_src_soft_dst_soft"]["correct"]
        or outcome["predicted_soft_edges_oracle_roles"]["correct"]!=
            outcome["edge_soft_src_hard_dst_hard"]["correct"]
    ):
        raise AssertionError("S3P original 58/64 baseline, oracle 64/64 or interventions drifted")

    cases=[]
    for i in raw["error_indices"]:
        r=dev_rows[i]
        case={
            "dev_index":i,
            "global_index":i+TRAIN,
            "prompt_sha256":r.prompt_sha256,
            "family":r.family,"graph_id":r.graph_id,
            "active_nodes":r.active_nodes,
            "query_style":r.query_style,
            "gold_yes":bool(r.reachable),
            "gold_source":int(gold_src[i]),
            "gold_target":int(gold_dst[i]),
            "edge_true_min_probability":round(float(margin[i]),6),
            "edge_false_max_probability":round(float(best_false[i]),6),
            "source_true_probability":round(float(s_soft[i,gold_src[i]]),6),
            "target_true_probability":round(float(t_soft[i,gold_dst[i]]),6),
            "raw_soft_logit":raw["logits"][i],
            "frozen_nra_diagnostic_logits":{
                key:float(x["logits"][i]) for key,x in outcome.items()
            },
            "factorial_prediction_correct":{
                key:bool(x["correct_mask"][i]) for key,x in outcome.items()
            },
        }
        cases.append(case)

    # Invariant: computed feed tensors include no training labels at
    # inference. Oracle labels are used for diagnostics only.
    control=torch.arange(7,dtype=torch.float32)
    if zero_gate_passthrough(control,torch.zeros_like(control)) is not control:
        raise AssertionError("S3P research-only output gate must be exactly zero")
    if sha_file(nra_fp)!=old_nra_sha or sha_file(fp)!=old_bridge_sha:
        raise AssertionError("S3P mutated frozen S3O/NRA physical weights")
    report={
        "stage":"NRA2_S3P_FROZEN_S3O_SIX_DEV_SOFT_INTERFACE_POSTHOC_DIAGNOSTIC",
        "S3N_success_run":S3N_RUN,
        "S3O_success_run":S3O_RUN,
        "S3O_pinned_FP16_sha256":S3O_SHA,
        "S3N_pinned_real_cpu_tokens_FP16_sha256":S3N_SHA,
        "frozen_independent_NRA_sha256":old_nra_sha,
        "original_616_generated_CPU_tokens_protected":True,
        "original_584_full_EOS_protected":True,
        "graph64_exact":True,
        "source64_exact":True,
        "target64_exact":True,
        "posthoc_DEV64_frozen_inference_only":True,
        "DEV_is_untouched_final_blind":False,
        "train_or_finetune_any_weights":False,
        "threshold_or_hardening_selected_on_DEV":False,
        "all_8_soft_hard_factorial_control_counts":{
            key:val["correct"] for key,val in outcome.items()
            if key.startswith("edge_")
        },
        "all_variant_error_indices":{
            key:val["error_indices"] for key,val in outcome.items()
        },
        "six_failed_cases":cases,
        "raw_soft_correct":raw["correct"],
        "predicted_graph_and_roles_hard_control_correct":hard["correct"],
        "frozen_oracle_graph_and_roles_correct":
            outcome["oracle_edges_oracle_roles"]["correct"],
        "candidate_accepted":False,
        "independent_final_blind_tested":False,
        "real_new_decoder_full_EOS_tested":False,
        "all_in_physical_bpp_tested":False,
        "paired_native_CPU_speed_tested":False,
        "output_gate":0.,
        "LIVE_GreenTunnel_Calibri_NRA_all_unchanged":True,
        "no_QA_TEST_LIVE_Render_Supabase_APK":True,
    }
    out.mkdir(parents=True)
    (out/"NRA2_S3P_SIX_SOFT_INTERFACE_DIAGNOSTIC.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n"
    )
    print(json.dumps({
        "stage":report["stage"],
        "raw_soft_correct":raw["correct"],
        "hard_predicted_graph_and_roles_correct":hard["correct"],
        "oracle_graph_and_roles_correct":
            report["frozen_oracle_graph_and_roles_correct"],
        "all_8_soft_hard_factorial_control_counts":
            report["all_8_soft_hard_factorial_control_counts"],
        "six_failed_cases":cases,
        "candidate_accepted":False,
        "output_gate":0.,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
