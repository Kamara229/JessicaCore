"""NRA2 S3S: previously frozen S3Q new-graph soft-vs-hard CPU interface trial.

NO optimization of weights, edges, thresholds, source or target.
Both arms use genuine S3R frozen original 43M CPU [64,T,320],
the exact S3O physically committed FP16 bridge, and SAME frozen NRA
2241p. Both use identical source/destination SOFT probabilities.
The ONLY intervention is predetermined sigmoid(edge)>0.5 hardening.
This is open public research about graph-topology transfer, NOT
sealed final QA, original decoder full+EOS, all-in bpp, or deployment.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from research.nra2_bridge.stage3q_independent_soft_hard_prereg import (
    build_rows,commitment,GROUP,FAMILIES,S3O_FP16_SHA256,
    S3N_FP16_SHA256,FROZEN_THRESHOLD,
)
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3k_neural_entity_bridge import bind_mentions_to_context
from research.nra2_bridge.stage3o_neural_query_roles import (
    S3KGraphWithLearnedRoles,build_query_text_mask,
)
from research.nra2_bridge.stage3k_train_real_entity_binding import (
    exact_original_tokenizer,sha_file,
)
from research.nra2_bridge.stage3f_token_attention import normalize
from research.nra2_bridge.stage3b_relational_adapter import (
    physical_frozen_nra,zero_gate_passthrough,
)

S3Q_COMMIT_SHA="5e9ea9cd66d5dedf0aa2cf574dcf78c8e0a7522c892f775fd8e9e014c46a26cb"
S3R_SUCCESS_RUN=43525586699
S3O_SUCCESS_RUN=43522663612
S3Q_PREREG_SUCCESS_RUN=43525211488
N=64
WIDTH=320
NODES=6


def slice_metrics(mask, soft_logit, hard_logit, gold, edges_pred, src_pred, dst_pred):
    idx=torch.where(mask)[0]
    if not len(idx):
        raise AssertionError("S3S empty predeclared category")
    yes=gold["reachability"][idx].bool()
    soft=(soft_logit[idx]>0)
    hard=(hard_logit[idx]>0)
    both=soft & hard
    gold_a=gold["adjacency"][idx].bool()
    exact_edge=(edges_pred[idx]==gold_a).flatten(1).all(1)
    src_true=src_pred[idx]==gold["source"][idx].argmax(1)
    dst_true=dst_pred[idx]==gold["target"][idx].argmax(1)
    return {
        "count":len(idx),
        "yes":int(yes.sum()),"no":int((~yes).sum()),
        "soft_correct":int((soft==yes).sum()),
        "hard_correct":int((hard==yes).sum()),
        "soft_false_positive":int((soft & ~yes).sum()),
        "hard_false_positive":int((hard & ~yes).sum()),
        "soft_false_negative":int((~soft & yes).sum()),
        "hard_false_negative":int((~hard & yes).sum()),
        "soft_correct_hard_wrong":int(((soft==yes)&(hard!=yes)).sum()),
        "hard_correct_soft_wrong":int(((hard==yes)&(soft!=yes)).sum()),
        "exact_predicted_graph":int(exact_edge.sum()),
        "correct_source":int(src_true.sum()),"correct_destination":int(dst_true.sum()),
        "soft_and_hard_agree":int((soft==hard).sum()),
    }


def main():
    if len(sys.argv)!=6:
        raise SystemExit("S3S S3R_REAL_CPU64_DIR S3Q_COMMIT_DIR S3O_FP16_DIR NRA_DIR OUTPUT_DIR")
    cpu_dir,prereg_dir,bridge_dir,nra_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3S immutable no-retune result may not overwrite")
    audit=json.loads((cpu_dir/"NRA2_S3R_64_INDEPENDENT_REAL_CPU_AUDIT.json").read_text())
    prereg=json.loads((prereg_dir/"NRA2_S3Q_GRAPH64_NO_SCORE_PRIOR_COMMITMENT.json").read_text())
    stage3o=json.loads((bridge_dir/"NRA2_S3O_NEURAL_SEMANTIC_ROLES_128TRAIN_64DEV_AUDIT.json").read_text())
    standalone=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    tok_file=cpu_dir/"NRA2_S3R_REAL_CPU_GRAPH64_ALL_TOKENS_FP16.safetensors"
    end_file=cpu_dir/"NRA2_S3R_REAL_43M_GRAPH64_HIDDEN320_FP16.safetensors"
    gold_file=cpu_dir/"NRA2_S3R_GRAPH64_STRUCTURAL_LABELS.safetensors"
    model_file=bridge_dir/"NRA2_S3O_NATIVE_CPU_LEARNED_ROLES_TRAIN128_FP16.safetensors"
    nra_file=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    rows=build_rows()
    frozen=commitment(rows)
    if (
        prereg!=frozen
        or prereg.get("commitment_sha256")!=S3Q_COMMIT_SHA
        or prereg.get("S3Q_predictions_not_inspected") is not True
        or prereg.get("NRA_model_not_retrained") is not True
        or prereg.get("S3O_model_not_retrained") is not True
        or prereg.get("S3O_frozen_sha256")!=S3O_FP16_SHA256
        or prereg.get("predeclared_edge_rule")!="hard_edge = (sigmoid(edge_logit) > 0.5).float()"
        or prereg.get("predeclared_role_rule")!="source and target remain SOFT in both arms"
        or FROZEN_THRESHOLD!=0.5
        or audit.get("stage")!="NRA2_S3R_REAL_CPU_FROZEN_64_INDEPENDENT_TOKEN_CAPTURE_ONLY"
        or audit.get("S3Q_predeclared_graph_commitment_sha256")!=S3Q_COMMIT_SHA
        or audit.get("S3Q_successful_prereg_run")!=S3Q_PREREG_SUCCESS_RUN
        or audit.get("S3N_native_CPU_source_run")!=43522555326
        or audit.get("S3N_192_source_token_fp16_sha256")!=S3N_FP16_SHA256
        or audit.get("all_616_token_digest_matches_S194") is not True
        or audit.get("original_456_full_EOS")!=456
        or audit.get("original_new128_full_EOS")!=128
        or audit.get("nra_output_gate")!=0.
        or audit.get("candidate_accepted") is not False
        or audit.get("frozen_soft_and_hard_candidate_scored") is not False
        or audit.get("graph_prefill_count")!=N
        or audit.get("train_count")!=0
        or audit.get("group_sizes")!={GROUP:N}
        or audit.get("physical_reload_bitwise_equal") is not True
        or audit.get("full_token_fp16_sha256")!=sha_file(tok_file)
        or audit.get("physical_feature_fp16_sha256")!=sha_file(end_file)
        or audit.get("physical_graph_labels_sha256")!=sha_file(gold_file)
        or audit["split_groups"][GROUP]["prompt_sha256"]!=[r.prompt_sha256 for r in rows]
        or stage3o.get("FP16_checkpoint_sha256")!=S3O_FP16_SHA256
        or stage3o.get("output_gate")!=0.
        or stage3o.get("source_S3N_token_FP16_sha256")!=S3N_FP16_SHA256
        or stage3o.get("FP16_checkpoint_physical_reload_bitwise_equal") is not True
        or stage3o.get("candidate_accepted") is not False
        or sha_file(model_file)!=S3O_FP16_SHA256
        or standalone.get("fp16_checkpoint_sha256")!=sha_file(nra_file)
    ):
        raise AssertionError("S3S true S3R CPU and previously committed S3Q/S3O/NRA provenance drift")

    seq=load_file(str(tok_file),device="cpu")
    h=seq["token_hidden320"]
    mask=seq["token_mask"]
    lens=seq["token_lengths"]
    final=load_file(str(end_file),device="cpu")["hidden320"]
    if (h.ndim!=3 or h.shape[0]!=N or h.shape[2]!=WIDTH or h.dtype!=torch.float16
        or mask.shape!=h.shape[:2] or mask.dtype!=torch.bool
        or lens.shape!=(N,) or not torch.equal(mask.sum(1).int(),lens.int())
        or not torch.equal(h[torch.arange(N),lens.long()-1],final)):
        raise AssertionError("S3S genuine 64 native CPU token320 mask/length physical ABI invalid")
    labels=load_file(str(gold_file),device="cpu")
    if (set(labels)!={"adjacency","source","target","reachability"}
        or labels["adjacency"].shape!=(N,NODES,NODES)
        or labels["source"].shape!=(N,NODES)
        or labels["target"].shape!=(N,NODES)
        or labels["reachability"].shape!=(N,)):
        raise AssertionError("S3S independently committed S3Q gold ABI invalid")
    gold={k:v.float() for k,v in labels.items()}
    tokenizer=exact_original_tokenizer()
    aligned=validate_without_gold(tokenizer,[r.question for r in rows])
    if any(a["tokens"]!=int(lens[i]) for i,a in enumerate(aligned)):
        raise AssertionError("S3S genuine frozen original S160 CPU tokenizer id/length drift")
    binding=bind_mentions_to_context(aligned,int(h.shape[1]))
    qmask=build_query_text_mask(tokenizer,[r.question for r in rows],aligned,int(h.shape[1]))

    f=load_file(str(model_file),device="cpu")
    model=S3KGraphWithLearnedRoles().eval()
    model.load_state_dict({k.removeprefix("bridge."):v
                           for k,v in f.items() if k.startswith("bridge.")},strict=True)
    model.requires_grad_(False)
    nra=physical_frozen_nra(nra_file,standalone["fp16_checkpoint_sha256"]).eval()
    nra.requires_grad_(False)
    original_fp16_sha=sha_file(model_file)
    original_nra_sha=sha_file(nra_file)
    inp=normalize(h,f["normalizer.mean"].float(),f["normalizer.scale"].float(),mask)
    with torch.inference_mode():
        logits,src_logits,dst_logits=model(inp,mask,binding,qmask)
        e_soft=torch.sigmoid(logits)
        e_hard=(e_soft>FROZEN_THRESHOLD).float()
        source_soft=F.softmax(src_logits,dim=-1)
        target_soft=F.softmax(dst_logits,dim=-1)
        # Fixed S3Q two arms, no third intervention and no tuning:
        soft_logits=nra(e_soft,source_soft,target_soft)
        hard_logits=nra(e_hard,source_soft,target_soft)
        oracle_logits=nra(gold["adjacency"],gold["source"],gold["target"])
        src_prediction=src_logits.argmax(1)
        dst_prediction=dst_logits.argmax(1)
        truth=gold["reachability"].bool()
        soft_right=(soft_logits>0)==truth
        hard_right=(hard_logits>0)==truth
        if any(not torch.isfinite(x).all() for x in
               [logits,src_logits,dst_logits,soft_logits,hard_logits,oracle_logits]):
            raise AssertionError("S3S frozen neural graph or NRA nonfinite logits")
        check=torch.arange(8,dtype=torch.float32)
        if zero_gate_passthrough(check,torch.zeros(8)) is not check:
            raise AssertionError("S3S original Calibri decoder output gate must remain zero")

    categories={
        "all":torch.ones(N,dtype=torch.bool),
        **{"family_"+f:torch.tensor([r.family==f for r in rows])
           for f in FAMILIES},
        "true_yes":truth,"true_no":~truth,
        **{"family_"+f+"_yes":torch.tensor([r.family==f and r.reachable for r in rows])
           for f in FAMILIES},
        **{"family_"+f+"_no":torch.tensor([r.family==f and not r.reachable for r in rows])
           for f in FAMILIES},
    }
    split_metrics={name:slice_metrics(m,soft_logits,hard_logits,gold,e_hard,
                                     src_prediction,dst_prediction)
                   for name,m in categories.items()}
    if (split_metrics["all"]["count"]!=N
        or split_metrics["all"]["yes"]!=32
        or split_metrics["all"]["no"]!=32
        or sum(split_metrics["family_"+fam]["count"] for fam in FAMILIES)!=N):
        raise AssertionError("S3S frozen category sizes invalid")

    disagreements=[]
    for i in range(N):
        if (not bool(soft_right[i]) or not bool(hard_right[i])
            or bool((soft_logits[i]>0)!=(hard_logits[i]>0))):
            r=rows[i]
            disagreements.append({
                "index":i,"prompt_sha256":r.prompt_sha256,
                "graph_id":r.graph_id,"family":r.family,"active_nodes":r.active_nodes,
                "query_style":r.query_style,"gold_yes":bool(r.reachable),
                "soft_correct":bool(soft_right[i]),
                "hard_correct":bool(hard_right[i]),
                "soft_logit":round(float(soft_logits[i]),6),
                "hard_logit":round(float(hard_logits[i]),6),
                "oracle_logit":round(float(oracle_logits[i]),6),
                "predicted_exact_graph":bool((e_hard[i]==gold["adjacency"][i]).all()),
                "source_correct":bool(src_prediction[i]==gold["source"][i].argmax()),
                "target_correct":bool(dst_prediction[i]==gold["target"][i].argmax()),
            })

    if (sha_file(model_file)!=original_fp16_sha
        or sha_file(nra_file)!=original_nra_sha):
        raise AssertionError("S3S frozen original S3O/NRA physical SHA mutated")
    result={
        "stage":"NRA2_S3S_PRECOMMITTED_S3Q64_SOFT_VS_HARD_FROZEN_NATIVE_CPU_TRIAL",
        "S3R_successful_native_CPU_run":S3R_SUCCESS_RUN,
        "S3Q_successful_prereg_run":S3Q_PREREG_SUCCESS_RUN,
        "S3O_successful_train128_run":S3O_SUCCESS_RUN,
        "S3Q_prior_commitment_sha256":S3Q_COMMIT_SHA,
        "S3O_frozen_adapter_sha256":original_fp16_sha,
        "S3R_native_CPU_64_hidden_fp16_sha256":sha_file(tok_file),
        "frozen_NRA_sha256":original_nra_sha,
        "independent_16_topologies_64_text_prompts":True,
        "original_616_S194_J7_hashes_preserved_by_S3R":True,
        "original_584_retained_full_EOS_unchanged":True,
        "S3Q_rules_predeclared_before_new_model_scores":True,
        "S3Q_topologies_not_seen_during_S3O_TRAIN128":True,
        "S3Q_query_phrasings_same_as_S3M":True,
        "source_destination_probabilities_soft_both_arms":True,
        "hardening_rule":"sigmoid(edge_logit)>0.5",
        "all_and_family_yes_no_accuracy":split_metrics,
        "oracle_true_graph_and_roles_correct":int(((oracle_logits>0)==truth).sum()),
        "disagreement_or_error_cases":disagreements,
        "new_training_or_parameter_tuning":False,
        "threshold_selected_using_S3Q":False,
        "this_is_not_sealed_final_blind":True,
        "candidate_accepted":False,
        "new_native_decoder_full_EOS_gain_tested":False,
        "all_in_physical_bpp_tested":False,
        "matched_CPU_speed_tested":False,
        "output_gate":0.0,
        "LIVE_Jessica_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    out.mkdir(parents=True)
    (out/"NRA2_S3S_S3Q64_PRECOMMITTED_SOFT_HARD_CPU_TRIAL.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n"
    )
    print(json.dumps({
        "stage":result["stage"],
        "S3Q_predeclared_commitment":result["S3Q_prior_commitment_sha256"],
        "overall":split_metrics["all"],
        "by_family":{f:split_metrics["family_"+f] for f in FAMILIES},
        "oracle_correct":result["oracle_true_graph_and_roles_correct"],
        "disagreement_or_error_cases":disagreements,
        "new_training":False,"candidate_accepted":False,"output_gate":0,
    },sort_keys=True),flush=True)


if __name__=="__main__":
    main()
