"""NRA2 S3BQ: THIRD once-only NEW S3BO128 TRUE original CPU, S3BA vs JOINT S3BN.

Read-only research experiment. No DEV scoring, training, tuning, model selection,
threshold search, repeated candidate comparisons or production mutations.
THIRD untouched S3BO protocol SHA locked before S3BP genuine CPU, NEW S3BN joint candidate frozen on TRAIN, previous two EVAL sets permanently consumed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from research.nra2_bridge.stage3bj_future_independent128_prereg import (
    build_rows,validate,commitment,SPLIT as EVAL,DESTINATION_FIRST as REVERSED_STYLES,FAMILIES,
)
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3k_neural_entity_bridge import bind_mentions_to_context
from research.nra2_bridge.stage3o_neural_query_roles import (
    S3KGraphWithLearnedRoles,build_query_text_mask,
)
from research.nra2_bridge.stage3bm_joint_semantic_pair_role_head import S3BMGraphWithJointRoles
from research.nra2_bridge.stage3k_train_real_entity_binding import exact_original_tokenizer,sha_file
from research.nra2_bridge.stage3f_token_attention import normalize
from research.nra2_bridge.stage3b_relational_adapter import physical_frozen_nra,zero_gate_passthrough

N=128
WIDTH=320
NODES=6
THRESHOLD=0.5
S3BO_SHA="01858b405fbb034055574fd27feb93d19a34833da342a664ca5daed2b3ad091b"
S3BO_PREREG_RUN=43646187689
S3BA_SHA="4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BN_SHA="ea8a81665370cba22c4e8bc75fbe82314089244d4b32278ee62c2c30d88ae811"
S3BP_RUN=43647105183

def require(ok,why):
    if not bool(ok):
        raise AssertionError("S3BQ fail-closed: "+why)

def exact_two_sided_mcnemar(left_only:int,right_only:int)->float:
    n=left_only+right_only
    if n==0:
        return 1.0
    tail=sum(math.comb(n,k) for k in range(min(left_only,right_only)+1))
    return min(1.0,2.0*tail/(2**n))

def summarize(mask,rows,gold,graph_exact,preds):
    ids=torch.where(mask)[0]
    require(ids.numel()>0,"empty predeclared subgroup")
    actual=gold["reachability"][ids].bool()
    result={"n":int(ids.numel()),"yes":int(actual.sum()),
            "exact_graph":int(graph_exact[ids].sum())}
    for name,p in preds.items():
        src=p["src"][ids];dst=p["dst"][ids];reach=p["reach"][ids]
        result[name]={
            "source_correct":int(src.sum()),
            "target_correct":int(dst.sum()),
            "both_roles_correct":int((src&dst).sum()),
            "reachability_correct":int(reach.sum()),
            "reachability_false_positive":int((p["reach_pred"][ids]&~actual).sum()),
            "reachability_false_negative":int((~p["reach_pred"][ids]&actual).sum()),
        }
    for metric in ("both_roles","reach"):
        a=preds["S3BA"][metric][ids]
        b=preds["S3BN"][metric][ids]
        a_only=int((a&~b).sum())
        b_only=int((b&~a).sum())
        result["paired_"+metric]={
            "S3BA_only_correct":a_only,"S3BN_only_correct":b_only,
            "both_correct":int((a&b).sum()),
            "both_wrong":int((~a&~b).sum()),
            "net_S3BN_minus_S3BA":b_only-a_only,
            "exact_two_sided_mcnemar_p":exact_two_sided_mcnemar(a_only,b_only),
        }
    return result

def main(cpu:Path,pre:Path,adir:Path,bdir:Path,nra_dir:Path,out:Path):
    require(not out.exists() or not any(out.iterdir()),
            "one-time independent evaluation cannot overwrite prior report")
    # Third independent holdout protocol was SEALED in S3BO run
    # #43646187689, and new S3BN physical head SHA frozen in TRAIN-only
    # run #43645412996, before original CPU graph prompt capture S3BP.
    prior=json.loads((pre/"NRA2_S3BO_THIRD_FUTURE_EVAL128_PREREG.json").read_text())
    rows=build_rows()
    require(len(rows)==N and validate(rows)
            and len({r.topology_key for r in rows})==16
            and prior==json.loads(json.dumps(commitment(rows)))
            and prior.get("commitment_sha256")==S3BO_SHA
            and prior.get("stage")=="NRA2_S3BO_THIRD_FUTURE_INDEPENDENT_EVAL128_NO_MODEL_NO_CPU"
            and prior.get("split")==EVAL
            and prior.get("question_count")==128
            and prior.get("new_graph_topology_count")==16
            and prior.get("new_grammar_style_count")==8
            and prior.get("source_first_count")==64
            and prior.get("destination_first_count")==64
            and prior.get("yes_count")==64
            and prior.get("no_count")==64
            and prior.get("questions_in_order_sha256")==[r.prompt_sha256 for r in rows]
            and prior.get("no_spent_EVALs_reused_as_NEW_blind_or_fit") is True
            and prior.get("no_model_weights_original_CPU_inference_or_adapter_scoring") is True
            and prior.get("prior_S3AQ_consume_heldout")==43640363396
            and prior.get("prior_S3BJ_consume_heldout")==43644153544
            and prior.get("S3BN_frozen_candidate_SHA_required_before_CPU_capture_or_scores") is True
            and prior.get("candidate_accepted") is False and prior.get("output_gate")==0.,
            "THIRD original S3BO unseen128 prereg SHA/questions/roles invalid")
    a_file=adir/"NRA2_S3BA_S3AW_INIT_ROLE1536_FP16.safetensors"
    b_file=bdir/"NRA2_S3BN_JOINT_ROLE_SAFE_TRAIN2048_FP16.safetensors"
    require(sha_file(a_file)==S3BA_SHA and sha_file(b_file)==S3BN_SHA,
            "original S3BA or joint S3BN physical SHA drift")
    cand=json.loads((bdir/"NRA2_S3BN_JOINT_TRAIN2048_RESEARCH_AUDIT.json").read_text())
    require(cand.get("stage")=="NRA2_S3BN_JOINT_PAIR_SEMANTIC_TRAIN2048_NO_EVAL"
            and cand.get("S3BM_only_new_pair_trainable_params")==42869
            and cand.get("S3BA_parent_FP16_SHA256")==S3BA_SHA
            and cand.get("safe_new_pairwise_fp16_sha256")==S3BN_SHA
            and cand.get("safe_residual_scale_alpha")==1.
            and cand.get("train_count")==2048
            and cand.get("original_S194_616_J7_verified_across_six_CPU_TRAIN_sources") is True
            and cand.get("S3AQ_old_independent_EVAL128_ALREADY_CONSUMED_not_loaded_for_fit") is True
            and cand.get("S3BJ_new_independent_EVAL128_ALREADY_CONSUMED_not_loaded_for_fit") is True
            and cand.get("no_independent_EVAL_evaluated_or_retuned") is True
            and cand.get("original_graph_frozen_physical_equal") is True
            and cand.get("original_S3BA_head_frozen_physical_equal") is True
            and cand.get("candidate_accepted") is False and cand.get("output_gate")==0.,
            "S3BN model not locked by original TRAIN-only physical SHA and J7")
    audit=json.loads((cpu/"NRA2_S3BP_EVAL128_REAL_CPU_J7_AUDIT.json").read_text())
    token_file=cpu/"NRA2_S3BP_REAL_CPU_EVAL128_ALL_TOKENS_FP16.safetensors"
    feature_file=cpu/"NRA2_S3BP_REAL_43M_EVAL128_HIDDEN320_FP16.safetensors"
    labels_file=cpu/"NRA2_S3BP_EVAL128_STRUCTURAL_LABELS.safetensors"
    require(audit.get("stage")=="NRA2_S3BP_FROZEN_S3BO_THIRD_INDEPENDENT128_CPU_NO_SCORE"
            and audit.get("S3BO_third_blind_prereg_sha256")==S3BO_SHA
            and audit.get("S3BO_prereg_success_run")==S3BO_PREREG_RUN
            and audit.get("S3BN_new_joint_TRAIN_only_success_run")==43645412996
            and audit.get("S3BN_new_joint_physically_locked_safe_fp16_sha256")==S3BN_SHA
            and audit.get("S3BA_original_baseline_fp16_sha256")==S3BA_SHA
            and audit.get("S3BM_pair_attention_trainable_params")==42869
            and audit.get("original_S3AQ_S3BJ_EVALs_ALREADY_CONSUMED") is True
            and audit.get("third_S3BO_CPU_only_after_model_and_protocol_SHA_locks") is True
            and audit.get("third_S3BO_no_model_or_role_scored_in_this_run") is True
            and audit.get("all_616_token_digest_matches_S194") is True
            and audit.get("original_456_full_EOS")==456
            and audit.get("original_new128_full_EOS")==128
            and audit.get("train_count")==0 and audit.get("dev_count")==0
            and audit.get("eval_count")==N and audit.get("graph_prefill_count")==N
            and audit.get("group_sizes")=={EVAL:N}
            and audit.get("split_groups",{}).get(EVAL,{}).get("prompt_sha256")==[r.prompt_sha256 for r in rows]
            and audit.get("native_tokenizer_S3O_query_alignment_verified") is True
            and audit.get("physical_reload_bitwise_equal") is True
            and audit.get("frozen_soft_and_hard_candidate_scored") is False
            and audit.get("candidate_accepted") is False and audit.get("nra_output_gate")==0.,
            "S3BP genuine original43M CPU third independent128 physical SHA/gold invalid")
    require(sha_file(token_file)==audit["full_token_fp16_sha256"]
            and sha_file(feature_file)==audit["physical_feature_fp16_sha256"]
            and sha_file(labels_file)==audit["physical_graph_labels_sha256"],
            "third EVAL128 physical token/feature/label FP16 SHA mismatch")
    packed=load_file(str(token_file),device="cpu")
    hidden=packed["token_hidden320"]
    mask=packed["token_mask"]
    lens=packed["token_lengths"]
    final=load_file(str(feature_file),device="cpu")["hidden320"]
    require(hidden.ndim==3 and hidden.shape[0]==N and hidden.shape[2]==WIDTH
            and hidden.dtype==torch.float16 and mask.shape==hidden.shape[:2]
            and mask.dtype==torch.bool and lens.shape==(N,)
            and torch.equal(mask.sum(1).int(),lens.int())
            and torch.equal(hidden[torch.arange(N),lens.long()-1],final),
            "S3BP true original 43M token320 data/mask ABI drift")
    gold_raw=load_file(str(labels_file),device="cpu")
    require(set(gold_raw)=={"adjacency","source","target","reachability"}
            and gold_raw["source"].shape==(N,NODES)
            and gold_raw["target"].shape==(N,NODES)
            and gold_raw["adjacency"].shape==(N,NODES,NODES)
            and gold_raw["reachability"].shape==(N,),
            "immutable independent graph gold labels ABI")
    # Gold is checked for integrity but is NEVER supplied to either model.
    require(all(int(gold_raw["source"][i].argmax())==r.src
                and int(gold_raw["target"][i].argmax())==r.dst
                and int(gold_raw["reachability"][i])==int(r.reachable)
                for i,r in enumerate(rows)),
            "physical labels must match precommitted EVAL topology")
    tok=exact_original_tokenizer()
    aligned=validate_without_gold(tok,[r.question for r in rows])
    require(all(a["tokens"]==int(lens[i]) for i,a in enumerate(aligned)),
            "native S160 tokenizer length mismatch")
    binding=bind_mentions_to_context(aligned,hidden.shape[1])
    q=build_query_text_mask(tok,[r.question for r in rows],aligned,hidden.shape[1])
    require(q.node_order.shape==(N,2)
            and bool((q.node_order[:,0]!=q.node_order[:,1]).all()),
            "literal two entity candidates without gold")
    wa=load_file(str(a_file),device="cpu")
    wb=load_file(str(b_file),device="cpu")
    # Architectures intentionally have DIFFERENT parameter namespaces.
    # Only the original frozen graph, normalizer and S3BA query base must
    # remain BITWISE identical. New S3BN pair layers are the sole change.
    require(all(k in wb and torch.equal(wb[k],v) for k,v in wa.items()
                if k.startswith("bridge.graph.") or k.startswith("normalizer.")),
            "old graph/normalizer changed in new semantic architecture")
    require(all("bridge.query.base."+k.removeprefix("bridge.query.") in wb
                and torch.equal(wb["bridge.query.base."+k.removeprefix("bridge.query.")],v)
                for k,v in wa.items() if k.startswith("bridge.query.")),
            "original frozen S3BA query teacher changed in novel S3BN")
    require(int(wb["bridge.query.residual_scale"].item())==1,
            "S3BN fixed trained joint residual scale changed")
    x=normalize(hidden,wa["normalizer.mean"].float(),wa["normalizer.scale"].float(),mask)
    nr_file=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    nr_report=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    require(sha_file(nr_file)==nr_report["fp16_checkpoint_sha256"],"independent 2241p NRA SHA")
    nra=physical_frozen_nra(nr_file,nr_report["fp16_checkpoint_sha256"]).eval().requires_grad_(False)
    torch.set_num_threads(2)
    preds={}
    reference_edges=None
    with torch.inference_mode():
        for name,w,typ in (("S3BA",wa,S3KGraphWithLearnedRoles),
                           ("S3BN",wb,S3BMGraphWithJointRoles)):
            model=typ().eval()
            model.load_state_dict({k.removeprefix("bridge."):v for k,v in w.items()
                                   if k.startswith("bridge.")},strict=True)
            model.requires_grad_(False)
            edges,src,dst=model(x,mask,binding,q)
            require(all(bool(torch.isfinite(z).all()) for z in (edges,src,dst)),
                    "frozen inference nonfinite logits")
            if reference_edges is None:
                reference_edges=edges.clone()
            else:
                require(torch.equal(edges,reference_edges),"frozen graph logits changed")
            hard_edges=(torch.sigmoid(edges)>THRESHOLD).float()
            reach_pred=nra(hard_edges,F.softmax(src,dim=-1),F.softmax(dst,dim=-1))>0
            src_ok=src.argmax(1)==gold_raw["source"].argmax(1)
            dst_ok=dst.argmax(1)==gold_raw["target"].argmax(1)
            preds[name]={
                "src":src_ok,"dst":dst_ok,"both_roles":src_ok&dst_ok,
                "reach_pred":reach_pred,
                "reach":reach_pred==gold_raw["reachability"].bool(),
            }
        check=torch.arange(7,dtype=torch.float32)
        require(zero_gate_passthrough(check,torch.zeros(7)) is check,
                "shadow NRA must never alter source decoder output")
        graph_exact=(hard_edges==gold_raw["adjacency"].float()).flatten(1).all(1)
    yes=gold_raw["reachability"].bool()
    groups={
        "all":torch.ones(N,dtype=torch.bool),
        "source_first":torch.tensor([r.query_style not in REVERSED_STYLES for r in rows]),
        "destination_first":torch.tensor([r.query_style in REVERSED_STYLES for r in rows]),
        "yes":yes,"no":~yes,
        **{"style_"+str(i):torch.tensor([r.query_style==i for r in rows]) for i in range(8)},
        **{"family_"+family:torch.tensor([r.family==family for r in rows]) for family in FAMILIES},
    }
    metrics={key:summarize(m,rows,gold_raw,graph_exact,preds) for key,m in groups.items()}
    require(metrics["all"]["n"]==N
            and metrics["source_first"]["n"]==64
            and metrics["destination_first"]["n"]==64
            and metrics["all"]["yes"]==64
            and metrics["all"]["paired_both_roles"]["S3BN_only_correct"]+
                metrics["all"]["paired_both_roles"]["S3BA_only_correct"]<=N,
            "locked independent128 split and paired comparison")
    require(sha_file(token_file)==audit["full_token_fp16_sha256"]
            and sha_file(a_file)==S3BA_SHA and sha_file(b_file)==S3BN_SHA,
            "FP16 provenance changed after one-time inference")
    result={
        "stage":"NRA2_S3BQ_ONCE_ONLY_THIRD_S3BO128_S3BA_VS_JOINT_S3BN",
        "S3BP_run":S3BP_RUN,
        "S3BO_prereg_commitment_sha256":S3BO_SHA,
        "S3BO_prereg_successful_run":S3BO_PREREG_RUN,
        "S3BP_original_CPU_token_fp16_sha256":audit["full_token_fp16_sha256"],
        "S3BP_gold_fp16_sha256":audit["physical_graph_labels_sha256"],
        "S3BA_fp16_sha256":S3BA_SHA,"S3BN_fp16_sha256":S3BN_SHA,
        "original_616_S194_J7_preserved_in_S3BP":True,
        "prior_S3AQ_old_independent_EVAL128_consumed":True,
        "prior_S3BJ_second_independent_EVAL128_consumed":True,
        "S3BM_42869_new_pairwise_params":True,
        "third_unseen_16_topologies_and_8_new_grammars":True,
        "one_time_independent_EVAL128":N,
        "threshold_precommitted":THRESHOLD,
        "graph_weights_bitwise_equal":True,"graph_logits_bitwise_equal":True,
        "frozen_nra_sha256":nr_report["fp16_checkpoint_sha256"],
        "metrics":metrics,
        "question_level":[{
            "prompt_sha256":r.prompt_sha256,"graph_id":r.graph_id,
            "family":r.family,"query_style":r.query_style,
            "destination_first":bool(r.query_style in REVERSED_STYLES),
            "gold_yes":bool(r.reachable),
            **{name:{
                "source_correct":bool(v["src"][i]),
                "target_correct":bool(v["dst"][i]),
                "both_roles_correct":bool(v["both_roles"][i]),
                "reachability_correct":bool(v["reach"][i]),
            } for name,v in preds.items()},
        } for i,r in enumerate(rows)],
        "no_training_or_calibration_or_threshold_tuning":True,
        "no_repeated_DEV_scoring":True,
        "independent_EVAL_is_now_consumed":True,
        "no_winner_accepted_based_on_this_EVAL":True,
        "full_EOS_tested":False,"physical_all_in_bpp_tested":False,
        "paired_CPU_speed_tested":False,
        "candidate_accepted":False,"output_gate":0.,
        "no_LIVE_Render_Supabase_APK_main_QA_TEST_changes":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3BQ_ONCE_ONLY_THIRD128_S3BA_VS_S3BN_NO_RETUNE.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"stage":result["stage"],"primary":metrics["all"],
                     "source_first":metrics["source_first"],
                     "destination_first":metrics["destination_first"],
                     "output_gate":0.,"candidate_accepted":False},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    if len(sys.argv)!=7:
        raise SystemExit("Usage: S3BQ S3BP_CPU_DIR S3BO_DIR S3BA_DIR S3BN_DIR NRA_DIR OUTPUT_DIR")
    main(*(Path(p) for p in sys.argv[1:]))
