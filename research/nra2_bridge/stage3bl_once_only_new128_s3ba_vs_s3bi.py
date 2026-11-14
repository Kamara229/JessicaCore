"""NRA2 S3BL: one-time sealed S3BA vs S3BI on NEW preregistered S3BJ EVAL128.

Read-only research experiment. No DEV scoring, training, tuning, model selection,
threshold search, repeated candidate comparisons or production mutations.
New S3BJ protocol SHA locked before S3BK genuine CPU, S3BI candidate frozen on TRAIN before any EVAL.
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
from research.nra2_bridge.stage3k_train_real_entity_binding import exact_original_tokenizer,sha_file
from research.nra2_bridge.stage3f_token_attention import normalize
from research.nra2_bridge.stage3b_relational_adapter import physical_frozen_nra,zero_gate_passthrough

N=128
WIDTH=320
NODES=6
THRESHOLD=0.5
S3BJ_SHA="e48faef1a8b164cad9641f1b5946932f15c27baeeab47d829fb5ec754dd46265"
S3BJ_PREREG_RUN=43641688882
S3BA_SHA="4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BI_SHA="294067f48298cce216bdc49b1f0fd5f9b8c09c44f6f51227e9ec983602b2765d"
S3BK_RUN=43643375827

def require(ok,why):
    if not bool(ok):
        raise AssertionError("S3BL fail-closed: "+why)

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
        b=preds["S3BI"][metric][ids]
        a_only=int((a&~b).sum())
        b_only=int((b&~a).sum())
        result["paired_"+metric]={
            "S3BA_only_correct":a_only,"S3BI_only_correct":b_only,
            "both_correct":int((a&b).sum()),
            "both_wrong":int((~a&~b).sum()),
            "net_S3BI_minus_S3BA":b_only-a_only,
            "exact_two_sided_mcnemar_p":exact_two_sided_mcnemar(a_only,b_only),
        }
    return result

def main(cpu:Path,pre:Path,adir:Path,bdir:Path,nra_dir:Path,out:Path):
    require(not out.exists() or not any(out.iterdir()),
            "one-time independent evaluation cannot overwrite prior report")
    # The immutable S3BJ new independent 128 prereg must predate original CPU
    # acquisition and physically SHA-fixed safe S3BI train-only candidate.
    prereg_path=pre/"NRA2_S3BJ_FUTURE_INDEPENDENT_EVAL128_PREREG.json"
    protocol=json.loads(prereg_path.read_text())
    rows=build_rows()
    require(validate(rows) and len(rows)==N
            and len({r.topology_key for r in rows})==32
            and protocol==json.loads(json.dumps(commitment(rows)))
            and protocol.get("commitment_sha256")==S3BJ_SHA
            and protocol.get("stage")=="NRA2_S3BJ_FUTURE_INDEPENDENT128_PREREG_NO_CPU_NO_MODEL"
            and protocol.get("split")==EVAL
            and protocol.get("count")==128
            and protocol.get("graph_count")==32
            and protocol.get("source_first")==64 and protocol.get("destination_first")==64
            and protocol.get("yes")==64 and protocol.get("no")==64
            and protocol.get("no_model_weights_or_CPU_capture_or_adaptor_scoring") is True
            and protocol.get("old_EVAL128_must_never_be_used_for_training_or_new_blind") is True
            and protocol.get("S3AQ_old_EVAL128_ALREADY_CONSUMED_at_S3BH")==43640363396
            and protocol.get("candidate_S3BI_physical_FP16_must_be_locked_before_new_eval_CPU") is True
            and protocol.get("output_gate")==0. and protocol.get("future_candidate_accepted") is False,
            "S3BJ sealed NEW blind128 precommit changed after candidates locked")
    a_file=adir/"NRA2_S3BA_S3AW_INIT_ROLE1536_FP16.safetensors"
    b_file=bdir/"NRA2_S3BI_S3BA_INIT_SAFE_ROLE2048_FP16.safetensors"
    require(sha_file(a_file)==S3BA_SHA and sha_file(b_file)==S3BI_SHA,
            "physical model SHA drift from S3BA/S3BI locked comparison")
    cand=json.loads((bdir/"NRA2_S3BI_S3BA_INIT_TRAIN2048_NO_DEV_ROLE_AUDIT.json").read_text())
    require(cand.get("S3BA_parent_fp16_sha256")==S3BA_SHA
            and cand.get("new_S3BI_physical_fp16_sha256")==S3BI_SHA
            and cand.get("S3BI_R0_rejected_train_guard_run")==43641294513
            and cand.get("TRAIN_only_safe_candidate_alpha")==.75
            and cand.get("train_count")==2048
            and cand.get("original_616_S194_J7_digests_verified_from_six_true_CPU_sources") is True
            and cand.get("S3BJ_new_blind128_unopened_and_cpu_not_captured") is True
            and cand.get("no_heldout_model_inference_this_stage") is True
            and cand.get("candidate_accepted") is False and cand.get("output_gate")==0.
            and all(x["candidate_net"]>=0 for k,x in
                    cand["replay_guard_original_five_TRAIN_groups"].items()
                    if k!="S3BC_TRAIN512"),
            "candidate not strictly the accepted TRAIN-only SHA-locked SAFE head")
    audit=json.loads((cpu/"NRA2_S3BK_EVAL128_REAL_CPU_J7_AUDIT.json").read_text())
    token_file=cpu/"NRA2_S3BK_REAL_CPU_EVAL128_ALL_TOKENS_FP16.safetensors"
    feature_file=cpu/"NRA2_S3BK_REAL_43M_EVAL128_HIDDEN320_FP16.safetensors"
    labels_file=cpu/"NRA2_S3BK_EVAL128_STRUCTURAL_LABELS.safetensors"
    require(audit.get("stage")=="NRA2_S3BK_FROZEN_S3BJ_NEW_INDEPENDENT_EVAL128_NO_SCORE_CPU_CAPTURE_ONLY"
            and audit.get("S3BJ_new_blind_prereg_sha256")==S3BJ_SHA
            and audit.get("S3BJ_prereg_successful_run")==S3BJ_PREREG_RUN
            and audit.get("S3BI_train_only_candidate_successful_run")==43642628668
            and audit.get("S3BI_physically_locked_candidate_fp16_sha256")==S3BI_SHA
            and audit.get("S3BA_original_baseline_fp16_sha256")==S3BA_SHA
            and audit.get("original_S3AQ_EVAL128_ALREADY_CONSUMED") is True
            and audit.get("new_S3BJ128_CPU_acquired_only_after_S3BI_S3BJ_SHA_locks") is True
            and audit.get("S3BJ128_no_role_head_evaluation_or_training_this_run") is True
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
            "S3BK original 43M native CPU new128 provenance is invalid")
    require(sha_file(token_file)==audit["full_token_fp16_sha256"]
            and sha_file(feature_file)==audit["physical_feature_fp16_sha256"]
            and sha_file(labels_file)==audit["physical_graph_labels_sha256"],
            "original new128 physical FP16 SHA mismatch")
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
            "S3BK true original 43M token320 data/mask ABI drift")
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
    require(set(wa)==set(wb),"S3BA vs S3BI FP16 parameter schema")
    require(all(torch.equal(wa[k],wb[k]) for k in wa if not k.startswith("bridge.query.")),
            "frozen graph and native normalizer must be bitwise unchanged")
    x=normalize(hidden,wa["normalizer.mean"].float(),wa["normalizer.scale"].float(),mask)
    nr_file=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    nr_report=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    require(sha_file(nr_file)==nr_report["fp16_checkpoint_sha256"],"independent 2241p NRA SHA")
    nra=physical_frozen_nra(nr_file,nr_report["fp16_checkpoint_sha256"]).eval().requires_grad_(False)
    torch.set_num_threads(2)
    preds={}
    reference_edges=None
    with torch.inference_mode():
        for name,w in (("S3BA",wa),("S3BI",wb)):
            model=S3KGraphWithLearnedRoles().eval()
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
        **{"style_"+str(i):torch.tensor([r.query_style==i for r in rows]) for i in range(4)},
        **{"family_"+family:torch.tensor([r.family==family for r in rows]) for family in FAMILIES},
    }
    metrics={key:summarize(m,rows,gold_raw,graph_exact,preds) for key,m in groups.items()}
    require(metrics["all"]["n"]==N
            and metrics["source_first"]["n"]==64
            and metrics["destination_first"]["n"]==64
            and metrics["all"]["yes"]==64
            and metrics["all"]["paired_both_roles"]["S3BI_only_correct"]+
                metrics["all"]["paired_both_roles"]["S3BA_only_correct"]<=N,
            "locked independent128 split and paired comparison")
    require(sha_file(token_file)==audit["full_token_fp16_sha256"]
            and sha_file(a_file)==S3BA_SHA and sha_file(b_file)==S3BI_SHA,
            "FP16 provenance changed after one-time inference")
    result={
        "stage":"NRA2_S3BL_ONCE_ONLY_NEW_S3BJ_BLIND128_S3BA_VS_S3BI",
        "S3BK_run":S3BK_RUN,
        "S3BJ_prereg_commitment_sha256":S3BJ_SHA,
        "S3BJ_prereg_successful_run":S3BJ_PREREG_RUN,
        "S3BK_original_CPU_token_fp16_sha256":audit["full_token_fp16_sha256"],
        "S3BK_gold_fp16_sha256":audit["physical_graph_labels_sha256"],
        "S3BA_fp16_sha256":S3BA_SHA,"S3BI_fp16_sha256":S3BI_SHA,
        "original_616_S194_J7_preserved_in_S3BK":True,
        "prior_S3AQ_old_independent_EVAL128_consumed":True,
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
    (out/"NRA2_S3BL_ONCE_ONLY_NEW128_S3BA_VS_S3BI_NO_RETUNE.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"stage":result["stage"],"primary":metrics["all"],
                     "source_first":metrics["source_first"],
                     "destination_first":metrics["destination_first"],
                     "output_gate":0.,"candidate_accepted":False},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    if len(sys.argv)!=7:
        raise SystemExit("Usage: S3BL S3BK_CPU_DIR S3BJ_DIR S3BA_DIR S3BI_DIR NRA_DIR OUTPUT_DIR")
    main(*(Path(p) for p in sys.argv[1:]))
