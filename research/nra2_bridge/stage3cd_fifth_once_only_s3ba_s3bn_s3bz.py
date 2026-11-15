"""NRA2 S3CD: FIFTH once-only untouched S3CB EVAL128, original43M CPU S3BA vs JOINT S3BZ.

Read-only research experiment. No DEV scoring, training, tuning, model selection,
threshold search, repeated candidate comparisons or production mutations.
FIFTH untouched S3CB SHA prereg locked before genuine S3CC CPU; S3BZ physically frozen after TRAIN2240 and four prior independent EVALs consumed.
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

from research.nra2_bridge.stage3cb_fifth_future_unseen128_prereg import (
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
S3CB_SHA="a23dd34eac45dfa17d00798c2e3837c6c837c490a9ba50e809d73d3c3374cd5b"
S3CB_PREREG_RUN=43675557570
S3BA_SHA="4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BZ_SHA="6084061cb1f9ea46ef3b504179bee09b392f4427ac52ec58d768645da98872ee"
S3BN_SHA="ea8a81665370cba22c4e8bc75fbe82314089244d4b32278ee62c2c30d88ae811"
S3CC_RUN=43676015708
S3CC_TOKEN_FP16_SHA="SHA256_PENDING_FINAL_S3CC_CAPTURE"
S3CC_GOLD_FP16_SHA="SHA256_PENDING_FINAL_S3CC_CAPTURE"

def require(ok,why):
    if not bool(ok):
        raise AssertionError("S3CD fail-closed: "+why)

def exact_two_sided_mcnemar(left_only:int,right_only:int)->float:
    n=left_only+right_only
    if n==0:
        return 1.0
    tail=sum(math.comb(n,k) for k in range(min(left_only,right_only)+1))
    return min(1.0,2.0*tail/(2**n))

def summarize(mask,rows,gold,graph_exact,preds):
    ids=torch.where(mask)[0]
    require(ids.numel()>0,"empty preregistered subgroup")
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
        paired={}
        for old,new in (("S3BA","S3BN"),("S3BA","S3BZ"),("S3BN","S3BZ")):
            a=preds[old][metric][ids]
            b=preds[new][metric][ids]
            a_only=int((a&~b).sum())
            b_only=int((b&~a).sum())
            paired[old+"_vs_"+new]={
                old+"_only_correct":a_only,
                new+"_only_correct":b_only,
                "both_correct":int((a&b).sum()),
                "both_wrong":int((~a&~b).sum()),
                "net_"+new+"_minus_"+old:b_only-a_only,
                "exact_two_sided_mcnemar_p":exact_two_sided_mcnemar(a_only,b_only),
            }
        result["paired_"+metric]=paired
    return result

def main(cpu:Path,pre:Path,adir:Path,ndir:Path,bdir:Path,nra_dir:Path,out:Path):
    require(not out.exists() or not any(out.iterdir()),
            "one-time independent evaluation cannot overwrite prior report")
    # FIFTH once-only S3CB independent128 prereg was SEALED in original
    # data-only CI #43649584951 BEFORE genuine original 43M EVAL CPU S3CC.
    # S3BZ pairwise head was physically SHA locked from TRAIN2240 in
    # successful CI #43674502209, BEFORE any FIFTH EVAL CPU or model scores.
    prior=json.loads((pre/"NRA2_S3CB_FIFTH_UNTOUCHED128_PREREG.json").read_text())
    rows=build_rows()
    require(len(rows)==N and validate(rows)
            and len({r.topology_key for r in rows})==16
            and prior==json.loads(json.dumps(commitment(rows)))
            and prior.get("commitment_sha256")==S3CB_SHA
            and prior.get("stage")=="NRA2_S3CB_FIFTH_UNTOUCHED_EVAL128_PREREG_NO_CPU_OR_MODEL"
            and prior.get("split")==EVAL
            and prior.get("future_EVAL128_count")==128
            and prior.get("fifth_independent_graphs")==16
            and prior.get("fifth_independent_new_grammar")==8
            and prior.get("source_first_count")==64
            and prior.get("destination_first_count")==64
            and prior.get("yes_count")==64 and prior.get("no_count")==64
            and prior.get("sha_prompts_in_order")==[r.prompt_sha256 for r in rows]
            and prior.get("S3BO_prior_third_consumed_commitment_sha256")=="01858b405fbb034055574fd27feb93d19a34833da342a664ca5daed2b3ad091b"
            and prior.get("S3BR_prior_new_TRAIN512_commitment_sha256")=="e941bd45f947b7fc16f0cce4307ed9ea69c0b0e8347da7c0754717430ff0262f"
            and prior.get("prior_first_independent_EVAL128_consumed_in")==43640363396
            and prior.get("prior_second_independent_EVAL128_consumed_in")==43644153544
            and prior.get("prior_third_independent_EVAL128_consumed_in")==43648038071
            and prior.get("prior_fourth_independent_EVAL128_consumed_in")==43651233135
            and prior.get("S3BU_prior_fourth_consumed_commitment_sha256")==
                "117fcd79776c6d163b572f8d04b81743429d8c72e74cd072b6a5bedbb38c7f7e"
            and prior.get("S3BZ_safe_frozen_before_fifth_EVAL_sha256")==S3BZ_SHA
            and prior.get("ALL_previous_independent_EVALs_excluded_from_optimizer_and_candidate_selection") is True
            and prior.get("S3BZ_candidate_physically_SHA_locked_before_fifth_EVAL_prereg") is True
            and prior.get("no_model_weights_or_Calibri_CPU_or_scores_or_training_this_stage") is True
            and prior.get("candidate_accepted") is False and prior.get("output_gate")==0.,
            "FIFTH untouched128 original S3CB SHA/graph/grammar validation failed")
    a_file=adir/"NRA2_S3BA_S3AW_INIT_ROLE1536_FP16.safetensors"
    n_file=ndir/"NRA2_S3BN_JOINT_ROLE_SAFE_TRAIN2048_FP16.safetensors"
    b_file=bdir/"NRA2_S3BZ_SAFE_JOINT_RELATION_TRAIN2240_FP16.safetensors"
    require(sha_file(a_file)==S3BA_SHA and sha_file(n_file)==S3BN_SHA
            and sha_file(b_file)==S3BZ_SHA,
            "physically trained 2240 TRAIN-only S3BZ or frozen S3BA FP16 SHA changed")
    cand=json.loads((bdir/"NRA2_S3BZ_DIRECTIONAL_TRAIN2240_AUDIT.json").read_text())
    require(cand.get("stage")=="NRA2_S3BZ_SEMANTIC_DIRECTIONAL_TRAIN2240_ONLY"
            and cand.get("S3BN_initialized_joint_pairwise_fp16_sha256")==S3BN_SHA
            and cand.get("S3BR_newTRAIN512_preregistered_commitment_sha256")=="e941bd45f947b7fc16f0cce4307ed9ea69c0b0e8347da7c0754717430ff0262f"
            and cand.get("S3BS_newTRAIN512_original43m_cpu_fp16_sha256")=="95912db37d26077f4abb2e1e4050aad13155c3988aa7ee592e6da8d81c3eec5c"
            and cand.get("physical_fp16_safe_sha256")==S3BZ_SHA
            and cand.get("train_only_selected_alpha")==0.0625
            and cand.get("train_count")==2240
            and cand.get("new_S3BR_TRAIN_count")==192
            and cand.get("guard320_excluded_from_all_training_inference_and_selection") is True
            and cand.get("S3BY_verified_parent_and_minibatch_gate_every_optimizer_step") is True
            and cand.get("rejected_S3BZ_never_initialized") is True
            and cand.get("original_S194_616_J7_unchanged_in_all_seven_CPU_sources") is True
            and cand.get("new_S3BM_only_pair_params_trained")==42869
            and cand.get("no_heldout_DEV_or_EVAL_training_scoring_or_selection") is True
            and cand.get("old_graph_and_original_S3BA_query_frozen_physical_FP16_equal") is True
            and cand.get("original_normalizer_frozen_physical_FP16_equal") is True
            and cand.get("candidate_accepted") is False and cand.get("output_gate")==0.,
            "locked S3BZ semantic relation-trained original 43M CPU/616 J7 train-only provenance invalid")
    audit=json.loads((cpu/"NRA2_S3CC_EVAL128_REAL_CPU_J7_AUDIT.json").read_text())
    token_file=cpu/"NRA2_S3CC_REAL_CPU_EVAL128_ALL_TOKENS_FP16.safetensors"
    feature_file=cpu/"NRA2_S3CC_REAL_43M_EVAL128_HIDDEN320_FP16.safetensors"
    labels_file=cpu/"NRA2_S3CC_EVAL128_STRUCTURAL_LABELS.safetensors"
    require(audit.get("stage")=="NRA2_S3CC_ORIGINAL43M_FIFTH_UNTOUCHED128_CPU_NO_SCORE"
            and audit.get("S3CB_fifth_future_prereg_sha256")==S3CB_SHA
            and audit.get("S3CB_fifth_future_prereg_run")==S3CB_PREREG_RUN
            and audit.get("S3BZ_joint_train2240_physically_pinned_candidate_sha256")==S3BZ_SHA
            and audit.get("S3BZ_joint_train2240_success_run")==43674502209
            and audit.get("S3BA_original_frozen_baseline_sha256")==S3BA_SHA
            and audit.get("fifth_new_independent_graphs")==16
            and audit.get("fifth_new_independent_grammar_styles")==8
            and audit.get("fifth_yes")==64 and audit.get("fifth_destination_first")==64
            and audit.get("four_previous_independent_EVALs_permanently_consumed_excluded") is True
            and audit.get("fifth_original_CPU_only_after_new_model_and_data_sha_locks") is True
            and audit.get("fifth_no_model_role_or_reachability_scored_this_run") is True
            and audit.get("full_token_fp16_sha256")==S3CC_TOKEN_FP16_SHA
            and audit.get("physical_graph_labels_sha256")==S3CC_GOLD_FP16_SHA
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
            "S3CC authentic FIFTH untouched128 original 43M CPU physical SHA/J7 invalid")
    require(sha_file(token_file)==audit["full_token_fp16_sha256"]
            and sha_file(feature_file)==audit["physical_feature_fp16_sha256"]
            and sha_file(labels_file)==audit["physical_graph_labels_sha256"],
            "FIFTH EVAL128 token/feature/gold data FP16 physical SHA mismatch")
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
            "S3CC true original 43M token320 data/mask ABI drift")
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
    wn=load_file(str(n_file),device="cpu")
    wb=load_file(str(b_file),device="cpu")
    # Architectures intentionally have DIFFERENT parameter namespaces.
    # Only the original frozen graph, normalizer and S3BA query base must
    # remain BITWISE identical. New S3BZ pair layers are the sole change.
    require(all(k in wn and k in wb and torch.equal(wn[k],v) and torch.equal(wb[k],v) for k,v in wa.items()
                if k.startswith("bridge.graph.") or k.startswith("normalizer.")),
            "old graph/normalizer changed in new semantic architecture")
    require(all("bridge.query.base."+k.removeprefix("bridge.query.") in wb
                and "bridge.query.base."+k.removeprefix("bridge.query.") in wn
                and torch.equal(wb["bridge.query.base."+k.removeprefix("bridge.query.")],v)
                and torch.equal(wn["bridge.query.base."+k.removeprefix("bridge.query.")],v)
                for k,v in wa.items() if k.startswith("bridge.query.")),
            "original frozen S3BA query teacher changed in novel S3BZ")
    require(abs(float(wb["bridge.query.residual_scale"].item())-0.0625)<1e-6
            and abs(float(wn["bridge.query.residual_scale"].item())-1.)<1e-6,
            "S3BZ fixed trained joint residual scale changed")
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
                           ("S3BN",wn,S3BMGraphWithJointRoles),
                           ("S3BZ",wb,S3BMGraphWithJointRoles)):
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
            and set(metrics["all"]["paired_both_roles"])==
                {"S3BA_vs_S3BN","S3BA_vs_S3BZ","S3BN_vs_S3BZ"},
            "locked fifth independent128 split and paired comparison")
    require(sha_file(token_file)==audit["full_token_fp16_sha256"]
            and sha_file(a_file)==S3BA_SHA and sha_file(n_file)==S3BN_SHA and sha_file(b_file)==S3BZ_SHA,
            "FP16 provenance changed after one-time inference")
    result={
        "stage":"NRA2_S3CD_FIFTH_ONCE_ONLY_S3CB128_S3BA_VS_S3BN_VS_JOINT_S3BZ",
        "S3CC_run":S3CC_RUN,
        "S3CB_prereg_commitment_sha256":S3CB_SHA,
        "S3CB_prereg_successful_run":S3CB_PREREG_RUN,
        "S3CC_original_CPU_token_fp16_sha256":audit["full_token_fp16_sha256"],
        "S3CC_gold_fp16_sha256":audit["physical_graph_labels_sha256"],
        "S3BA_fp16_sha256":S3BA_SHA,"S3BZ_fp16_sha256":S3BZ_SHA,
        "S3BZ_training_FIT192_plus_2048_replay_verified":True,
        "S3BN_fp16_sha256":S3BN_SHA,
        "original_616_S194_J7_preserved_in_S3CC":True,
        "prior_S3AQ_old_independent_EVAL128_consumed":True,
        "prior_S3BJ_second_independent_EVAL128_consumed":True,
        "prior_S3BO_third_independent_EVAL128_consumed":True,
        "S3BM_42869_new_pairwise_params":True,
        "fifth_unseen_16_topologies_and_8_new_grammars":True,
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
    (out/"NRA2_S3CD_FIFTH_ONCE_ONLY128_S3BA_S3BN_S3BZ_NO_RETUNE.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"stage":result["stage"],"primary":metrics["all"],
                     "source_first":metrics["source_first"],
                     "destination_first":metrics["destination_first"],
                     "output_gate":0.,"candidate_accepted":False},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    if len(sys.argv)!=8:
        raise SystemExit("Usage: S3CD S3CC_CPU_DIR S3CB_DIR S3BA_DIR S3BN_DIR S3BZ_DIR NRA_DIR OUTPUT_DIR")
    main(*(Path(p) for p in sys.argv[1:]))
