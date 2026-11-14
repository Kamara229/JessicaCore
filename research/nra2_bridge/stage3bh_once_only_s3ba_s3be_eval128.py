"""NRA2 S3BH: one-time locked S3BA vs S3BE on fresh independent S3AQ EVAL128.

Read-only research experiment. No DEV scoring, training, tuning, model selection,
threshold search, repeated candidate comparisons or production mutations.
The one-time protocol was physically SHA sealed in S3BF before any EVAL CPU.
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

from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows,validate,commitment,EVAL,TRAIN,DEV,REVERSED_STYLES,FAMILIES,
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
S3AQ_SHA="c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839"
S3BF_SHA="6286225dde86e444425bf853fd88811a472b4b962412c770a89b45c92d67ad2b"
S3BA_SHA="4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BE_SHA="9b6f0c455492c1fd508c38476a3505900898ef2ae08907329a3953751a93e513"
S3BG_RUN=43639726200

def require(ok,why):
    if not bool(ok):
        raise AssertionError("S3BH fail-closed: "+why)

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
        b=preds["S3BE"][metric][ids]
        a_only=int((a&~b).sum())
        b_only=int((b&~a).sum())
        result["paired_"+metric]={
            "S3BA_only_correct":a_only,"S3BE_only_correct":b_only,
            "both_correct":int((a&b).sum()),
            "both_wrong":int((~a&~b).sum()),
            "net_S3BE_minus_S3BA":b_only-a_only,
            "exact_two_sided_mcnemar_p":exact_two_sided_mcnemar(a_only,b_only),
        }
    return result

def main(cpu:Path,pre:Path,bfdir:Path,adir:Path,bdir:Path,nra_dir:Path,out:Path):
    require(not out.exists() or not any(out.iterdir()),
            "one-time output must not be overwritten")
    bf_bytes=(bfdir/"NRA2_S3BF_BLIND_EVAL128_PROTOCOL.json").read_bytes().rstrip(b"\n")
    protocol=json.loads(bf_bytes)
    bf=json.loads((bfdir/"NRA2_S3BF_PHYSICAL_AUDIT.json").read_text())
    require(hashlib.sha256(bf_bytes).hexdigest()==S3BF_SHA
            and bf["protocol_sha256"]==S3BF_SHA
            and protocol["stage"]=="NRA2_S3BF_PRECOMMIT_ONE_TIME_BLIND_S3AQ_EVAL128_NO_SCORE"
            and protocol["original_S3AQ_all_splits_prereg_commitment_sha256"]==S3AQ_SHA
            and protocol["blind_split_identifier"]==EVAL
            and protocol["blind_n"]==N
            and protocol["blind_unseen_topologies"]==32
            and protocol["primary_pair"]==["S3BA_FROZEN_FP16","S3BE_FROZEN_FP16"]
            and protocol["parent_fp16_sha256"]==S3BA_SHA
            and protocol["candidate_fp16_sha256"]==S3BE_SHA
            and protocol["candidate_accepted"] is False and protocol["output_gate"]==0.,
            "locked S3BF protocol changed or unapproved pair")

    a_file=adir/"NRA2_S3BA_S3AW_INIT_ROLE1536_FP16.safetensors"
    b_file=bdir/"NRA2_S3BE_S3BA_INIT_ROLE2048_FP16.safetensors"
    require(sha_file(a_file)==S3BA_SHA and sha_file(b_file)==S3BE_SHA,
            "physical model SHA does not match precommitted pair")
    rows_all=build_rows()
    require(len(rows_all)==512 and validate(rows_all),"source S3AQ split changed")
    rows=[r for r in rows_all if r.split==EVAL]
    prereg=json.loads((pre/"NRA2_S3AQ_TRAIN256_DEV128_FUTURE_EVAL128_PREREG.json").read_text())
    require(prereg==json.loads(json.dumps(commitment(rows_all)))
            and prereg["commitment_sha256"]==S3AQ_SHA
            and prereg["splits"][EVAL]["question_sha256"]==[r.prompt_sha256 for r in rows]
            and len(rows)==N and len({r.topology_key for r in rows})==32,
            "S3AQ independent EVAL128 order/topology mismatch")
    audit=json.loads((cpu/"NRA2_S3BG_EVAL128_REAL_CPU_J7_AUDIT.json").read_text())
    token_file=cpu/"NRA2_S3BG_REAL_CPU_EVAL128_ALL_TOKENS_FP16.safetensors"
    feature_file=cpu/"NRA2_S3BG_REAL_43M_EVAL128_HIDDEN320_FP16.safetensors"
    labels_file=cpu/"NRA2_S3BG_EVAL128_STRUCTURAL_LABELS.safetensors"
    require(audit.get("stage")=="NRA2_S3BG_FROZEN_S3AQ_INDEPENDENT_EVAL128_NO_SCORE_CPU_CAPTURE_ONLY"
            and audit.get("S3AQ_registered_commitment_sha256")==S3AQ_SHA
            and audit.get("S3BF_protocol_sha256")==S3BF_SHA
            and audit.get("S3BA_parent_fp16_sha256")==S3BA_SHA
            and audit.get("S3BE_candidate_fp16_sha256")==S3BE_SHA
            and audit.get("S3AQ_EVAL128_prefilled_only_after_S3BF_sealed") is True
            and audit.get("S3AQ_EVAL128_not_scored_or_trained") is True
            and audit.get("all_616_token_digest_matches_S194") is True
            and audit.get("train_count")==0 and audit.get("dev_count")==0
            and audit.get("eval_count")==N and audit.get("graph_prefill_count")==N
            and audit.get("group_sizes")=={EVAL:N}
            and audit.get("split_groups",{}).get(EVAL,{}).get("prompt_sha256")==[r.prompt_sha256 for r in rows]
            and audit.get("native_tokenizer_S3O_query_alignment_verified") is True
            and audit.get("physical_reload_bitwise_equal") is True
            and audit.get("frozen_soft_and_hard_candidate_scored") is False
            and audit.get("candidate_accepted") is False and audit.get("nra_output_gate")==0.,
            "authentic S3BG CPU EVAL128 provenance invalid")
    require(sha_file(token_file)==audit["full_token_fp16_sha256"]
            and sha_file(feature_file)==audit["physical_feature_fp16_sha256"]
            and sha_file(labels_file)==audit["physical_graph_labels_sha256"],
            "original EVAL128 physical FP16 SHA mismatch")
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
            "S3BG true original 43M token320 data/mask ABI drift")
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
    require(set(wa)==set(wb),"S3BA vs S3BE FP16 parameter schema")
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
        for name,w in (("S3BA",wa),("S3BE",wb)):
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
            and metrics["all"]["paired_both_roles"]["S3BE_only_correct"]+
                metrics["all"]["paired_both_roles"]["S3BA_only_correct"]<=N,
            "locked 128 split and paired comparison")
    require(sha_file(token_file)==audit["full_token_fp16_sha256"]
            and sha_file(a_file)==S3BA_SHA and sha_file(b_file)==S3BE_SHA,
            "FP16 provenance changed after one-time inference")
    result={
        "stage":"NRA2_S3BH_ONE_TIME_BLIND_S3AQ_EVAL128_S3BA_VS_S3BE",
        "S3BG_run":S3BG_RUN,"S3BF_protocol_sha256":S3BF_SHA,
        "S3AQ_commitment_sha256":S3AQ_SHA,
        "S3BG_original_CPU_token_fp16_sha256":audit["full_token_fp16_sha256"],
        "S3BG_gold_fp16_sha256":audit["physical_graph_labels_sha256"],
        "S3BA_fp16_sha256":S3BA_SHA,"S3BE_fp16_sha256":S3BE_SHA,
        "original_616_S194_J7_preserved_in_S3BG":True,
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
    (out/"NRA2_S3BH_ONE_TIME_EVAL128_S3BA_VS_S3BE_NO_RETUNE.json").write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"stage":result["stage"],"primary":metrics["all"],
                     "source_first":metrics["source_first"],
                     "destination_first":metrics["destination_first"],
                     "output_gate":0.,"candidate_accepted":False},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    if len(sys.argv)!=8:
        raise SystemExit("Usage: S3BH S3BG_CPU_DIR S3AQ_DIR S3BF_DIR S3BA_DIR S3BE_DIR NRA_DIR OUTPUT_DIR")
    main(*(Path(p) for p in sys.argv[1:]))
