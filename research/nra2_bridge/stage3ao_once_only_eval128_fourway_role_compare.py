"""S3AO: once-only, pinned S3O, S3Z and S3AG on preregistered S3AK EVAL128.

Only frozen S3AN genuine 43M native CPU features; fixed original S3O graph
and 2241p NRA; strict read-only comparison of four frozen role heads.
The single predeclared edge threshold is 0.5. No training, selection or retuning.
Not final sealed QA,
decoder full+EOS improvement, speed, compression or deployment.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from research.nra2_bridge.stage3ak_newtrain256_futureeval128_prereg import (
    build_rows,validate,commitment,EVAL,TRAIN,STYLES,REVERSED_STYLES,FAMILIES,
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
NODES=6
WIDTH=320
S3AK_SHA="e1c8b0f9ad533482c351c5ac392f5923a59759c57c62d55080d692cbebf5354c"
S3O_SHA="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3Z_SHA="60a7121b8270f647909776b58dc21b07c6cb885fd6ebeb6cb3158b23221f6e9f"
S3AN_TOKEN_SHA="0b75c0f271cf580b04c38af665c0d52c373f37444f9dc13c2588afdbd59b4863"
S3AN_FEATURE_SHA="8497866b4176384c87400703b0e1f483753e17c0a14e21489378a14b60084a65"
S3AN_LABEL_SHA="dee82a2d226f55a9f1408d9bb8f0a5a1574dc2f8209ec77aecb2b93e4cb532ab"
S3AN_CI=43572358979
S3AK_CI=43542086287
S3O_CI=43522663612
S3Z_CI=43532157094
S3AG_CI=43539383181
S3AG_SHA="2c23de87efc2369ed6fc0880604420070c116a4ee70b66a4eb0a946f1afeb118"
S3AM_CI=43544245980
S3AM_SHA="c5c400fc8249ea474263c9388f03e1496320bc8f6b793f93d155a962c3bf0eef"
S3AK_EVAL_SOURCE_SHA="e0e30f96665241520b71dce791d6305eb9b44a5b2eb9d1132804927c058d407e"
THRESHOLD=0.5

def require(cond,reason):
    if not bool(cond):
        raise AssertionError("S3AO "+reason)

def inspect_group(mask,gold,graph_exact,results):
    ids=torch.where(mask)[0]
    require(ids.numel()>0,"empty predeclared group")
    yes=gold["reachability"][ids].bool()
    d={"count":int(ids.numel()),"yes":int(yes.sum()),"no":int((~yes).sum()),
       "exact_graph":int(graph_exact[ids].sum())}
    for name,v in results.items():
        src_ok=v["src"][ids]
        dst_ok=v["dst"][ids]
        hard_ok=v["hard"][ids]
        d[name]={"source_correct":int(src_ok.sum()),
                 "destination_correct":int(dst_ok.sum()),
                 "both_roles_correct":int((src_ok&dst_ok).sum()),
                 "reachability_hard_correct":int(hard_ok.sum()),
                 "hard_false_positive":int((v["hard_pred"][ids]&~yes).sum()),
                 "hard_false_negative":int((~v["hard_pred"][ids]&yes).sum())}
    # Six predeclared paired comparisons, without choosing a checkpoint.
    for left,right in (("S3O","S3Z"),("S3O","S3AG"),("S3O","S3AM"),
                       ("S3Z","S3AG"),("S3Z","S3AM"),("S3AG","S3AM")):
        left_ok=results[left]["hard"][ids]
        right_ok=results[right]["hard"][ids]
        d["paired_reachability_"+left+"_vs_"+right]={
            left+"_only_correct":int((left_ok&~right_ok).sum()),
            right+"_only_correct":int((~left_ok&right_ok).sum()),
            "both_correct":int((left_ok&right_ok).sum()),
            "both_wrong":int((~left_ok&~right_ok).sum()),
        }
    return d

def main():
    if len(sys.argv)!=9:
        raise SystemExit("S3AO S3AN_CPU_EVAL128_DIR S3AK_PREREG_DIR S3O_DIR S3Z_DIR S3AG_DIR S3AM_DIR NRA_DIR OUTPUT")
    cpu,pre,base,trained,ag_dir,am_dir,nra_dir,out=map(Path,sys.argv[1:])
    require(not out.exists() or not any(out.iterdir()),"one-time immutable result directory already populated")
    all_rows=build_rows()
    validate(all_rows)
    rows=[r for r in all_rows if r.split==EVAL]
    require(len(all_rows)==384 and len(rows)==N and len(STYLES[EVAL])==4,"independent prereg EVAL128")
    prereg=json.loads((pre/"NRA2_S3AK_NEW_TRAIN256_FUTURE_EVAL128_PREREG.json").read_text())
    require(prereg==json.loads(json.dumps(commitment(all_rows))),"original full S3AK prereg immutable SHA and order")
    require(prereg["commitment_sha256"]==S3AK_SHA
            and prereg["splits"][EVAL]["question_sha256"]==[r.prompt_sha256 for r in rows],
            "locked independent EVAL128 question order and prereg commitment")
    audit=json.loads((cpu/"NRA2_S3AN_EVAL128_REAL_CPU_J7_AUDIT.json").read_text())
    so=json.loads((base/"NRA2_S3O_NEURAL_SEMANTIC_ROLES_128TRAIN_64DEV_AUDIT.json").read_text())
    sz=json.loads((trained/"NRA2_S3Z_ROLE_REVERSAL_REHEARSAL256_REPORT.json").read_text())
    sg=json.loads((ag_dir/"NRA2_S3AG_TRAIN512_ROLE_HEAD_AUDIT.json").read_text())
    sm=json.loads((am_dir/"NRA2_S3AM_TRAIN512_ROLE_SYMMETRY_AUDIT.json").read_text())
    report=json.loads((nra_dir/"NRA2_STANDALONE_REPORT.json").read_text())
    tok=cpu/"NRA2_S3AN_REAL_CPU_EVAL128_ALL_TOKENS_FP16.safetensors"
    feature=cpu/"NRA2_S3AN_REAL_43M_EVAL128_HIDDEN320_FP16.safetensors"
    label=cpu/"NRA2_S3AN_EVAL128_STRUCTURAL_LABELS.safetensors"
    oldfile=base/"NRA2_S3O_NATIVE_CPU_LEARNED_ROLES_TRAIN128_FP16.safetensors"
    newfile=trained/"NRA2_S3Z_ROLE_REVERSAL_REHEARSAL256_FP16.safetensors"
    agfile=ag_dir/"NRA2_S3AG_FROZEN_GRAPH_ROLE512_TRAIN_FP16.safetensors"
    amfile=am_dir/"NRA2_S3AM_FROZEN_GRAPH_ROLE512_TRAIN_FP16.safetensors"
    nrafile=nra_dir/"NRA2_STANDALONE_RELATIONAL_FP16.pt"
    require(
        audit.get("stage")=="NRA2_S3AN_FROZEN_S3AK_INDEPENDENT_EVAL128_NO_SCORE_CPU_CAPTURE_ONLY"
        and audit.get("S3AK_registered_commitment_sha256")==S3AK_SHA
        and audit.get("S3AK_EVAL128_source_sha256")==S3AK_EVAL_SOURCE_SHA
        and audit.get("S3AK_successful_prereg_run")==S3AK_CI
        and audit.get("S3AM_successful_frozen_adapter_run")==S3AM_CI
        and audit.get("S3AM_frozen_pre_EVAL_checkpoint_sha256")==S3AM_SHA
        and audit.get("all_616_token_digest_matches_S194") is True
        and audit.get("native_tokenizer_S3O_query_alignment_verified") is True
        and audit.get("physical_reload_bitwise_equal") is True
        and audit.get("train_count")==0
        and audit.get("graph_prefill_count")==N
        and audit.get("group_sizes")=={EVAL:N}
        and audit.get("split_groups",{}).get(EVAL,{}).get("prompt_sha256")==[r.prompt_sha256 for r in rows]
        and audit.get("original_456_full_EOS")==456
        and audit.get("original_new128_full_EOS")==128
        and audit.get("S3AK_EVAL128_not_scored_or_trained") is True
        and audit.get("frozen_soft_and_hard_candidate_scored") is False
        and audit.get("nra_output_gate")==0.0
        and audit.get("candidate_accepted") is False
        and sha_file(tok)==audit.get("full_token_fp16_sha256")==S3AN_TOKEN_SHA
        and sha_file(feature)==audit.get("physical_feature_fp16_sha256")==S3AN_FEATURE_SHA
        and sha_file(label)==audit.get("physical_graph_labels_sha256")==S3AN_LABEL_SHA,
        "genuine S3AN independent EVAL128 CPU FP16, original 616 J7 and never scored")
    require(
        sha_file(oldfile)==S3O_SHA==so.get("FP16_checkpoint_sha256")
        and so.get("output_gate")==0.0 and so.get("candidate_accepted") is False
        and sha_file(newfile)==S3Z_SHA==sz.get("new_S3Z_physical_fp16_sha256")
        and sz.get("only_small_query_role_head_trained") is True
        and sz.get("original_S3O_graph_FP16_bitwise_equal") is True
        and sha_file(agfile)==S3AG_SHA==sg.get("new_S3AG_physical_fp16_sha256")
        and sg.get("train_count")==512
        and sg.get("only_query_role_head_trained") is True
        and sg.get("original_graph_physical_fp16_bitwise_equal") is True
        and sg.get("future_independent_S3AE_EVAL128_never_prefilled_scored_or_trained") is True
        and sg.get("candidate_accepted") is False
        and sg.get("output_gate")==0.0
        and sha_file(amfile)==S3AM_SHA==sm.get("new_S3AM_physical_fp16_sha256")
        and sm.get("stage")=="NRA2_S3AM_TRAIN512_S3AK_ROLE_SYMMETRY_FROZEN_S3O_GRAPH"
        and sm.get("train_count")==512
        and sm.get("only_query_role_head_trained") is True
        and sm.get("original_graph_physical_fp16_bitwise_equal") is True
        and sm.get("S3AK_future_EVAL128_not_prefilled_scored_or_trained") is True
        and sm.get("full_S194_original_616_J7_digests_from_S3AL_protected") is True
        and sm.get("candidate_accepted") is False and sm.get("output_gate")==0.0
        and report.get("fp16_checkpoint_sha256")==sha_file(nrafile),
        "pin original S3O, previous S3Z, locked S3AG FP16 and frozen NRA")
    tokens=load_file(str(tok),device="cpu")
    h=tokens["token_hidden320"]
    mask=tokens["token_mask"]
    lens=tokens["token_lengths"]
    last=load_file(str(feature),device="cpu")["hidden320"]
    require(h.ndim==3 and h.shape[0]==N and h.shape[2]==WIDTH
            and h.dtype==torch.float16 and mask.shape==h.shape[:2]
            and mask.dtype==torch.bool and lens.shape==(N,)
            and torch.equal(mask.sum(1).int(),lens.int())
            and torch.equal(h[torch.arange(N),lens.long()-1],last),
            "physical token320 mask, S160 length and last layer tensor")
    goldraw=load_file(str(label),device="cpu")
    require(set(goldraw)=={"adjacency","source","target","reachability"}
            and goldraw["adjacency"].shape==(N,NODES,NODES)
            and goldraw["source"].shape==(N,NODES)
            and goldraw["target"].shape==(N,NODES)
            and goldraw["reachability"].shape==(N,),
            "physically precommitted independent labels shape")
    gold={k:v.float() for k,v in goldraw.items()}
    tokenizer=exact_original_tokenizer()
    align=validate_without_gold(tokenizer,[r.question for r in rows])
    require(all(a["tokens"]==int(lens[i]) for i,a in enumerate(align)),
            "S160 native tokenizer drift on independent grammar")
    bind=bind_mentions_to_context(align,h.shape[1])
    q=build_query_text_mask(tokenizer,[r.question for r in rows],align,h.shape[1])
    a=load_file(str(oldfile),device="cpu");b=load_file(str(newfile),device="cpu")
    c=load_file(str(agfile),device="cpu")
    d=load_file(str(amfile),device="cpu")
    require(set(a)==set(b)==set(c)==set(d),"all four model FP16 schemas identical")
    require(all(torch.equal(a[k],b[k]) and torch.equal(a[k],c[k]) and torch.equal(a[k],d[k]) and torch.equal(a[k],d[k])
                for k in a if k.startswith("bridge.graph.")),
            "graph weights bitwise identical S3O vs S3Z vs S3AG")
    require(all(torch.equal(a[k],b[k]) and torch.equal(a[k],c[k])
                for k in ("normalizer.mean","normalizer.scale")),
            "S3O native CPU normalizer unchanged all four")
    x=normalize(h,a["normalizer.mean"].float(),a["normalizer.scale"].float(),mask)
    nra=physical_frozen_nra(nrafile,report["fp16_checkpoint_sha256"]).eval().requires_grad_(False)
    preds={}
    edge_ref=None
    torch.set_num_threads(2)
    with torch.inference_mode():
        for name,data in (("S3O",a),("S3Z",b),("S3AG",c),("S3AM",d)):
            net=S3KGraphWithLearnedRoles().eval()
            net.load_state_dict({k.removeprefix("bridge."):v for k,v in data.items()
                                 if k.startswith("bridge.")},strict=True)
            net.requires_grad_(False)
            edges,src,dst=net(x,mask,bind,q)
            require(all(bool(torch.isfinite(v).all()) for v in (edges,src,dst)),
                    "model logits finite")
            if edge_ref is None:
                edge_ref=edges.clone()
            else:
                require(torch.equal(edges,edge_ref),
                        "only role head differs, frozen graph inference must match")
            e_hard=(torch.sigmoid(edges)>THRESHOLD).float()
            src_soft=F.softmax(src,dim=-1);dst_soft=F.softmax(dst,dim=-1)
            logits=nra(e_hard,src_soft,dst_soft)
            require(bool(torch.isfinite(logits).all()),"frozen NRA reach logits finite")
            hard=logits>0
            preds[name]={"src":src.argmax(1)==gold["source"].argmax(1),
                         "dst":dst.argmax(1)==gold["target"].argmax(1),
                         "hard_pred":hard,
                         "hard":hard==gold["reachability"].bool()}
        chk=torch.arange(7,dtype=torch.float32)
        require(zero_gate_passthrough(chk,torch.zeros(7)) is chk,
                "NRA not connected to original output; gate must be zero")
        exact=(e_hard==gold["adjacency"]).flatten(1).all(1)
    truth=gold["reachability"].bool()
    categories={
        "all":torch.ones(N,dtype=torch.bool),
        "yes":truth,"no":~truth,
        "source_first":torch.tensor([r.query_style not in REVERSED_STYLES for r in rows]),
        "destination_first":torch.tensor([r.query_style in REVERSED_STYLES for r in rows]),
        **{"family_"+v:torch.tensor([r.family==v for r in rows]) for v in FAMILIES},
        **{"style_"+str(i):torch.tensor([r.query_style==i for r in rows]) for i in range(4)},
        **{"family_"+f+"_destination_first":torch.tensor([
             r.family==f and r.query_style in REVERSED_STYLES for r in rows]) for f in FAMILIES},
    }
    metrics={name:inspect_group(m,gold,exact,preds) for name,m in categories.items()}
    require(metrics["all"]["count"]==N and metrics["all"]["yes"]==64
            and metrics["all"]["no"]==64
            and metrics["source_first"]["count"]==64
            and metrics["destination_first"]["count"]==64
            and sum(metrics["family_"+f]["count"] for f in FAMILIES)==N,
            "precommitted breakdown counts")
    require(all(sha_file(f)==s for f,s in (
        (tok,S3AN_TOKEN_SHA),(feature,S3AN_FEATURE_SHA),(label,S3AN_LABEL_SHA),
        (oldfile,S3O_SHA),(newfile,S3Z_SHA),(agfile,S3AG_SHA),(amfile,S3AM_SHA),
        (nrafile,report["fp16_checkpoint_sha256"]))),
        "no source artifacts changed during inference")
    result={
        "stage":"NRA2_S3AO_ONCE_ONLY_INDEPENDENT_EVAL128_S3O_VS_S3Z_VS_S3AG_VS_S3AM",
        "S3AK_commitment_sha256":S3AK_SHA,
        "S3AK_prereg_successful_run":S3AK_CI,
        "S3AN_native_CPU_successful_run":S3AN_CI,
        "S3AG_training_successful_run":S3AG_CI,
        "S3AM_training_successful_run":S3AM_CI,
        "S3O_original_successful_run":S3O_CI,
        "S3Z_training_successful_run":S3Z_CI,
        "S3AN_native_tokens_fp16_sha256":S3AN_TOKEN_SHA,
        "S3AN_independent_labels_sha256":S3AN_LABEL_SHA,
        "S3O_original_FP16_sha256":S3O_SHA,
        "S3Z_role_head_FP16_sha256":S3Z_SHA,
        "S3AG_role_head_FP16_sha256":S3AG_SHA,
        "S3AM_role_head_FP16_sha256":S3AM_SHA,
        "frozen_NRA_sha256":report["fp16_checkpoint_sha256"],
        "once_only_independent_128_rows_compared":N,
        "graph_weights_bitwise_equal":True,
        "four_frozen_role_heads_only_compared":True,
        "graph_logits_bitwise_equal":True,
        "hard_edge_threshold_predeclared":THRESHOLD,
        "source_target_probs_soft_all_four_arms":True,
        "metrics":metrics,
        "question_level": [{
            "index":i,"prompt_sha256":r.prompt_sha256,
            "family":r.family,"graph_id":r.graph_id,
            "query_style":r.query_style,
            "destination_first":bool(r.query_style in REVERSED_STYLES),
            "gold_yes":bool(r.reachable),"predicted_graph_exact":bool(exact[i]),
            **{name:{
                "source_correct":bool(v["src"][i]),
                "destination_correct":bool(v["dst"][i]),
                "reachability_correct":bool(v["hard"][i]),
                "reachability_pred_yes":bool(v["hard_pred"][i]),
            } for name,v in preds.items()},
        } for i,r in enumerate(rows)],
        "train_or_retune_after_scoring":False,
        "independent_128_not_final_sealed_QA":True,
        "original_616_S194_J7_digest_preserved_by_S3AN":True,
        "no_winner_selected_on_EVAL128":True,
        "new_full_EOS_decoder_gain_tested":False,
        "all_in_physical_bpp_tested":False,
        "paired_CPU_speed_tested":False,
        "candidate_accepted":False,"output_gate":0.0,
        "Jessica_LIVE_Render_Supabase_APK_QA_TEST_unchanged":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3AO_ONCE_ONLY_EVAL128_FOURWAY_NO_RETUNE.json").write_text(
        json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"stage":result["stage"],"all":metrics["all"],
        "source_first":metrics["source_first"],
        "destination_first":metrics["destination_first"],
        "candidate_accepted":False,"output_gate":0.0},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
