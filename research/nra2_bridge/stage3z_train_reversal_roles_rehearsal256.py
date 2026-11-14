"""S3Z: isolated 256 TRAIN role-order remediation, preserving S3O graph and NRA.

Original S3M TRAIN128 rehearsed alongside physically captured S3X TRAIN128
(half new query wordings mention destination before source). Frozen original
43M CPU; ONLY S3O query-role adapter learns. DEV64 of S3M is explicitly
exploratory, once after physical FP16; S3T previously scored, not a blind.
No original changes, LIVE, Render, Supabase, QA, TEST or Jessica.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from safetensors.torch import load_file,save_file

from research.nra2_bridge.stage3m_role_balanced_graph_data import build_rows as s3m_rows
from research.nra2_bridge.stage3x_reversal_train_prereg import (
    build_rows as s3x_rows,commitment as s3x_commitment,
)
from research.nra2_bridge.stage3j_token_entity_alignment import validate_without_gold
from research.nra2_bridge.stage3k_neural_entity_bridge import bind_mentions_to_context
from research.nra2_bridge.stage3o_neural_query_roles import (
    build_query_text_mask,S3KGraphWithLearnedRoles,
)
from research.nra2_bridge.stage3k_train_real_entity_binding import (
    exact_original_tokenizer,sha_file,
)
from research.nra2_bridge.stage3f_token_attention import normalize

SEED=2026111419
STEPS=240
LR=0.0007
TRAIN=128
DEV=64
WIDTH=320
NODES=6
S3O_SHA="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3X_SHA="9628606e8f40d0efc6c7a41858a95fd86abed1ae1663caaba5f1802d75cdd22a"
S3N_SHA="b8a375982e556d67dfd5f741050c18c5460249184da9c3fcebc3d174d80db81b"
S3Y_SHA="4e5491ab5adfaca669c75debf4464fcc83b3465ca44e90ee57e0488dc96ae975"
S3Y_RUN=43530742301

def require(cond,msg):
    if not cond:
        raise AssertionError("S3Z "+msg)

def gold_roles(gold,lo,hi):
    return (gold["source"][lo:hi].argmax(1).long(),
            gold["target"][lo:hi].argmax(1).long())

def bind(tokenizer,questions,tokens,mask,lengths):
    al=validate_without_gold(tokenizer,questions)
    require(len(al)==len(questions)==len(tokens),"row counts")
    require(all(x["tokens"]==int(lengths[i]) for i,x in enumerate(al)),
            "S160 native CPU tokenizer length mismatch")
    require(tokens.ndim==3 and tokens.shape[2]==WIDTH and tokens.dtype==torch.float16
            and mask.shape==tokens.shape[:2] and mask.dtype==torch.bool
            and torch.equal(mask.sum(1).to(torch.int32),lengths.to(torch.int32))
            and bool(torch.isfinite(tokens).all()),"native token320/mask/length ABI")
    binding=bind_mentions_to_context(al,tokens.shape[1])
    query=build_query_text_mask(tokenizer,questions,al,tokens.shape[1])
    return query,binding

def role_logits(model,tokens,mask,q):
    return model.query(tokens,mask,q)

def loss_fn(model,old,novel):
    a,b=role_logits(model,*old[:3])
    c,d=role_logits(model,*novel[:3])
    lo=(F.cross_entropy(a,old[3])+F.cross_entropy(b,old[4]))*.5
    ln=(F.cross_entropy(c,novel[3])+F.cross_entropy(d,novel[4]))*.5
    return lo+ln,lo,ln

def role_metrics(model,data,rows):
    with torch.inference_mode():
        src,dst=role_logits(model,*data[:3])
        srcok=src.argmax(1)==data[3]
        dstok=dst.argmax(1)==data[4]
        forward=torch.tensor([r.query_style in (1,3) for r in rows]) if (
            rows and rows[0].split.startswith("S3X")) else torch.ones(len(rows),dtype=torch.bool)
        reverse=~forward
        def sl(m):
            return {"count":int(m.sum()),
                    "source_correct":int((srcok&m).sum()),
                    "target_correct":int((dstok&m).sum()),
                    "both_correct":int((srcok&dstok&m).sum())}
        return {"all":sl(torch.ones(len(rows),dtype=torch.bool)),
                "new_text_source_first":sl(forward),
                "new_text_destination_first":sl(reverse)}

def main():
    if len(sys.argv)!=6:
        raise SystemExit("S3Z S3N_CPU192_DIR S3Y_CPU128_DIR S3X_PREREG_DIR S3O_FP16_DIR OUT_DIR")
    s3n_dir,s3y_dir,s3x_dir,s3o_dir,out=map(Path,sys.argv[1:])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3Z new physical weights and results cannot overwrite")
    sn=json.loads((s3n_dir/"NRA2_S3N_192_ROLEBALANCED_REAL_CPU_AUDIT.json").read_text())
    sy=json.loads((s3y_dir/"NRA2_S3Y_128_REVERSAL_TRAIN_REAL_CPU_AUDIT.json").read_text())
    sc=json.loads((s3x_dir/"NRA2_S3X_TRAIN128_REVERSE_GRAMMAR_PREREG.json").read_text())
    so=json.loads((s3o_dir/"NRA2_S3O_NEURAL_SEMANTIC_ROLES_128TRAIN_64DEV_AUDIT.json").read_text())
    oldtok=s3n_dir/"NRA2_S3N_REAL_CPU_GRAPH192_ALL_TOKENS_FP16.safetensors"
    oldgold=s3n_dir/"NRA2_S3N_GRAPH192_STRUCTURAL_LABELS.safetensors"
    newtok=s3y_dir/"NRA2_S3Y_REAL_CPU_TRAIN128_ALL_TOKENS_FP16.safetensors"
    newgold=s3y_dir/"NRA2_S3Y_TRAIN128_STRUCTURAL_LABELS.safetensors"
    oldmodel=s3o_dir/"NRA2_S3O_NATIVE_CPU_LEARNED_ROLES_TRAIN128_FP16.safetensors"
    sxrows=s3x_rows()
    oldrows=s3m_rows()
    require(
        sc==s3x_commitment(sxrows)
        and sc["commitment_sha256"]==S3X_SHA
        and sc["original_S3T64_excluded_from_new_TRAIN"]
        and sc["original_S3T64_seen_during_failure_analysis_not_blind"]
        and sn.get("stage")=="NRA2_S3N_REAL_CPU_FROZEN_192_ROLEBALANCED_TOKEN_CAPTURE_ONLY"
        and sn.get("all_616_token_digest_matches_S194") is True
        and sn.get("physical_reload_bitwise_equal") is True
        and sn.get("full_token_fp16_sha256")==S3N_SHA==sha_file(oldtok)
        and sn.get("physical_graph_labels_sha256")==sha_file(oldgold)
        and sn.get("group_sizes")=={"TRAIN128":128,"DEV64":64}
        and sy.get("stage")=="NRA2_S3Y_FROZEN_128_REVERSAL_TRAIN_CPU_TOKEN_CAPTURE_ONLY"
        and sy.get("S3X_precommitted_TRAIN128_sha256")==S3X_SHA
        and sy.get("S3X_successful_prereg_run")==43530401603
        and sy.get("all_616_token_digest_matches_S194") is True
        and sy.get("physical_reload_bitwise_equal") is True
        and sy.get("full_token_fp16_sha256")==S3Y_SHA==sha_file(newtok)
        and sy.get("physical_graph_labels_sha256")==sha_file(newgold)
        and sy.get("graph_prefill_count")==TRAIN
        and sy.get("train_count")==TRAIN
        and sy.get("candidate_accepted") is False
        and sy.get("nra_output_gate")==0.0
        and sy.get("native_tokenizer_S3O_query_alignment_verified") is True
        and so.get("FP16_checkpoint_sha256")==S3O_SHA==sha_file(oldmodel)
        and so.get("candidate_accepted") is False
        and so.get("source_S3N_token_FP16_sha256")==S3N_SHA
        and so.get("output_gate")==0.0,
        "S3N S3Y S3X S3O frozen provenance and original 616 J7 must pass"
    )
    require(len(sxrows)==TRAIN and len(oldrows)==TRAIN+DEV,"fixed dataset rows")
    for group,start,count in (("TRAIN128",0,TRAIN),("DEV64",TRAIN,DEV)):
        require(sn["split_groups"][group]["prompt_sha256"]==
                [r.prompt_sha256 for r in oldrows[start:start+count]],
                "original S3N prompt SHA/order drift")
    require(sy["split_groups"][sc["group"]]["prompt_sha256"]==
            [r.prompt_sha256 for r in sxrows],"new S3Y prompt SHA/order drift")
    require(len(set(r.prompt_sha256 for r in sxrows)&
                set(r.prompt_sha256 for r in oldrows))==0,
            "S3X SHA overlaps original S3M")
    orig=load_file(str(oldtok),device="cpu")
    new=load_file(str(newtok),device="cpu")
    labels_orig=load_file(str(oldgold),device="cpu")
    labels_new=load_file(str(newgold),device="cpu")
    require(set(labels_orig)==set(labels_new)==
            {"adjacency","source","target","reachability"},"gold schemas drift")
    for key,size in ((labels_orig,TRAIN+DEV),(labels_new,TRAIN)):
        require(key["source"].shape==(size,NODES) and
                key["target"].shape==(size,NODES),"role target ABI drift")
    tokenizer=exact_original_tokenizer()
    a="token_hidden320";m="token_mask";length_key="token_lengths"
    oldq,_=bind(tokenizer,[r.question for r in oldrows[:TRAIN]],
                orig[a][:TRAIN],orig[m][:TRAIN],orig[length_key][:TRAIN])
    newq,_=bind(tokenizer,[r.question for r in sxrows],
                new[a],new[m],new[length_key])
    origweights=load_file(str(oldmodel),device="cpu")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model=S3KGraphWithLearnedRoles()
    model.load_state_dict({k.removeprefix("bridge."):v for k,v in origweights.items()
                           if k.startswith("bridge.")},strict=True)
    model.graph.requires_grad_(False)
    graph_before={k:v.detach().clone() for k,v in model.graph.state_dict().items()}
    old_weights_before=sha_file(oldmodel)
    normmean=origweights["normalizer.mean"].float()
    normscale=origweights["normalizer.scale"].float()
    oldn=normalize(orig[a][:TRAIN],normmean,normscale,orig[m][:TRAIN])
    newn=normalize(new[a],normmean,normscale,new[m])
    osrc,odst=gold_roles(labels_orig,0,TRAIN)
    nsrc,ndst=gold_roles(labels_new,0,TRAIN)
    olddata=(oldn,orig[m][:TRAIN],oldq,osrc,odst)
    newdata=(newn,new[m],newq,nsrc,ndst)
    opt=torch.optim.AdamW(model.query.parameters(),lr=LR,weight_decay=0.02)
    with torch.no_grad():
        before_total,before_old,before_new=loss_fn(model,olddata,newdata)
    trace=[]
    for i in range(STEPS):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss,lo,ln=loss_fn(model,olddata,newdata)
        require(bool(torch.isfinite(loss)),"role loss is not finite")
        loss.backward()
        require(all(p.grad is None for p in model.graph.parameters()),
                "frozen graph received gradients")
        torch.nn.utils.clip_grad_norm_(model.query.parameters(),1.)
        opt.step()
        if i+1 in (1,40,80,160,240):
            trace.append({"step":i+1,"TRAIN_loss":round(float(loss.detach()),8),
                          "original_rehearsal_loss":round(float(lo.detach()),8),
                          "new_reversal_loss":round(float(ln.detach()),8)})
    model.eval()
    with torch.no_grad():
        after_total,after_old,after_new=loss_fn(model,olddata,newdata)
    require(after_total<before_total and torch.isfinite(after_total).item(),
            "TRAIN loss did not decrease")
    require(all(torch.equal(v,model.graph.state_dict()[k]) for k,v in graph_before.items()),
            "original graph changed during role-only remediation")
    require(sha_file(oldmodel)==old_weights_before==S3O_SHA,
            "original frozen S3O checkpoint mutated")

    # Physical FP16 commit before inspecting original S3M DEV64.
    out.mkdir(parents=True,exist_ok=False)
    model_path=out/"NRA2_S3Z_ROLE_REVERSAL_REHEARSAL256_FP16.safetensors"
    packed={"bridge."+k:v.detach().cpu().half().contiguous()
            for k,v in model.state_dict().items()}
    packed["normalizer.mean"]=normmean.half().contiguous()
    packed["normalizer.scale"]=normscale.half().contiguous()
    save_file(packed,str(model_path))
    physical=load_file(str(model_path),device="cpu")
    require(set(physical)==set(packed)
            and all(torch.equal(physical[k],v) for k,v in packed.items()),
            "new model physical FP16 reload mismatch")
    new_sha=sha_file(model_path)
    reloaded=S3KGraphWithLearnedRoles().eval()
    reloaded.load_state_dict({k.removeprefix("bridge."):v for k,v in physical.items()
                              if k.startswith("bridge.")},strict=True)
    reloaded.requires_grad_(False)
    for k,v in origweights.items():
        if k.startswith("bridge.graph."):
            require(torch.equal(physical[k],v),"S3O graph FP16 weights not bitwise conserved")
    rereadmean=physical["normalizer.mean"].float()
    rereadscale=physical["normalizer.scale"].float()
    trainold=(normalize(orig[a][:TRAIN],rereadmean,rereadscale,orig[m][:TRAIN]),
              orig[m][:TRAIN],oldq,osrc,odst)
    trainnew=(normalize(new[a],rereadmean,rereadscale,new[m]),
              new[m],newq,nsrc,ndst)
    trainoldmetrics=role_metrics(reloaded,trainold,oldrows[:TRAIN])
    trainnewmetrics=role_metrics(reloaded,trainnew,sxrows)

    # One post-physical-commit exploratory DEV64. No tuning or checkpoint re-selection.
    devq,_=bind(tokenizer,[r.question for r in oldrows[TRAIN:]],
                orig[a][TRAIN:],orig[m][TRAIN:],orig[length_key][TRAIN:])
    dsrc,ddst=gold_roles(labels_orig,TRAIN,TRAIN+DEV)
    devdata=(normalize(orig[a][TRAIN:],rereadmean,rereadscale,orig[m][TRAIN:]),
             orig[m][TRAIN:],devq,dsrc,ddst)
    devmetrics=role_metrics(reloaded,devdata,oldrows[TRAIN:])
    require(sha_file(model_path)==new_sha and sha_file(oldmodel)==S3O_SHA,
            "model or original weights changed after DEV scoring")
    report={
        "stage":"NRA2_S3Z_TRAIN_ONLY_ROLE_REVERSAL_REHEARSAL256_FROZEN_GRAPH",
        "S3Y_successful_frozen_cpu_run":S3Y_RUN,
        "S3N_successful_frozen_cpu_run":43522555326,
        "S3X_successful_grammar_prereg_run":43530401603,
        "S3O_original_successful_128_train_run":43522663612,
        "original_616_J7_preserved_independently_in_S3N_and_S3Y":True,
        "original_S3O_physical_fp16_unchanged_sha256":S3O_SHA,
        "new_S3Z_physical_fp16_sha256":new_sha,
        "new_S3Z_physical_fp16_size":model_path.stat().st_size,
        "S3X_pretrained_TRAIN128_grammar_SHA":S3X_SHA,
        "S3N_original_token_SHA":S3N_SHA,
        "S3Y_new_training_token_SHA":S3Y_SHA,
        "original_S3O_graph_FP16_bitwise_equal":True,
        "independent_2241_parameter_NRA_not_modified":True,
        "only_small_query_role_head_trained":True,
        "optimizer_original_S3M_TRAIN128_and_new_S3X_TRAIN128":True,
        "original_S3M_DEV64_never_in_optimizer":True,
        "original_S3T64_never_in_optimizer_and_not_scored_in_this_stage":True,
        "S3T64_already_scored_so_not_sealed_final_blind":True,
        "new_independent_blind_future_test_required":True,
        "fixed_training_seed":SEED,"optimizer_steps":STEPS,"lr":LR,
        "train_count":256,"exploratory_dev_count":DEV,
        "train_loss_before":float(before_total),"train_loss_after":float(after_total),
        "original_rehearsal_loss_before":float(before_old),
        "original_rehearsal_loss_after":float(after_old),
        "new_grammar_loss_before":float(before_new),
        "new_grammar_loss_after":float(after_new),
        "train_trace":trace,
        "original_S3M_TRAIN128_metrics":trainoldmetrics,
        "new_S3X_TRAIN128_metrics":trainnewmetrics,
        "old_S3M_DEV64_exploratory_metrics":devmetrics,
        "candidate_accepted":False,
        "output_gate":0.0,
        "original_full_decoder_new_EOS_gain_tested":False,
        "all_in_physical_bpp_tested":False,
        "paired_CPU_speed_tested":False,
        "LIVE_Render_Supabase_Jessica_APK_QA_TEST_main_unchanged":True,
    }
    (out/"NRA2_S3Z_ROLE_REVERSAL_REHEARSAL256_REPORT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({k:report[k] for k in
          ("stage","train_count","new_S3Z_physical_fp16_sha256",
           "original_S3M_TRAIN128_metrics","new_S3X_TRAIN128_metrics",
           "old_S3M_DEV64_exploratory_metrics","candidate_accepted","output_gate")},
          sort_keys=True),flush=True)

if __name__=="__main__":
    main()
