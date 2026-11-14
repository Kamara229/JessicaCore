"""S3AG: isolate S3Z query-role head training on 512 authorized TRAIN-only rows.

Rehearse S3M original TRAIN128 and S3X reversed TRAIN128; add new S3AE
TRAIN256 CPU-captured in S3AF. S3O/S3Z graph branch and 43M CPU stay frozen.
FUTURE EVAL128 and previously scored S3AA/S3AC tests never run or train here.
No Jessica, QA/TEST, LIVE, Render, Supabase, main, or accepted model changes.
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
from research.nra2_bridge.stage3ae_train256_future_eval128_prereg import (
    build_rows as s3ae_rows, TRAIN as S3AE_TRAIN, EVAL as S3AE_EVAL,
    commitment as s3ae_commitment, REVERSED_STYLES as S3AE_REVERSED,
)
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

SEED=2026111427
STEPS=160
LR=0.0003
TRAIN=128
DEV=64
WIDTH=320
NODES=6
S3O_SHA="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3X_SHA="9628606e8f40d0efc6c7a41858a95fd86abed1ae1663caaba5f1802d75cdd22a"
S3N_SHA="b8a375982e556d67dfd5f741050c18c5460249184da9c3fcebc3d174d80db81b"
S3Y_SHA="4e5491ab5adfaca669c75debf4464fcc83b3465ca44e90ee57e0488dc96ae975"
S3Y_RUN=43530742301
S3AF_RUN=43537851706
S3Z_RUN=43532157094
S3AF_SHA="995c657496f6131daab04735ba8ec673aff4b91c18d1a92721f1179c8e94bc3e"
S3Z_SHA="60a7121b8270f647909776b58dc21b07c6cb885fd6ebeb6cb3158b23221f6e9f"
S3AE_COMMIT_SHA="0ccbb2b19b3feb93b79f5dcbc892f2e414a564b6e4d83e52997098346e4f480f"

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


def role_loss(model, group):
    src,dst=model.query(group[0],group[1],group[2])
    return .5*(F.cross_entropy(src,group[3])+F.cross_entropy(dst,group[4]))

def metrics(model,group,rows,reverse_selector):
    with torch.inference_mode():
        a,b=model.query(group[0],group[1],group[2])
        sa=a.argmax(1)==group[3]
        sb=b.argmax(1)==group[4]
        mask=torch.tensor([reverse_selector(r) for r in rows],dtype=torch.bool)
        def sample(m):
            return {"count":int(m.sum()),"source_correct":int((sa&m).sum()),
                    "target_correct":int((sb&m).sum()),
                    "both_correct":int((sa&sb&m).sum())}
        return {"all":sample(torch.ones(len(rows),dtype=torch.bool)),
                "destination_first":sample(mask),"source_first":sample(~mask)}

def main():
    if len(sys.argv)!=8:
        raise SystemExit("S3AG S3N_CPU192_DIR S3Y_CPU128_DIR S3X_PREREG_DIR S3O_FP16_DIR S3Z_FP16_DIR S3AF_CPU256_DIR NEW_OUT")
    n_dir,y_dir,x_dir,o_dir,z_dir,af_dir,out=map(Path,sys.argv[1:])
    require(not out.exists() or not any(out.iterdir()),"cannot overwrite research artifacts")
    sn=json.loads((n_dir/"NRA2_S3N_192_ROLEBALANCED_REAL_CPU_AUDIT.json").read_text())
    sy=json.loads((y_dir/"NRA2_S3Y_128_REVERSAL_TRAIN_REAL_CPU_AUDIT.json").read_text())
    sx=json.loads((x_dir/"NRA2_S3X_TRAIN128_REVERSE_GRAMMAR_PREREG.json").read_text())
    so=json.loads((o_dir/"NRA2_S3O_NEURAL_SEMANTIC_ROLES_128TRAIN_64DEV_AUDIT.json").read_text())
    sz=json.loads((z_dir/"NRA2_S3Z_ROLE_REVERSAL_REHEARSAL256_REPORT.json").read_text())
    saf=json.loads((af_dir/"NRA2_S3AF_TRAIN256_REAL_CPU_J7_AUDIT.json").read_text())
    sn_tok=n_dir/"NRA2_S3N_REAL_CPU_GRAPH192_ALL_TOKENS_FP16.safetensors"
    sn_lab=n_dir/"NRA2_S3N_GRAPH192_STRUCTURAL_LABELS.safetensors"
    sy_tok=y_dir/"NRA2_S3Y_REAL_CPU_TRAIN128_ALL_TOKENS_FP16.safetensors"
    sy_lab=y_dir/"NRA2_S3Y_TRAIN128_STRUCTURAL_LABELS.safetensors"
    af_tok=af_dir/"NRA2_S3AF_REAL_CPU_TRAIN256_ALL_TOKENS_FP16.safetensors"
    af_lab=af_dir/"NRA2_S3AF_TRAIN256_STRUCTURAL_LABELS.safetensors"
    orig=o_dir/"NRA2_S3O_NATIVE_CPU_LEARNED_ROLES_TRAIN128_FP16.safetensors"
    parent=z_dir/"NRA2_S3Z_ROLE_REVERSAL_REHEARSAL256_FP16.safetensors"
    require(sx==s3x_commitment(s3x_rows()) and sx["commitment_sha256"]==S3X_SHA,
            "pinned S3X reversal TRAIN128 prereg drift")
    require(sn["all_616_token_digest_matches_S194"] and sy["all_616_token_digest_matches_S194"]
            and saf["all_616_token_digest_matches_S194"],"original S194 616 J7 is not verified")
    require(sn["full_token_fp16_sha256"]==S3N_SHA==sha_file(sn_tok)
            and sy["full_token_fp16_sha256"]==S3Y_SHA==sha_file(sy_tok)
            and saf["full_token_fp16_sha256"]==S3AF_SHA==sha_file(af_tok),
            "original native 43M CPU token physical SHA drift")
    require(saf["S3AE_registered_commitment_sha256"]==S3AE_COMMIT_SHA
            and saf["S3AE_future_EVAL128_never_prefilled_scored_or_trained"] is True
            and saf["graph_prefill_count"]==256 and saf["train_count"]==256
            and saf["group_sizes"]=={S3AE_TRAIN:256}
            and saf["adapter_trained"] is False and saf["nra_output_gate"]==0.0
            and saf["physical_reload_bitwise_equal"] is True,
            "S3AF TRAIN256 prereg/physical/independent EVAL boundary drift")
    require(sy["S3X_precommitted_TRAIN128_sha256"]==S3X_SHA and sy["train_count"]==128
            and sy["physical_reload_bitwise_equal"] and sy["nra_output_gate"]==0.0,
            "S3Y source TRAIN128 guards drift")
    require(sn["group_sizes"]=={"TRAIN128":128,"DEV64":64}
            and sn["physical_reload_bitwise_equal"] and sn["nra_output_gate"]==0.0,
            "S3N original TRAIN/DEV boundary drift")
    require(so["FP16_checkpoint_sha256"]==S3O_SHA==sha_file(orig)
            and sz["new_S3Z_physical_fp16_sha256"]==S3Z_SHA==sha_file(parent)
            and sz["original_S3O_physical_fp16_unchanged_sha256"]==S3O_SHA
            and sz["original_S3O_graph_FP16_bitwise_equal"] is True
            and sz["candidate_accepted"] is False and sz["output_gate"]==0.0,
            "pinned S3O/S3Z graph and trained parent SHA drift")
    rows0=s3m_rows()[:128]
    rows1=s3x_rows()
    allrows=s3ae_rows()
    rows2=[r for r in allrows if r.split==S3AE_TRAIN]
    require(len(rows0)==len(rows1)==128 and len(rows2)==256 and
            len([r for r in allrows if r.split==S3AE_EVAL])==128,
            "authorized TRAIN512 + distinct untouched future EVAL128")
    require(saf["S3AE_TRAIN256_graph_topologies"]==64
            and saf["S3AE_TRAIN256_source_first"]==128
            and saf["S3AE_TRAIN256_destination_first"]==128,
            "S3AE balanced question order drift")
    require(len({r.prompt_sha256 for r in rows0+rows1+rows2})==512,
            "TRAIN512 prompt SHA overlap")
    require(sn["split_groups"]["TRAIN128"]["prompt_sha256"]==[r.prompt_sha256 for r in rows0]
            and sy["split_groups"][sx["group"]]["prompt_sha256"]==[r.prompt_sha256 for r in rows1],
            "S3N/S3Y native CPU prompt order drift")
    # Only train rows2 are ever converted to token features. Future EVAL128
    # stays a preregistered commitment; no model sees its text, gold or features.
    require(saf["S3AE_TRAIN256_source_sha256"]==
            hashlib.sha256(json.dumps([r.prompt_sha256 for r in rows2],
              sort_keys=True,separators=(",",":")).encode()).hexdigest(),
            "S3AF TRAIN256 source SHA/order drift")
    toks=[load_file(str(f),device="cpu") for f in (sn_tok,sy_tok,af_tok)]
    labs=[load_file(str(f),device="cpu") for f in (sn_lab,sy_lab,af_lab)]
    tokenizer=exact_original_tokenizer()
    names="token_hidden320","token_mask","token_lengths"
    parent_weights=load_file(str(parent),device="cpu")
    orig_weights=load_file(str(orig),device="cpu")
    for k,v in orig_weights.items():
        if k.startswith("bridge.graph."):
            require(torch.equal(v,parent_weights[k]),"S3Z vs S3O frozen graph bitwise drift")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model=S3KGraphWithLearnedRoles()
    model.load_state_dict({k.removeprefix("bridge."):v for k,v in parent_weights.items()
                           if k.startswith("bridge.")},strict=True)
    model.graph.requires_grad_(False)
    graph_before={k:v.detach().clone() for k,v in model.graph.state_dict().items()}
    mean=parent_weights["normalizer.mean"].float()
    scale=parent_weights["normalizer.scale"].float()
    inputs=[]
    for rows,t,g in zip((rows0,rows1,rows2),toks,labs):
        n=len(rows)
        require(set(g)=={"adjacency","source","target","reachability"}
                and g["source"].shape[1]==NODES and g["target"].shape[1]==NODES,
                "physical role gold ABI drift")
        tt=t[names[0]][:n]; tm=t[names[1]][:n]; tl=t[names[2]][:n]
        q,_binding=bind(tokenizer,[r.question for r in rows],tt,tm,tl)
        src,dst=gold_roles(g,0,n)
        require(bool(torch.all(q.node_order.sort(1).values ==
                    torch.stack([src,dst],dim=-1).sort(1).values)),
                "physical source/target not among literal two query entities")
        inputs.append((normalize(tt,mean,scale,tm),tm,q,src,dst))
    # Freeze original 43M, original S3O graph and independent NRA.
    # Fixed optimizer: only learned query-role head, with old/reversal rehearsal.
    optimizer=torch.optim.AdamW(model.query.parameters(),lr=LR,weight_decay=0.02)
    def aggregate():
        ls=[role_loss(model,v) for v in inputs]
        return ls[0]+ls[1]+2*ls[2],ls
    with torch.no_grad():
        before,per_before=aggregate()
    trace=[]
    for step in range(STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss,parts=aggregate()
        require(bool(torch.isfinite(loss)),"nonfinite role-head loss")
        loss.backward()
        require(all(p.grad is None for p in model.graph.parameters()),
                "frozen S3O graph received gradients")
        torch.nn.utils.clip_grad_norm_(model.query.parameters(),1.0)
        optimizer.step()
        if step+1 in (1,40,80,120,160):
            trace.append({"step":step+1,
                          "original_TRAIN128_loss":float(parts[0].detach()),
                          "reversal_TRAIN128_loss":float(parts[1].detach()),
                          "new_S3AE_TRAIN256_loss":float(parts[2].detach())})
    model.eval()
    with torch.no_grad():
        after,per_after=aggregate()
    require(bool(torch.isfinite(after)) and after<before,
            "TRAIN512 total role loss did not decrease")
    require(all(torch.equal(v,model.graph.state_dict()[k])
                for k,v in graph_before.items()),"graph branch changed during role training")
    require(sha_file(parent)==S3Z_SHA and sha_file(orig)==S3O_SHA,
            "parent S3Z or original S3O physical weights mutated")
    # Commit FP16 BEFORE calculating even TRAIN-only accuracy; never score EVAL128.
    out.mkdir(parents=True,exist_ok=False)
    model_file=out/"NRA2_S3AG_FROZEN_GRAPH_ROLE512_TRAIN_FP16.safetensors"
    packed={"bridge."+k:v.detach().cpu().half().contiguous()
            for k,v in model.state_dict().items()}
    packed["normalizer.mean"]=mean.half().contiguous()
    packed["normalizer.scale"]=scale.half().contiguous()
    save_file(packed,str(model_file))
    reloaded_data=load_file(str(model_file),device="cpu")
    require(set(reloaded_data)==set(packed)
            and all(torch.equal(reloaded_data[k],v) for k,v in packed.items()),
            "S3AG FP16 physical reload mismatch")
    for k,v in orig_weights.items():
        if k.startswith("bridge.graph."):
            require(torch.equal(v,reloaded_data[k]),
                    "S3O graph FP16 not identical after S3AG physical save")
    new_fp16_sha=sha_file(model_file)
    loaded=S3KGraphWithLearnedRoles().eval()
    loaded.load_state_dict({k.removeprefix("bridge."):v for k,v in reloaded_data.items()
                            if k.startswith("bridge.")},strict=True)
    loaded.requires_grad_(False)
    results=[metrics(loaded,data,rows,fn) for data,rows,fn in
             ((inputs[0],rows0,lambda r:False),
              (inputs[1],rows1,lambda r:r.query_style in (0,2)),
              (inputs[2],rows2,lambda r:r.query_style in S3AE_REVERSED))]
    require(sha_file(model_file)==new_fp16_sha and sha_file(parent)==S3Z_SHA
            and sha_file(orig)==S3O_SHA,"physical SHA changed during TRAIN scoring")
    report={
        "stage":"NRA2_S3AG_TRAIN512_FROZEN_S3O_GRAPH_S3Z_ROLE_ONLY",
        "S3AF_real_cpu_successful_run":S3AF_RUN,
        "S3Z_parent_training_successful_run":S3Z_RUN,
        "S3AE_preregistered_commitment_sha256":S3AE_COMMIT_SHA,
        "new_S3AE_cpu_token_fp16_unchanged_sha256":S3AF_SHA,
        "S3Z_parent_fp16_unchanged_sha256":S3Z_SHA,
        "S3O_original_fp16_unchanged_sha256":S3O_SHA,
        "new_S3AG_physical_fp16_sha256":new_fp16_sha,
        "new_S3AG_physical_fp16_size":model_file.stat().st_size,
        "physical_fp16_reload_equal":True,
        "original_graph_physical_fp16_bitwise_equal":True,
        "full_S194_original_616_J7_digest_preserved_in_S3N_S3Y_S3AF":True,
        "train_count":512,"new_S3AE_TRAIN_count":256,
        "rehearsal_S3M_TRAIN_count":128,"rehearsal_S3X_TRAIN_count":128,
        "original_S3M_DEV64_not_scored_or_trained":True,
        "S3AA_S3AC_already_scored_not_reused_as_blind":True,
        "future_independent_S3AE_EVAL128_never_prefilled_scored_or_trained":True,
        "only_query_role_head_trained":True,
        "original_graph_and_independent_NRA_frozen":True,
        "optimizer_steps":STEPS,"seed":SEED,"lr":LR,
        "train_loss_before":float(before),"train_loss_after":float(after),
        "rehearsal_S3M_loss_before":float(per_before[0]),
        "rehearsal_S3M_loss_after":float(per_after[0]),
        "rehearsal_S3X_loss_before":float(per_before[1]),
        "rehearsal_S3X_loss_after":float(per_after[1]),
        "new_S3AE_loss_before":float(per_before[2]),
        "new_S3AE_loss_after":float(per_after[2]),
        "train_trace":trace,"TRAIN_only_role_metrics":{
            "S3M_original128":results[0],"S3X_reversed128":results[1],
            "S3AE_new256":results[2]},
        "candidate_accepted":False,"output_gate":0.0,
        "full_original_CPU_J7_candidate_gain_tested":False,
        "all_in_physical_bpp_tested":False,"paired_candidate_CPU_speed_tested":False,
        "Jessica_LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    (out/"NRA2_S3AG_TRAIN512_ROLE_HEAD_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in (
         "stage","train_count","new_S3AG_physical_fp16_sha256",
         "train_loss_before","train_loss_after","TRAIN_only_role_metrics",
         "candidate_accepted","output_gate")},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
