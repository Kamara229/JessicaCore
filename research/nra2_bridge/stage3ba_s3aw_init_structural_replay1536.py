"""S3AW: S3AS-initialized semantic role rehearsal S3AU new 512, DEV128 post-lock.

Rehearse S3M original TRAIN128 and S3X reversed TRAIN128, initialize from
physical S3AS; S3O graph, true 43M, Green Tunnel and independent NRA frozen.
Four prompts per topology enforce same-role paraphrase and reversed-role symmetry.
New S3AQ future EVAL128 never prefilled, scored or trained. Prior scored S3AI
EVAL128 never used. Original protected 616 J7 unchanged; gate zero.
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
from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows as s3aq_rows, TRAIN as S3AQ_TRAIN, DEV as S3AQ_DEV,
    EVAL as S3AQ_EVAL, REVERSED_STYLES as S3AQ_REVERSED,
)
from research.nra2_bridge.stage3ay_structural_train512_prereg import (
    build_rows as s3ay_rows, commitment as s3ay_commitment,
    SPLIT as S3AY_TRAIN, DESTINATION_FIRST as S3AY_REVERSED,
)
from research.nra2_bridge.stage3au_train512_paraphrase_prereg import (
    build_rows as s3au_rows, commitment as s3au_commitment,
    SPLIT as S3AU_TRAIN, DESTINATION_FIRST as S3AU_REVERSED,
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

SEED=2026111496
STEPS=160
LR=0.00005
SYMMETRY_WEIGHT=0.15
S3AU_SYMMETRY_WEIGHT=0.15
S3AY_SYMMETRY_WEIGHT=0.15
TRAIN=128
DEV=64
WIDTH=320
NODES=6
S3O_SHA="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3X_SHA="9628606e8f40d0efc6c7a41858a95fd86abed1ae1663caaba5f1802d75cdd22a"
S3N_SHA="b8a375982e556d67dfd5f741050c18c5460249184da9c3fcebc3d174d80db81b"
S3Y_SHA="4e5491ab5adfaca669c75debf4464fcc83b3465ca44e90ee57e0488dc96ae975"
S3Y_RUN=43530742301
S3AR_RUN=43580373587
S3AS_RUN=43581666491
S3AV_RUN=43582744317
S3AU_RUN=43582403331
S3AR_SHA="5c44f4d52b5f2f618b93885f288975e269a4d44d21506cd7906368fd20467f2b"
S3AS_SHA="06420e22270dd082116086a5174e95c04e9ff2b52664583bc8d5a1b4a074d488"
S3AV_SHA="b48b74404009d67f24ba40830310c0e5aea95c47ef043bd8e74955d014c511d5"
S3AU_SHA="ac038f50dab0ace449f1df32ec7e8728476420615f60c85a8d8fce64c787558a"
S3AQ_COMMIT_SHA="c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839"
S3AW_RUN=43583928298
S3AW_SHA="8541eea24d5074e29b6a4acd8b611d2d29d608dacf2dfe909e8df05eb0fb2130"
S3AZ_RUN=43587917560
S3AZ_SHA="978cb86c1e30c211744a800b8c37a7e20dc04e2b7ee85126dea3e91d7894feec"
S3AY_RUN=43587711528
S3AY_SHA="90ba2d0347a333dc9ff2e85adbbaca20becb4f92051c1ec43267102b63c8a118"

def require(cond,msg):
    if not cond:
        raise AssertionError("S3BA "+msg)

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

def role_symmetry_loss(model,group,rows):
    """Only new TRAIN: same graph four styles; equivalence and reversed-role constraint."""
    require(len(rows)==256 and all(rows[k].graph_id==rows[k+i].graph_id
            and rows[k+i].query_style==i for k in range(0,256,4) for i in range(4)),
            "S3AQ grouped four-style TRAIN quartet layout drift")
    require(all(rows[k].src==rows[k+2].src and rows[k].dst==rows[k+2].dst
            and rows[k+1].src==rows[k+3].src and rows[k+1].dst==rows[k+3].dst
            and rows[k].src==rows[k+1].dst and rows[k].dst==rows[k+1].src
            for k in range(0,256,4)), "quartet role reversal gold topology drift")
    src,dst=model.query(group[0],group[1],group[2])
    ps,pd=F.softmax(src,dim=-1),F.softmax(dst,dim=-1)
    terms=(
       F.mse_loss(ps[0::4],ps[2::4]),
       F.mse_loss(pd[0::4],pd[2::4]),
       F.mse_loss(ps[1::4],ps[3::4]),
       F.mse_loss(pd[1::4],pd[3::4]),
       F.mse_loss(ps[0::4],pd[1::4]),
       F.mse_loss(pd[0::4],ps[1::4]),
       F.mse_loss(ps[2::4],pd[3::4]),
       F.mse_loss(pd[2::4],ps[3::4]),
    )
    return sum(terms)/len(terms)

def octet_symmetry_loss(model,group,rows):
    """Only 512 fresh TRAIN samples: four reversed pairs and paraphrase agreement."""
    require(len(rows)==512 and all(rows[k].graph_id==rows[k+j].graph_id
            and rows[k+j].query_style==j for k in range(0,512,8) for j in range(8)),
            "S3AU TRAIN octet order drift")
    require(all(rows[k+i].src==rows[k+i+2].src and
                rows[k+i].dst==rows[k+i+2].dst
                for k in range(0,512,8) for i in range(6)),
            "S3AU same-role paraphrase gold drift")
    require(all(rows[k+i].src==rows[k+i+1].dst and
                rows[k+i].dst==rows[k+i+1].src
                for k in range(0,512,8) for i in (0,2,4,6)),
            "S3AU source/destination reversal pairing drift")
    a,b=model.query(group[0],group[1],group[2])
    p,q=F.softmax(a,dim=-1),F.softmax(b,dim=-1)
    terms=[]
    for i in (0,2,4,6):
        terms.append(F.mse_loss(p[i::8],q[i+1::8]))
        terms.append(F.mse_loss(q[i::8],p[i+1::8]))
    for i in (2,4,6):
        terms.append(F.mse_loss(p[0::8],p[i::8]))
        terms.append(F.mse_loss(q[0::8],q[i::8]))
    for i in (3,5,7):
        terms.append(F.mse_loss(p[1::8],p[i::8]))
        terms.append(F.mse_loss(q[1::8],q[i::8]))
    return sum(terms)/len(terms)


def main():
    if len(sys.argv)!=13:
        raise SystemExit("S3BA S3N S3Y S3X S3O S3AS S3AR S3AV S3AU S3AW S3AZ S3AY OUT")
    n_dir,y_dir,x_dir,o_dir,z_dir,af_dir,av_dir,au_dir,aw_dir,az_dir,ay_dir,out=map(Path,sys.argv[1:])
    require(not out.exists() or not any(out.iterdir()),"cannot overwrite research artifacts")
    sn=json.loads((n_dir/"NRA2_S3N_192_ROLEBALANCED_REAL_CPU_AUDIT.json").read_text())
    sy=json.loads((y_dir/"NRA2_S3Y_128_REVERSAL_TRAIN_REAL_CPU_AUDIT.json").read_text())
    sx=json.loads((x_dir/"NRA2_S3X_TRAIN128_REVERSE_GRAMMAR_PREREG.json").read_text())
    so=json.loads((o_dir/"NRA2_S3O_NEURAL_SEMANTIC_ROLES_128TRAIN_64DEV_AUDIT.json").read_text())
    sz=json.loads((z_dir/"NRA2_S3AS_S3Z_INIT_TRAIN512_DEV128_ROLE_AUDIT.json").read_text())
    saf=json.loads((af_dir/"NRA2_S3AR_TRAIN256_DEV128_REAL_CPU_J7_AUDIT.json").read_text())
    sav=json.loads((av_dir/"NRA2_S3AV_TRAIN512_REAL_CPU_J7_AUDIT.json").read_text())
    sau=json.loads((au_dir/"NRA2_S3AU_NEW_TRAIN512_PARAPHRASE_PREREG.json").read_text())
    saw=json.loads((aw_dir/"NRA2_S3AW_S3AS_INIT_TRAIN1024_DEV128_ROLE_AUDIT.json").read_text())
    saz=json.loads((az_dir/"NRA2_S3AZ_TRAIN512_REAL_CPU_J7_AUDIT.json").read_text())
    say=json.loads((ay_dir/"NRA2_S3AY_NEW_TRAIN512_PARAPHRASE_PREREG.json").read_text())
    sn_tok=n_dir/"NRA2_S3N_REAL_CPU_GRAPH192_ALL_TOKENS_FP16.safetensors"
    sn_lab=n_dir/"NRA2_S3N_GRAPH192_STRUCTURAL_LABELS.safetensors"
    sy_tok=y_dir/"NRA2_S3Y_REAL_CPU_TRAIN128_ALL_TOKENS_FP16.safetensors"
    sy_lab=y_dir/"NRA2_S3Y_TRAIN128_STRUCTURAL_LABELS.safetensors"
    af_tok=af_dir/"NRA2_S3AR_REAL_CPU_TRAIN256_DEV128_ALL_TOKENS_FP16.safetensors"
    af_lab=af_dir/"NRA2_S3AR_TRAIN256_DEV128_STRUCTURAL_LABELS.safetensors"
    av_tok=av_dir/"NRA2_S3AV_REAL_CPU_TRAIN512_ALL_TOKENS_FP16.safetensors"
    av_lab=av_dir/"NRA2_S3AV_TRAIN512_STRUCTURAL_LABELS.safetensors"
    az_tok=az_dir/"NRA2_S3AZ_REAL_CPU_TRAIN512_ALL_TOKENS_FP16.safetensors"
    az_lab=az_dir/"NRA2_S3AZ_TRAIN512_STRUCTURAL_LABELS.safetensors"
    orig=o_dir/"NRA2_S3O_NATIVE_CPU_LEARNED_ROLES_TRAIN128_FP16.safetensors"
    s3as_parent=z_dir/"NRA2_S3AS_S3Z_INIT_ROLE512_FP16.safetensors"
    parent=aw_dir/"NRA2_S3AW_S3AS_INIT_ROLE1024_FP16.safetensors"
    require(sx==s3x_commitment(s3x_rows()) and sx["commitment_sha256"]==S3X_SHA,
            "pinned S3X reversal TRAIN128 prereg drift")
    require(sn["all_616_token_digest_matches_S194"] and sy["all_616_token_digest_matches_S194"]
            and saf["all_616_token_digest_matches_S194"] and sav["all_616_token_digest_matches_S194"]
            and saz["all_616_token_digest_matches_S194"],"original S194 616 J7 is not verified")
    require(sn["full_token_fp16_sha256"]==S3N_SHA==sha_file(sn_tok)
            and sy["full_token_fp16_sha256"]==S3Y_SHA==sha_file(sy_tok)
            and saf["full_token_fp16_sha256"]==S3AR_SHA==sha_file(af_tok)
            and sav["full_token_fp16_sha256"]==S3AV_SHA==sha_file(av_tok)
            and saz["full_token_fp16_sha256"]==S3AZ_SHA==sha_file(az_tok),
            "original native 43M CPU token physical SHA drift")
    require(saf["S3AQ_registered_commitment_sha256"]==S3AQ_COMMIT_SHA
            and saf["S3AQ_future_EVAL128_never_prefilled_scored_or_trained"] is True
            and saf["graph_prefill_count"]==384 and saf["train_count"]==256
            and saf["dev_count"]==128
            and saf["group_sizes"]=={S3AQ_TRAIN:256,S3AQ_DEV:128}
            and saf["adapter_trained"] is False and saf["nra_output_gate"]==0.0
            and saf["physical_reload_bitwise_equal"] is True,
            "S3AR TRAIN256 DEV128 prereg/physical/future EVAL boundary drift")
    require(sy["S3X_precommitted_TRAIN128_sha256"]==S3X_SHA and sy["train_count"]==128
            and sy["physical_reload_bitwise_equal"] and sy["nra_output_gate"]==0.0,
            "S3Y source TRAIN128 guards drift")
    require(sn["group_sizes"]=={"TRAIN128":128,"DEV64":64}
            and sn["physical_reload_bitwise_equal"] and sn["nra_output_gate"]==0.0,
            "S3N original TRAIN/DEV boundary drift")
    require(so["FP16_checkpoint_sha256"]==S3O_SHA==sha_file(orig)
            and sz["new_S3AS_physical_fp16_sha256"]==S3AS_SHA==sha_file(s3as_parent)
            and sz["S3O_original_fp16_unchanged_sha256"]==S3O_SHA
            and sz["original_graph_physical_fp16_bitwise_equal"] is True
            and sz["candidate_accepted"] is False and sz["output_gate"]==0.0,
            "pinned original S3O graph and S3AS role teacher SHA drift")
    require(saw["new_S3AW_physical_fp16_sha256"]==S3AW_SHA==sha_file(parent)
            and saw["S3AS_parent_fp16_sha256"]==S3AS_SHA
            and saw["train_count"]==1024 and saw["DEV_count"]==128
            and saw["DEV128_paired_S3AW_candidate"]["all"]["both_correct"]==89
            and saw["DEV128_scored_only_after_FP16_SHA_commit"]
            and saw["original_graph_physical_fp16_bitwise_equal"]
            and saw["full_S194_original_616_J7_digests_from_S3AV_protected"]
            and saw["output_gate"]==0.0 and not saw["candidate_accepted"]
            and saw["S3AQ_future_EVAL128_not_prefilled_scored_or_trained"],
            "S3AW physical head, 1024 rehearsal or DEV baseline changed")
    newrows=s3au_rows()
    require(sau==s3au_commitment(newrows)
            and sau["commitment_sha256"]==S3AU_SHA
            and sau["no_new_model_weights_or_training_or_scores"] is True
            and sau["future_EVAL128_never_prefilled_scored_or_trained"] is True
            and sav["stage"]=="NRA2_S3AV_FROZEN_S3AU_NEW_TRAIN512_NO_SCORE_CPU_CAPTURE_ONLY"
            and sav["S3AU_registered_commitment_sha256"]==S3AU_SHA
            and sav["graph_prefill_count"]==512 and sav["train_count"]==512
            and sav["group_sizes"]=={S3AU_TRAIN:512}
            and sav["physical_reload_bitwise_equal"] is True
            and sav["nra_output_gate"]==0.0 and sav["adapter_trained"] is False,
            "S3AU TRAIN512 CPU native features or prereg provenance drift")
    require(sav["physical_graph_labels_sha256"]==sha_file(av_lab)
            and sav["S3AU_future_EVAL128_never_prefilled_scored_or_trained"] is True
            and sav["S3AU_scored_DEV128_not_prefilled_or_used_in_optimizer"] is True
            and sav["S3AU_new_wording_count"]==8
            and sav["S3AU_train_source_first"]==256
            and sav["S3AU_train_destination_first"]==256,
            "new TRAIN512 original CPU physical labels or future EVAL protection drift")
    newerrows=s3ay_rows()
    require(say==s3ay_commitment(newerrows)
            and say["commitment_sha256"]==S3AY_SHA
            and say["S3AW_physical_query_role_sha256"]==S3AW_SHA
            and say["future_EVAL128_never_prefilled_scored_or_trained"]
            and say["no_new_model_weights_or_training_or_scores"]
            and saz["stage"]=="NRA2_S3AZ_FROZEN_S3AY_NEW_TRAIN512_NO_SCORE_CPU_CAPTURE_ONLY"
            and saz["S3AY_registered_commitment_sha256"]==S3AY_SHA
            and saz["S3AY_successful_prereg_run"]==S3AY_RUN
            and saz["S3AY_future_EVAL128_never_prefilled_scored_or_trained"]
            and saz["S3AY_scored_DEV128_not_prefilled_or_used_in_optimizer"]
            and saz["graph_prefill_count"]==512 and saz["train_count"]==512
            and saz["group_sizes"]=={S3AY_TRAIN:512}
            and saz["nra_output_gate"]==0.0 and not saz["adapter_trained"]
            and saz["physical_reload_bitwise_equal"]
            and saz["physical_graph_labels_sha256"]==sha_file(az_lab),
            "S3AY prereg or S3AZ native CPU/J7/no-leak source drift")
    rows0=s3m_rows()[:128]
    rows1=s3x_rows()
    allrows=s3aq_rows()
    rows2=[r for r in allrows if r.split==S3AQ_TRAIN]
    devrows=[r for r in allrows if r.split==S3AQ_DEV]
    rows3=newrows
    rows4=newerrows
    require(len(rows0)==len(rows1)==128 and len(rows2)==256 and len(rows3)==len(rows4)==512 and len(devrows)==128
            and len([r for r in allrows if r.split==S3AQ_EVAL])==128,
            "TRAIN512 DEV128 and untouched future EVAL128")
    require(saf["S3AQ_train_graph_topologies"]==64
            and saf["S3AQ_train_destination_first"]==128
            and saf["S3AQ_dev_graph_topologies"]==32
            and saf["S3AQ_dev_destination_first"]==64,
            "S3AQ graph topology and role balance drift")
    require(len({r.prompt_sha256 for r in rows0+rows1+rows2+rows3+rows4+devrows})==1664,
            "TRAIN512 DEV128 prompt SHA overlap")
    require(sn["split_groups"]["TRAIN128"]["prompt_sha256"]==[r.prompt_sha256 for r in rows0]
            and sy["split_groups"][sx["group"]]["prompt_sha256"]==[r.prompt_sha256 for r in rows1],
            "S3N/S3Y CPU prompt order drift")
    def split_sha(rows):
        return hashlib.sha256(json.dumps([r.prompt_sha256 for r in rows],
            sort_keys=True,separators=(",",":")).encode()).hexdigest()
    require(saf["S3AQ_TRAIN256_source_sha256"]==split_sha(rows2)
            and saf["S3AQ_DEV128_source_sha256"]==split_sha(devrows),
            "S3AR TRAIN256 / DEV128 original CPU prompt SHA/order drift")
    toks=[load_file(str(f),device="cpu") for f in (sn_tok,sy_tok,af_tok,av_tok,az_tok)]
    labs=[load_file(str(f),device="cpu") for f in (sn_lab,sy_lab,af_lab,av_lab,az_lab)]
    tokenizer=exact_original_tokenizer()
    names="token_hidden320","token_mask","token_lengths"
    parent_weights=load_file(str(parent),device="cpu")
    orig_weights=load_file(str(orig),device="cpu")
    for k,v in orig_weights.items():
        if k.startswith("bridge.graph."):
            require(torch.equal(v,parent_weights[k]),"S3AS vs S3O frozen graph bitwise drift")
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
    for rows,t,g in zip((rows0,rows1,rows2,rows3,rows4),toks,labs):
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
    # Optimizer only role head; 1024 authorized TRAIN rows, no DEV/EVAL.
    optimizer=torch.optim.AdamW(model.query.parameters(),lr=LR,weight_decay=0.02)
    def aggregate():
        ls=[role_loss(model,v) for v in inputs]
        sym=role_symmetry_loss(model,inputs[2],rows2)
        novel_sym=octet_symmetry_loss(model,inputs[3],rows3)
        return (2*ls[0]+2*ls[1]+2*ls[2]+ls[3]+
                SYMMETRY_WEIGHT*sym+S3AU_SYMMETRY_WEIGHT*novel_sym),ls,sym,novel_sym
    with torch.no_grad():
        before,per_before,sym_before,novel_sym_before=aggregate()
    trace=[]
    for step in range(STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss,parts,sym,novel_sym=aggregate()
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
                          "new_S3AQ_TRAIN256_loss":float(parts[2].detach()),
                          "new_S3AU_TRAIN512_loss":float(parts[3].detach()),
                          "new_TRAIN_only_role_symmetry_loss":float(sym.detach()),
                          "S3AU_TRAIN512_octet_symmetry_loss":float(novel_sym.detach())})
    model.eval()
    with torch.no_grad():
        after,per_after,sym_after,novel_sym_after=aggregate()
    require(bool(torch.isfinite(after)) and after<before,
            "TRAIN512 total role loss did not decrease")
    require(all(torch.equal(v,model.graph.state_dict()[k])
                for k,v in graph_before.items()),"graph branch changed during role training")
    require(sha_file(parent)==S3AS_SHA and sha_file(orig)==S3O_SHA,
            "parent S3AS or original S3O physical weights mutated")
    # Commit FP16 BEFORE calculating even TRAIN-only accuracy; never score EVAL128.
    out.mkdir(parents=True,exist_ok=False)
    model_file=out/"NRA2_S3AW_S3AS_INIT_ROLE1024_FP16.safetensors"
    packed={"bridge."+k:v.detach().cpu().half().contiguous()
            for k,v in model.state_dict().items()}
    packed["normalizer.mean"]=mean.half().contiguous()
    packed["normalizer.scale"]=scale.half().contiguous()
    save_file(packed,str(model_file))
    reloaded_data=load_file(str(model_file),device="cpu")
    require(set(reloaded_data)==set(packed)
            and all(torch.equal(reloaded_data[k],v) for k,v in packed.items()),
            "S3AW FP16 physical reload mismatch")
    for k,v in orig_weights.items():
        if k.startswith("bridge.graph."):
            require(torch.equal(v,reloaded_data[k]),
                    "S3O graph FP16 not identical after S3AW physical save")
    new_fp16_sha=sha_file(model_file)
    loaded=S3KGraphWithLearnedRoles().eval()
    loaded.load_state_dict({k.removeprefix("bridge."):v for k,v in reloaded_data.items()
                            if k.startswith("bridge.")},strict=True)
    loaded.requires_grad_(False)
    # DEV128 must only be opened after physically committing FP16 checkpoint.
    dev_t=toks[2]
    dev_g=labs[2]
    dt=dev_t["token_hidden320"][256:384]
    dm=dev_t["token_mask"][256:384]
    dl=dev_t["token_lengths"][256:384]
    dq,_binding=bind(tokenizer,[r.question for r in devrows],dt,dm,dl)
    ds,dd=gold_roles(dev_g,256,384)
    require(torch.all(dq.node_order.sort(1).values ==
            torch.stack([ds,dd],dim=-1).sort(1).values),
            "DEV128 query source/destination physical alignment drift")
    dev_group=(normalize(dt,mean,scale,dm),dm,dq,ds,dd)
    prior=S3KGraphWithLearnedRoles().eval()
    prior.load_state_dict({k.removeprefix("bridge."):v for k,v in parent_weights.items()
                           if k.startswith("bridge.")},strict=True)
    prior.requires_grad_(False)
    prior_dev=metrics(prior,dev_group,devrows,lambda r:r.query_style in S3AQ_REVERSED)
    after_dev=metrics(loaded,dev_group,devrows,lambda r:r.query_style in S3AQ_REVERSED)
    train_results=[metrics(loaded,data,rows,fn) for data,rows,fn in
             ((inputs[0],rows0,lambda r:False),
              (inputs[1],rows1,lambda r:r.query_style in (0,2)),
              (inputs[2],rows2,lambda r:r.query_style in S3AQ_REVERSED),
              (inputs[3],rows3,lambda r:r.query_style in S3AU_REVERSED))]
    dev_source_first_gain=(after_dev["source_first"]["both_correct"]
                           -prior_dev["source_first"]["both_correct"])
    dev_dest_first_gain=(after_dev["destination_first"]["both_correct"]
                         -prior_dev["destination_first"]["both_correct"])
    require(sha_file(model_file)==new_fp16_sha and sha_file(parent)==S3AS_SHA
            and sha_file(orig)==S3O_SHA,
            "parent or physical role-head SHA changed during DEV score")
    report={
        "stage":"NRA2_S3AW_S3AS_INIT_ROLE1024_TRAIN_S3AU512_DEV128_SCORED",
        "S3AS_locked_parent_run":S3AS_RUN,"S3AV_native_CPU_run":S3AV_RUN,
        "S3AU_prereg_run":S3AU_RUN,
        "S3AS_parent_fp16_sha256":S3AS_SHA,
        "S3AV_real_CPU_token_sha256":S3AV_SHA,
        "S3AU_preregistered_sha256":S3AU_SHA,
        "S3AR_real_CPU_token_sha256":S3AR_SHA,
        "S3O_original_fp16_unchanged_sha256":S3O_SHA,
        "new_S3AW_physical_fp16_sha256":new_fp16_sha,
        "new_S3AW_physical_fp16_size":model_file.stat().st_size,
        "physical_fp16_reload_equal":True,
        "original_graph_physical_fp16_bitwise_equal":True,
        "full_S194_original_616_J7_digests_from_S3AV_protected":True,
        "train_count":1024,"S3AU_new_TRAIN_count":512,"S3AQ_prior_TRAIN_count":256,
        "rehearsal_S3M_TRAIN_count":128,"rehearsal_S3X_TRAIN_count":128,
        "DEV_count":128,"DEV128_excluded_from_gradient_optimizer":True,
        "DEV128_scored_only_after_FP16_SHA_commit":True,
        "S3AQ_future_EVAL128_not_prefilled_scored_or_trained":True,
        "S3AO_already_scored_EVAL128_NOT_reused_for_training_or_selection":True,
        "only_query_role_head_trained":True,
        "original_graph_and_independent_NRA_frozen":True,
        "optimizer_steps":STEPS,"seed":SEED,"lr":LR,
        "S3AQ_TRAIN256_role_symmetry_weight":SYMMETRY_WEIGHT,
        "S3AU_TRAIN512_role_symmetry_weight":S3AU_SYMMETRY_WEIGHT,
        "S3AQ_role_symmetry_before":float(sym_before),
        "S3AQ_role_symmetry_after":float(sym_after),
        "S3AU_role_symmetry_before":float(novel_sym_before),
        "S3AU_role_symmetry_after":float(novel_sym_after),
        "train_loss_before":float(before),"train_loss_after":float(after),
        "train_trace":trace,"TRAIN_only_role_metrics":{
            "S3M_original128":train_results[0],"S3X_reversed128":train_results[1],
            "S3AQ_new256":train_results[2],"S3AU_new512":train_results[3]},
        "DEV128_paired_S3AS_baseline":prior_dev,
        "DEV128_paired_S3AW_candidate":after_dev,
        "DEV128_both_source_first_gain":dev_source_first_gain,
        "DEV128_both_destination_first_gain":dev_dest_first_gain,
        "DEV128_source_first_non_regression":dev_source_first_gain>=0,
        "DEV128_destination_first_non_regression":dev_dest_first_gain>=0,
        "candidate_accepted":False,"output_gate":0.0,
        "full_original_CPU_J7_candidate_gain_tested":False,
        "all_in_physical_bpp_tested":False,"paired_candidate_CPU_speed_tested":False,
        "Jessica_LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    (out/"NRA2_S3AW_S3AS_INIT_TRAIN1024_DEV128_ROLE_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in (
         "stage","train_count","new_S3AW_physical_fp16_sha256",
         "train_loss_before","train_loss_after","TRAIN_only_role_metrics",
         "candidate_accepted","output_gate")},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
