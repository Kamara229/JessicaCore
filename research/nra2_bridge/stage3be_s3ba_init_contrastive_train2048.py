"""S3BE: S3BA-initialized query role head, 2048 TRAIN only; no DEV or future EVAL model inputs.

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
from research.nra2_bridge.stage3bc_contrastive_train512_prereg import (
    build_rows as s3bc_rows, commitment as s3bc_commitment,
    SPLIT as S3BC_TRAIN, DESTINATION_FIRST as S3BC_REVERSED,
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

SEED=2026111498
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
        raise AssertionError("S3BE "+msg)

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


S3BA_RUN=43590032315
S3BA_SHA="4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BC_RUN=43591378375
S3BC_SHA="65832674475452dbf4d5800ab310ee07ce2d6f5a9e237307e6ec3299e19e2d4c"
S3BD_RUN=43591957498
S3BD_TOKEN_SHA="0d3596a6ef1bac4d7fc3b1b4edbe34103966d2f12b040757d3609eace59a5c78"
S3BD_HIDDEN_SHA="241dfd2e9ab99bb8e494f9cf3041a6a61e1c6a4f65c82bfe493e6ceeb3f7d587"
S3BD_LABELS_SHA="9ac2bef5579fd4d6083d1c418ef0293f66d820f659447caf4c67dbafe9a0b4ea"

def main():
    if len(sys.argv)!=16:
        raise SystemExit("S3BE S3N S3Y S3X S3O S3AS S3AR S3AV S3AU S3AW S3AZ S3AY S3BA S3BC S3BD OUT")
    (n_dir,y_dir,x_dir,o_dir,z_dir,af_dir,av_dir,au_dir,aw_dir,az_dir,ay_dir,
     ba_dir,bc_dir,bd_dir,out)=map(Path,sys.argv[1:])
    require(not out.exists() or not any(out.iterdir()),"immutable research output cannot be overwritten")
    def read(base,name):
        return json.loads((base/name).read_text())
    sn=read(n_dir,"NRA2_S3N_192_ROLEBALANCED_REAL_CPU_AUDIT.json")
    sy=read(y_dir,"NRA2_S3Y_128_REVERSAL_TRAIN_REAL_CPU_AUDIT.json")
    sx=read(x_dir,"NRA2_S3X_TRAIN128_REVERSE_GRAMMAR_PREREG.json")
    so=read(o_dir,"NRA2_S3O_NEURAL_SEMANTIC_ROLES_128TRAIN_64DEV_AUDIT.json")
    sz=read(z_dir,"NRA2_S3AS_S3Z_INIT_TRAIN512_DEV128_ROLE_AUDIT.json")
    saf=read(af_dir,"NRA2_S3AR_TRAIN256_DEV128_REAL_CPU_J7_AUDIT.json")
    sav=read(av_dir,"NRA2_S3AV_TRAIN512_REAL_CPU_J7_AUDIT.json")
    sau=read(au_dir,"NRA2_S3AU_NEW_TRAIN512_PARAPHRASE_PREREG.json")
    saw=read(aw_dir,"NRA2_S3AW_S3AS_INIT_TRAIN1024_DEV128_ROLE_AUDIT.json")
    saz=read(az_dir,"NRA2_S3AZ_TRAIN512_REAL_CPU_J7_AUDIT.json")
    say=read(ay_dir,"NRA2_S3AY_NEW_TRAIN512_PARAPHRASE_PREREG.json")
    sba=read(ba_dir,"NRA2_S3BA_S3AW_INIT_TRAIN1536_DEV128_ROLE_AUDIT.json")
    sbc=read(bc_dir,"NRA2_S3BC_CONTRASTIVE_ROLE_TRAIN512_PREREG.json")
    sbd=read(bd_dir,"NRA2_S3BD_TRAIN512_REAL_CPU_J7_AUDIT.json")

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
    bd_tok=bd_dir/"NRA2_S3BD_REAL_CPU_TRAIN512_ALL_TOKENS_FP16.safetensors"
    bd_lab=bd_dir/"NRA2_S3BD_TRAIN512_STRUCTURAL_LABELS.safetensors"
    bd_hidden=bd_dir/"NRA2_S3BD_REAL_43M_TRAIN512_HIDDEN320_FP16.safetensors"
    orig=o_dir/"NRA2_S3O_NATIVE_CPU_LEARNED_ROLES_TRAIN128_FP16.safetensors"
    s3as_parent=z_dir/"NRA2_S3AS_S3Z_INIT_ROLE512_FP16.safetensors"
    saw_parent=aw_dir/"NRA2_S3AW_S3AS_INIT_ROLE1024_FP16.safetensors"
    parent=ba_dir/"NRA2_S3BA_S3AW_INIT_ROLE1536_FP16.safetensors"

    require(sx==s3x_commitment(s3x_rows()) and sx["commitment_sha256"]==S3X_SHA,
            "S3X source TRAIN128 prereg SHA mismatch")
    for audit in (sn,sy,saf,sav,saz,sbd):
        require(audit["all_616_token_digest_matches_S194"] is True and
                audit["physical_reload_bitwise_equal"] is True and
                audit["nra_output_gate"]==0.0 and audit["adapter_trained"] is False if "adapter_trained" in audit
                else audit["all_616_token_digest_matches_S194"] is True and
                audit["physical_reload_bitwise_equal"] is True and
                audit["nra_output_gate"]==0.0,
                "authentic original S194 616 J7 or frozen CPU readout drift")
    require(all(a["full_token_fp16_sha256"]==sha_file(p)
                for a,p in ((sn,sn_tok),(sy,sy_tok),(saf,af_tok),
                            (sav,av_tok),(saz,az_tok),(sbd,bd_tok))),
            "physical original 43M full token FP16 audit SHA mismatches")
    require((sn["full_token_fp16_sha256"],sy["full_token_fp16_sha256"],
             saf["full_token_fp16_sha256"],sav["full_token_fp16_sha256"],
             saz["full_token_fp16_sha256"],sbd["full_token_fp16_sha256"])==
            (S3N_SHA,S3Y_SHA,S3AR_SHA,S3AV_SHA,S3AZ_SHA,S3BD_TOKEN_SHA),
            "one of six pinned full-token source SHA changed")
    require(sbd["physical_feature_fp16_sha256"]==S3BD_HIDDEN_SHA==sha_file(bd_hidden)
            and sbd["physical_graph_labels_sha256"]==S3BD_LABELS_SHA==sha_file(bd_lab)
            and sbd["stage"]=="NRA2_S3BD_FROZEN_S3BC_NEW_ROLE_CONTRASTIVE_TRAIN512_NO_SCORE_CPU_CAPTURE_ONLY"
            and sbd["S3BC_registered_commitment_sha256"]==S3BC_SHA
            and sbd["S3BC_successful_prereg_run"]==S3BC_RUN
            and sbd["graph_prefill_count"]==512 and sbd["train_count"]==512
            and sbd["group_sizes"]=={S3BC_TRAIN:512}
            and sbd["S3BC_scored_DEV128_not_prefilled_or_used_in_optimizer"]
            and sbd["S3BC_future_EVAL128_never_prefilled_scored_or_trained"]
            and sbd["native_tokenizer_S3O_query_alignment_verified"]
            and sbd["real_hidden320_shape"]==[512,320]
            and sbd["calibri_and_green_tunnel_unchanged"],
            "S3BD physical true CPU TRAIN512 proof, SHA or excluded evaluation drift")
    require(saf["S3AQ_registered_commitment_sha256"]==S3AQ_COMMIT_SHA and
            saf["S3AQ_future_EVAL128_never_prefilled_scored_or_trained"] and
            saf["group_sizes"]=={S3AQ_TRAIN:256,S3AQ_DEV:128} and
            saf["train_count"]==256 and saf["dev_count"]==128 and
            saf["adapter_trained"] is False,
            "S3AR source TRAIN256 must exclude 128 scored DEV tokens")
    require(sn["group_sizes"]=={"TRAIN128":128,"DEV64":64} and sy["train_count"]==128,
            "S3N original and S3Y reversal TRAIN128 provenance drift")
    require(so["FP16_checkpoint_sha256"]==S3O_SHA==sha_file(orig)
            and sz["new_S3AS_physical_fp16_sha256"]==S3AS_SHA==sha_file(s3as_parent)
            and saw["new_S3AW_physical_fp16_sha256"]==S3AW_SHA==sha_file(saw_parent)
            and saw["S3O_original_fp16_unchanged_sha256"]==S3O_SHA,
            "original S3O graph/S3AS/S3AW physical FP16 ancestor SHA mismatch")
    require(sba["new_S3BA_physical_fp16_sha256"]==S3BA_SHA==sha_file(parent)
            and sba["train_count"]==1536 and sba["DEV_count"]==128
            and sba["DEV128_paired_S3BA_candidate"]["all"]["both_correct"]==94
            and sba["DEV128_is_already_scored_exploratory_not_independent"]
            and sba["DEV128_excluded_from_gradient_optimizer"]
            and sba["S3AQ_future_EVAL128_not_prefilled_scored_or_trained"]
            and sba["original_graph_physical_fp16_bitwise_equal"] and
            sba["full_S194_original_616_J7_digests_from_S3AV_protected"]
            and sba["output_gate"]==0.0 and not sba["candidate_accepted"],
            "pinned S3BA parent physical FP16 or guarded TRAIN/DEV status drift")
    rows3=s3au_rows()
    rows4=s3ay_rows()
    rows5=s3bc_rows()
    require(sau==s3au_commitment(rows3) and sau["commitment_sha256"]==S3AU_SHA
            and say==s3ay_commitment(rows4) and say["commitment_sha256"]==S3AY_SHA
            and sbc==s3bc_commitment(rows5) and sbc["commitment_sha256"]==S3BC_SHA,
            "precommitted TRAIN-only data SHA or exact order mismatch")
    require(sbc["no_new_CPU_prefill_or_model_weights_or_training_or_scores"] and
            sbc["new_prompts_sha_disjoint_from_all_S3AQ_and_S3AU_S3AY"]
            and sbc["future_EVAL128_not_prefilled_scored_or_trained"]
            and sbc["S3BA_pinned_role_fp16_sha256"]==S3BA_SHA,
            "S3BC protected new contrastive TRAIN512 prereg drift")
    rows0=s3m_rows()[:128]
    rows1=s3x_rows()
    aq_rows=s3aq_rows()
    rows2=[r for r in aq_rows if r.split==S3AQ_TRAIN]
    # Neither the already-scored DEV nor future independent EVAL are passed to a
    # tokenizer, model, optimizer, model-score routine or heldout feature capture.
    require([len(v) for v in (rows0,rows1,rows2,rows3,rows4,rows5)]==
            [128,128,256,512,512,512], "exactly 2048 precommitted TRAIN-only rows")
    train_sha=[r.prompt_sha256 for v in (rows0,rows1,rows2,rows3,rows4,rows5) for r in v]
    forbid={r.prompt_sha256 for r in aq_rows if r.split in (S3AQ_DEV,S3AQ_EVAL)}
    require(len(train_sha)==2048 and len(set(train_sha))==2048 and
            not (set(train_sha)&forbid),
            "S3BE TRAIN2048 must never overlap S3AQ DEV128/EVAL128 prompt")
    require(sn["split_groups"]["TRAIN128"]["prompt_sha256"]==[r.prompt_sha256 for r in rows0] and
            sy["split_groups"][sx["group"]]["prompt_sha256"]==[r.prompt_sha256 for r in rows1],
            "original first two native CPU TRAIN128 prompt-order mismatch")
    def split_sha(rows):
        return hashlib.sha256(json.dumps([r.prompt_sha256 for r in rows],
            sort_keys=True,separators=(",",":")).encode()).hexdigest()
    require(saf["S3AQ_TRAIN256_source_sha256"]==split_sha(rows2) and
            sbd["S3BC_TRAIN512_source_sha256"]==split_sha(rows5),
            "S3AQ/S3BC TRAIN prompt-hash CPU capture order mismatch")
    require(sav["S3AU_registered_commitment_sha256"]==S3AU_SHA and
            saz["S3AY_registered_commitment_sha256"]==S3AY_SHA and
            sav["physical_graph_labels_sha256"]==sha_file(av_lab) and
            saz["physical_graph_labels_sha256"]==sha_file(az_lab),
            "previous TRAIN512 protected physical labels changed")

    # No original CPU new inference here: frozen genuine 43M token artifacts.
    tok_files=(sn_tok,sy_tok,af_tok,av_tok,az_tok,bd_tok)
    lab_files=(sn_lab,sy_lab,af_lab,av_lab,az_lab,bd_lab)
    toks=[load_file(str(f),device="cpu") for f in tok_files]
    labs=[load_file(str(f),device="cpu") for f in lab_files]
    parent_weights=load_file(str(parent),device="cpu")
    orig_weights=load_file(str(orig),device="cpu")
    for k,v in orig_weights.items():
        if k.startswith("bridge.graph."):
            require(torch.equal(v,parent_weights[k]),
                    "S3BA pinned original graph must remain identical to S3O")
    require(torch.equal(parent_weights["normalizer.mean"],orig_weights["normalizer.mean"])
            and torch.equal(parent_weights["normalizer.scale"],orig_weights["normalizer.scale"]),
            "original native 43M normalizer must remain physically pinned")
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    model=S3KGraphWithLearnedRoles()
    model.load_state_dict({k.removeprefix("bridge."):v for k,v in parent_weights.items()
                           if k.startswith("bridge.")},strict=True)
    model.graph.requires_grad_(False)
    graph_before={k:v.detach().clone() for k,v in model.graph.state_dict().items()}
    mean=parent_weights["normalizer.mean"].float()
    scale=parent_weights["normalizer.scale"].float()
    tokenizer=exact_original_tokenizer()
    inputs=[]
    names=("token_hidden320","token_mask","token_lengths")
    source_rows=(rows0,rows1,rows2,rows3,rows4,rows5)
    for rows,t,g in zip(source_rows,toks,labs):
        n=len(rows)
        require(set(g)=={"adjacency","source","target","reachability"}
                and g["source"].shape[1]==NODES and g["target"].shape[1]==NODES,
                "native original CPU physical source/target gold ABI drift")
        tt,tm,tl=(t[names[0]][:n],t[names[1]][:n],t[names[2]][:n])
        q,_binding=bind(tokenizer,[r.question for r in rows],tt,tm,tl)
        src,dst=gold_roles(g,0,n)
        require(bool(torch.all(q.node_order.sort(1).values ==
                    torch.stack([src,dst],dim=-1).sort(1).values)),
                "literal TRAIN query roles do not match physical CPU role labels")
        inputs.append((normalize(tt,mean,scale,tm),tm,q,src,dst))

    # Optimize ONLY query-role head, no original CPU/graph/Green Tunnel/NRA updates.
    optimizer=torch.optim.AdamW(model.query.parameters(),lr=LR,weight_decay=0.02)
    def aggregate():
        ls=[role_loss(model,v) for v in inputs]
        syms=[role_symmetry_loss(model,inputs[2],rows2)]
        syms.extend(octet_symmetry_loss(model,inputs[i],source_rows[i]) for i in (3,4,5))
        return 2*ls[0]+2*ls[1]+2*ls[2]+ls[3]+ls[4]+ls[5]+.15*sum(syms),ls,syms
    model.eval()
    with torch.no_grad():
        before,per_before,sym_before=aggregate()
    trace=[]
    for step in range(STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss,parts,syms=aggregate()
        require(bool(torch.isfinite(loss)),"nonfinite TRAIN-only role-head loss")
        loss.backward()
        require(all(p.grad is None for p in model.graph.parameters()),
                "frozen S3O graph received unauthorized gradient")
        torch.nn.utils.clip_grad_norm_(model.query.parameters(),1.0)
        optimizer.step()
        if step+1 in (1,40,80,120,160):
            trace.append({"step":step+1,
                          "TRAIN_only_group_losses":[float(v.detach()) for v in parts],
                          "TRAIN_only_group_symmetry":[float(v.detach()) for v in syms]})
    model.eval()
    with torch.no_grad():
        after,per_after,sym_after=aggregate()
    require(bool(torch.isfinite(after)) and after<before,
            "2048 TRAIN-only aggregate role loss did not decrease")
    require(all(torch.equal(v,model.graph.state_dict()[k]) for k,v in graph_before.items()),
            "original frozen S3O graph changed during TRAIN-only training")
    require(sha_file(parent)==S3BA_SHA and sha_file(orig)==S3O_SHA and
            sha_file(bd_tok)==S3BD_TOKEN_SHA and sha_file(bd_lab)==S3BD_LABELS_SHA,
            "original source frozen FP16 files changed during training")
    # Commit physical FP16 *before* computing training role metrics. No DEV/EVAL scoring.
    out.mkdir(parents=True,exist_ok=False)
    model_file=out/"NRA2_S3BE_S3BA_INIT_ROLE2048_FP16.safetensors"
    packed={"bridge."+k:v.detach().cpu().half().contiguous()
            for k,v in model.state_dict().items()}
    packed["normalizer.mean"]=mean.half().contiguous()
    packed["normalizer.scale"]=scale.half().contiguous()
    save_file(packed,str(model_file))
    reloaded=load_file(str(model_file),device="cpu")
    require(set(reloaded)==set(packed) and
            all(torch.equal(reloaded[k],v) for k,v in packed.items()),
            "physical S3BE FP16 bitwise reload mismatch")
    for k,v in orig_weights.items():
        if k.startswith("bridge.graph."):
            require(torch.equal(v,reloaded[k]),"original S3O graph FP16 drift")
    require(torch.equal(parent_weights["normalizer.mean"],reloaded["normalizer.mean"]) and
            torch.equal(parent_weights["normalizer.scale"],reloaded["normalizer.scale"]),
            "S3BA original frozen normalizer drift")
    fp16_sha=sha_file(model_file)
    candidate=S3KGraphWithLearnedRoles().eval()
    candidate.load_state_dict({k.removeprefix("bridge."):v for k,v in reloaded.items()
                               if k.startswith("bridge.")},strict=True)
    candidate.requires_grad_(False)
    baseline=S3KGraphWithLearnedRoles().eval()
    baseline.load_state_dict({k.removeprefix("bridge."):v for k,v in parent_weights.items()
                              if k.startswith("bridge.")},strict=True)
    baseline.requires_grad_(False)
    reversed_selectors=(
        lambda r:False,
        lambda r:r.query_style in (0,2),
        lambda r:r.query_style in S3AQ_REVERSED,
        lambda r:r.query_style in S3AU_REVERSED,
        lambda r:r.query_style in S3AY_REVERSED,
        lambda r:r.query_style in S3BC_REVERSED,
    )
    train_metrics_new=[metrics(candidate,data,rows,selector) for data,rows,selector
                       in zip(inputs,source_rows,reversed_selectors)]
    train_metrics_parent=[metrics(baseline,data,rows,selector) for data,rows,selector
                          in zip(inputs,source_rows,reversed_selectors)]
    require(sha_file(model_file)==fp16_sha and sha_file(parent)==S3BA_SHA
            and sha_file(bd_tok)==S3BD_TOKEN_SHA,
            "physical role-head or source model SHA changed after TRAIN-only readout")
    group_names=("S3M_TRAIN128","S3X_TRAIN128","S3AQ_TRAIN256",
                 "S3AU_TRAIN512","S3AY_TRAIN512","S3BC_TRAIN512")
    report={
        "stage":"NRA2_S3BE_S3BA_INIT_2048_TRAIN_ONLY_ROLE_HEAD_NO_DEV_SCORE",
        "S3BA_pinned_parent_run":S3BA_RUN,
        "S3BA_parent_fp16_sha256":S3BA_SHA,
        "S3BC_prereg_run":S3BC_RUN,
        "S3BC_registered_commitment_sha256":S3BC_SHA,
        "S3BD_native_CPU_run":S3BD_RUN,
        "S3BD_full_token_fp16_sha256":S3BD_TOKEN_SHA,
        "S3BD_hidden320_fp16_sha256":S3BD_HIDDEN_SHA,
        "S3BD_labels_sha256":S3BD_LABELS_SHA,
        "S3O_graph_original_fp16_sha256":S3O_SHA,
        "new_S3BE_physical_fp16_sha256":fp16_sha,
        "new_S3BE_physical_fp16_size":model_file.stat().st_size,
        "physical_fp16_reload_equal":True,
        "original_graph_physical_fp16_bitwise_equal":True,
        "original_frozen_43M_normalizer_bitwise_equal":True,
        "original_616_S194_J7_digests_verified_from_six_true_CPU_sources":True,
        "train_count":2048,
        "train_groups":dict(zip(group_names,(128,128,256,512,512,512))),
        "S3BC_new_TRAIN_count":512,
        "optimizer_steps":STEPS,"seed":SEED,"lr":LR,"weight_decay":0.02,
        "gradient_clip_norm":1.0,
        "TRAIN_group_rehearsal_weights":dict(zip(group_names,(2,2,2,1,1,1))),
        "role_symmetry_weight_per_TRAIN_group":.15,
        "training_only_role_symmetry_before":[float(v) for v in sym_before],
        "training_only_role_symmetry_after":[float(v) for v in sym_after],
        "TRAIN_loss_before":float(before),"TRAIN_loss_after":float(after),
        "TRAIN_group_loss_before":dict(zip(group_names,(float(v) for v in per_before))),
        "TRAIN_group_loss_after":dict(zip(group_names,(float(v) for v in per_after))),
        "train_trace":trace,
        "TRAIN_only_parent_role_metrics":dict(zip(group_names,train_metrics_parent)),
        "TRAIN_only_S3BE_role_metrics":dict(zip(group_names,train_metrics_new)),
        "no_DEV128_loaded_into_optimizer_or_scored":True,
        "S3AQ_DEV128_repeatedly_scored_not_independent":True,
        "S3AQ_future_EVAL128_never_CPU_prefilled_scored_or_trained":True,
        "no_heldout_model_inference_this_stage":True,
        "only_query_role_head_trained":True,
        "original_graph_43M_green_tunnel_NRA_frozen":True,
        "output_gate":0.0,"candidate_accepted":False,
        "full_original_CPU_J7_candidate_gain_tested":False,
        "all_in_physical_bpp_tested":False,
        "paired_candidate_CPU_speed_tested":False,
        "Jessica_LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    (out/"NRA2_S3BE_S3BA_INIT_TRAIN2048_NO_DEV_ROLE_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in (
         "stage","train_count","new_S3BE_physical_fp16_sha256",
         "TRAIN_loss_before","TRAIN_loss_after","train_groups",
         "TRAIN_only_parent_role_metrics","TRAIN_only_S3BE_role_metrics",
         "candidate_accepted","output_gate")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
