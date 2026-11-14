"""S3BT: extend physically pinned JOINT S3BN to new language semantics on 2560 genuine TRAIN-only CPU questions.

Rehearse S3M original TRAIN128 and S3X reversed TRAIN128, initialize from
physical S3AS; S3O graph, true 43M, Green Tunnel and independent NRA frozen.
Four prompts per topology enforce same-role paraphrase and reversed-role symmetry.
All three S3AQ/S3BJ/S3BO independent EVAL128 were consumed; none may be reused.
No DEV or old/new EVAL model inference; original protected 616 J7 unchanged.
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
from research.nra2_bridge.stage3bm_joint_semantic_pair_role_head import S3BMGraphWithJointRoles
from research.nra2_bridge.stage3br_relation_language_train512_prereg import (
    build_rows as s3br_rows,commitment as s3br_commitment,
    SPLIT as S3BR_TRAIN, DESTINATION_FIRST as S3BR_REVERSED,
)
from research.nra2_bridge.stage3k_train_real_entity_binding import (
    exact_original_tokenizer,sha_file,
)
from research.nra2_bridge.stage3f_token_attention import normalize

SEED=2026111570
STEPS=96
LR=0.00015
SYMMETRY_WEIGHT=0.30
TEACHER_KL_WEIGHT=0.35
TEACHER_TEMPERATURE=2.0
GROUP_WEIGHTS=(2.,2.,2.,1.5,1.5,1.5,3.)
S3BN_SHA="ea8a81665370cba22c4e8bc75fbe82314089244d4b32278ee62c2c30d88ae811"
S3BR_SHA="e941bd45f947b7fc16f0cce4307ed9ea69c0b0e8347da7c0754717430ff0262f"
S3BS_TOKEN_SHA="95912db37d26077f4abb2e1e4050aad13155c3988aa7ee592e6da8d81c3eec5c"
S3BS_RUN=43649375691
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
        raise AssertionError("S3BT "+msg)

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
    if len(sys.argv)!=19:
        raise SystemExit("S3BT original 14 TRAIN source dirs + S3BN S3BR S3BS + OUTPUT")
    (n_dir,y_dir,x_dir,o_dir,z_dir,af_dir,av_dir,au_dir,aw_dir,az_dir,ay_dir,
     ba_dir,bc_dir,bd_dir,sbn_dir,sbr_dir,sbs_dir,out)=map(Path,sys.argv[1:])
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
            "S3BN TRAIN2048 must never overlap S3AQ DEV128/EVAL128 prompt")
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


    # Physically pinned 2048 source inputs, physical original 43M CPU is never
    # executed here. The new S3BR training split has 512 fresh grammars on
    # 64 original TRAIN graph topologies. All three independent EVAL128 splits
    # are forbidden in this optimizer or in any checkpoint selection.
    init=sbn_dir/"NRA2_S3BN_JOINT_ROLE_SAFE_TRAIN2048_FP16.safetensors"
    src_init=json.loads((sbn_dir/"NRA2_S3BN_JOINT_TRAIN2048_RESEARCH_AUDIT.json").read_text())
    require(src_init.get("stage")=="NRA2_S3BN_JOINT_PAIR_SEMANTIC_TRAIN2048_NO_EVAL"
            and src_init.get("safe_new_pairwise_fp16_sha256")==S3BN_SHA==sha_file(init)
            and src_init.get("S3BA_parent_FP16_SHA256")==S3BA_SHA
            and src_init.get("S3BM_only_new_pair_trainable_params")==42869
            and src_init.get("train_count")==2048
            and src_init.get("original_S194_616_J7_verified_across_six_CPU_TRAIN_sources") is True
            and src_init.get("no_independent_EVAL_evaluated_or_retuned") is True
            and src_init.get("candidate_accepted") is False
            and src_init.get("output_gate")==0.,
            "S3BN joint semantic physical parent or 616 J7 drift")
    sbr=json.loads((sbr_dir/"NRA2_S3BR_DIRECTIONAL_TRAIN512_PREREG.json").read_text())
    new_rows=s3br_rows()
    require(sbr==json.loads(json.dumps(s3br_commitment(new_rows)))
            and sbr.get("stage")=="NRA2_S3BR_RELATION_LANGUAGE_TRAIN512_PREREG_ONLY"
            and sbr.get("commitment_sha256")==S3BR_SHA
            and sbr.get("TRAIN512_count")==512
            and sbr.get("TRAIN_graphs")==64
            and sbr.get("prior_EVAL1_consumed_run")==43640363396
            and sbr.get("prior_EVAL2_consumed_run")==43644153544
            and sbr.get("prior_EVAL3_consumed_run")==43648038071
            and sbr.get("prior_EVALs_NEVER_model_input_nor_target_for_train") is True
            and sbr.get("source_original_TRAIN_graphs_only") is True
            and sbr.get("S3BN_frozen_initial_candidate_sha256")==S3BN_SHA
            and sbr.get("output_gate")==0. and sbr.get("candidate_accepted") is False,
            "S3BR new semantic TRAIN512 prereg SHA/labels changed")
    sbs=json.loads((sbs_dir/"NRA2_S3BS_TRAIN512_REAL_CPU_J7_AUDIT.json").read_text())
    bs_tokens=sbs_dir/"NRA2_S3BS_REAL_CPU_TRAIN512_ALL_TOKENS_FP16.safetensors"
    bs_labels=sbs_dir/"NRA2_S3BS_TRAIN512_STRUCTURAL_LABELS.safetensors"
    bs_hidden=sbs_dir/"NRA2_S3BS_REAL_43M_TRAIN512_HIDDEN320_FP16.safetensors"
    require(sbs.get("stage")=="NRA2_S3BS_ORIGINAL43M_CPU_NEW_RELATION_TRAIN512_ONLY"
            and sbs.get("S3BR_new_relation_TRAIN512_prereg_sha256")==S3BR_SHA
            and sbs.get("S3BR_new_relation_TRAIN512_prereg_run")==43648885215
            and sbs.get("S3BN_TRAIN2048_initial_physically_locked_fp16_sha256")==S3BN_SHA
            and sbs.get("original_3_independent128_EVAL_splits_permanently_consumed_not_used") is True
            and sbs.get("S3BR_exclusively_original_S3AQ_TRAIN_topologies") is True
            and sbs.get("S3BR_no_original43M_adapter_head_scored_during_CPU") is True
            and sbs.get("all_616_token_digest_matches_S194") is True
            and sbs.get("graph_prefill_count")==512
            and sbs.get("train_count")==512 and sbs.get("dev_count")==0 and sbs.get("eval_count")==0
            and sbs.get("group_sizes")=={S3BR_TRAIN:512}
            and sbs.get("split_groups",{}).get(S3BR_TRAIN,{}).get("prompt_sha256")==
                [r.prompt_sha256 for r in new_rows]
            and sbs.get("native_tokenizer_S3O_query_alignment_verified") is True
            and sbs.get("physical_reload_bitwise_equal") is True
            and sbs.get("labels_materialized_after_native_CPU") is True
            and sbs.get("full_token_fp16_sha256")==S3BS_TOKEN_SHA==sha_file(bs_tokens)
            and sbs.get("physical_graph_labels_sha256")==sha_file(bs_labels)
            and sbs.get("physical_feature_fp16_sha256")==sha_file(bs_hidden)
            and sbs.get("candidate_accepted") is False and sbs.get("nra_output_gate")==0.,
            "S3BS authentic new original43M TRAIN512 physical SHA/616 J7 missing")

    # Only original 2048 TRAIN physical tokens plus new S3BR TRAIN512.
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    weights=load_file(str(init),device="cpu")
    for k,v in parent_weights.items():
        if k.startswith("bridge.graph.") or k.startswith("normalizer."):
            require(k in weights and torch.equal(v,weights[k]),
                    "S3BT original frozen graph and normalizer parent mismatch")
        if k.startswith("bridge.query."):
            rem=k.removeprefix("bridge.query.")
            require(torch.equal(v,weights["bridge.query.base."+rem]),
                    "S3BT original S3BA teacher head changed in S3BN")
    model=S3BMGraphWithJointRoles()
    model.load_state_dict({
        k.removeprefix("bridge."):v for k,v in weights.items()
        if k.startswith("bridge.")},strict=True)
    model.graph.requires_grad_(False)
    model.query.base.requires_grad_(False)
    require(all(not v.requires_grad for v in model.graph.parameters())
            and all(not v.requires_grad for v in model.query.base.parameters()),
            "original 43M graph+role teacher requires gradient")
    mean=weights["normalizer.mean"].float()
    scale=weights["normalizer.scale"].float()
    tokenizer=exact_original_tokenizer()
    names=("token_hidden320","token_mask","token_lengths")
    source_rows=[rows0,rows1,rows2,rows3,rows4,rows5,new_rows]
    frozen_toks=toks+[load_file(str(bs_tokens),device="cpu")]
    frozen_labs=labs+[load_file(str(bs_labels),device="cpu")]
    inputs=[]
    for rows,pack,gold in zip(source_rows,frozen_toks,frozen_labs):
        n=len(rows)
        tt,tm,tl=(pack[names[0]][:n],pack[names[1]][:n],pack[names[2]][:n])
        q,_=bind(tokenizer,[r.question for r in rows],tt,tm,tl)
        src,dst=gold_roles(gold,0,n)
        require(bool(torch.equal(q.node_order.sort(1).values,
                    torch.stack((src,dst),dim=1).sort(1).values)),
                "S3BT TRAIN literal source/target identity mismatch")
        inputs.append((normalize(tt,mean,scale,tm),tm,q,src,dst))
    require(sum(len(g[0]) for g in inputs)==2560,
            "S3BT must be 2048 historical plus 512 NEW TRAIN, NO EVAL")
    original_graph={k:v.detach().clone() for k,v in model.graph.state_dict().items()}
    original_teacher={k:v.detach().clone() for k,v in model.query.base.state_dict().items()}
    new_params=[p for p in model.query.parameters() if p.requires_grad]
    require(sum(v.numel() for v in new_params)==42869,"only S3BM pair layers trainable")
    selectors=(
        lambda r:False,
        lambda r:r.query_style in (0,2),
        lambda r:r.query_style in S3AQ_REVERSED,
        lambda r:r.query_style in S3AU_REVERSED,
        lambda r:r.query_style in S3AY_REVERSED,
        lambda r:r.query_style in S3BC_REVERSED,
        lambda r:r.query_style in S3BR_REVERSED,
    )
    group_names=("S3M_TRAIN128","S3X_TRAIN128","S3AQ_TRAIN256",
                 "S3AU_TRAIN512","S3AY_TRAIN512","S3BC_TRAIN512",
                 "S3BR_NEW_RELATION_TRAIN512")
    model.eval()
    with torch.no_grad():
        baseline_metrics=[metrics(model,g,rows,sel)
                          for g,rows,sel in zip(inputs,source_rows,selectors)]
        require([m["all"]["both_correct"] for m in baseline_metrics[:6]]==
                [128,128,256,512,512,465],
                "S3BT pinned 2048 TRAIN S3BN initial logits not physically reproduced")
        original_teacher_logits=[tuple(v.detach()
                             for v in model.query(g[0],g[1],g[2]))
                             for g in inputs[:6]]
    def sym_new(pred,rows):
        require(len(rows)==512 and all(rows[i].query_style==i%8
                    for i in range(512)),
                "new S3BR TRAIN512 must be eight grammatical variants per graph")
        a,b=pred
        pa,pb=F.softmax(a,dim=-1),F.softmax(b,dim=-1)
        terms=[]
        # Pairwise same semantic relation despite different surface voices.
        for i in (0,2,4,6):
            terms.extend((F.mse_loss(pa[i::8],pa[i+1::8]),
                          F.mse_loss(pb[i::8],pb[i+1::8])))
        # Reverse source-target on same graph, no label injected into forward.
        for i in (0,4):
            terms.extend((F.mse_loss(pa[i::8],pb[i+2::8]),
                          F.mse_loss(pb[i::8],pa[i+2::8])))
        for i,j in ((0,4),(2,6)):
            terms.extend((F.mse_loss(pa[i::8],pa[j::8]),
                          F.mse_loss(pb[i::8],pb[j::8])))
        return sum(terms)/len(terms)
    def aggregate():
        predictions=[model.query(g[0],g[1],g[2]) for g in inputs]
        ls=[.5*(F.cross_entropy(pred[0],g[3])+F.cross_entropy(pred[1],g[4]))
            for g,pred in zip(inputs,predictions)]
        kl=torch.zeros((),dtype=torch.float32)
        for (src,dst),(ref_s,ref_t) in zip(predictions[:6],original_teacher_logits):
            kl+=.5*(F.kl_div(F.log_softmax(src/2.,dim=-1),
                            F.softmax(ref_s/2.,dim=-1),reduction="batchmean")
                    +F.kl_div(F.log_softmax(dst/2.,dim=-1),
                             F.softmax(ref_t/2.,dim=-1),reduction="batchmean"))*4.
        kl=kl/6.
        symmetry=sym_new(predictions[-1],new_rows)
        total=(sum(w*loss for w,loss in zip(GROUP_WEIGHTS,ls))
               +TEACHER_KL_WEIGHT*kl+.45*symmetry)
        return total,ls,kl,symmetry
    model.eval()
    with torch.no_grad():
        before,per_before,kl_before,sym_before=aggregate()
    optimizer=torch.optim.AdamW(new_params,lr=LR,weight_decay=.02)
    trace=[]
    for step in range(STEPS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss,parts,teacher_kl,symmetry=aggregate()
        require(bool(torch.isfinite(loss)), "nonfinite TRAIN-only curriculum loss")
        loss.backward()
        require(all(p.grad is None for p in model.graph.parameters()) and
                all(p.grad is None for p in model.query.base.parameters()),
                "frozen Calibri original graph or S3BA role teacher got gradients")
        torch.nn.utils.clip_grad_norm_(new_params,1.)
        optimizer.step()
        if step+1 in (1,24,48,72,96):
            trace.append({"step":step+1,
                          "TRAIN_only_group_losses":[float(x.detach()) for x in parts],
                          "TRAIN_only_original_teacher_KL":float(teacher_kl.detach()),
                          "TRAIN_only_new_relation_symmetry":float(symmetry.detach())})
    model.eval()
    with torch.no_grad():
        after,per_after,kl_after,sym_after=aggregate()
    require(bool(torch.isfinite(after)) and after<before,
            "S3BT original plus novel TRAIN2560 objective did not improve")
    require(all(torch.equal(v,model.graph.state_dict()[k])
                for k,v in original_graph.items())
            and all(torch.equal(v,model.query.base.state_dict()[k])
                    for k,v in original_teacher.items()),
            "S3BT original graph+old teacher NOT physically frozen")
    require(sha_file(init)==S3BN_SHA and sha_file(bs_tokens)==S3BS_TOKEN_SHA
            and sha_file(parent)==S3BA_SHA,
            "old frozen or new TRAIN-only physical source SHA changed after fit")

    out.mkdir(parents=True,exist_ok=False)
    raw_path=out/"NRA2_S3BT_RAW_JOINT_RELATION_TRAIN2560_FP16.safetensors"
    raw={"bridge."+k:v.detach().cpu().half().contiguous()
         for k,v in model.state_dict().items()}
    raw["normalizer.mean"]=mean.half().contiguous()
    raw["normalizer.scale"]=scale.half().contiguous()
    save_file(raw,str(raw_path))
    raw_reload=load_file(str(raw_path),device="cpu")
    require(set(raw_reload)==set(weights)
            and all(torch.equal(weights[k],v) for k,v in raw_reload.items()
                    if k.startswith("bridge.graph.") or
                       k.startswith("bridge.query.base.") or
                       k.startswith("normalizer.")),
            "raw physical weights changed original frozen graph, S3BA teacher, normalizer")
    raw_sha=sha_file(raw_path)
    candidate=S3BMGraphWithJointRoles().eval()
    candidate.requires_grad_(False)
    scales=(1.,.75,.5,.25,.125,.0625,.03125,.015625)
    trials=[]
    accepted=None
    for alpha in scales:
        blended={k:v.detach().clone().contiguous() for k,v in raw_reload.items()}
        for k in blended:
            if (k.startswith("bridge.query.")
                    and not k.startswith("bridge.query.base.")
                    and k!="bridge.query.residual_scale"):
                blended[k]=(weights[k].float()+(raw_reload[k].float()-weights[k].float())*alpha).half().contiguous()
        candidate.load_state_dict({k.removeprefix("bridge."):v
                                   for k,v in blended.items() if k.startswith("bridge.")},
                                  strict=True)
        with torch.inference_mode():
            candidate_metrics=[metrics(candidate,g,rows,selector)
                               for g,rows,selector in zip(inputs,source_rows,selectors)]
        nets=[m["all"]["both_correct"]-orig["all"]["both_correct"]
              for m,orig in zip(candidate_metrics,baseline_metrics)]
        guard=all(n>=0 for n in nets[:6])
        novel=nets[6]
        trials.append({"alpha":alpha,"old_six_TRAIN_replay_nets":nets[:6],
                       "new_S3BR_TRAIN512_net":novel,"historical_replay_safe":guard})
        if guard and novel>0:
            accepted=(alpha,blended,candidate_metrics,nets)
            break
    print(json.dumps({"S3BT_TRAIN_only_fixed_alpha_grid":trials},sort_keys=True),flush=True)
    require(accepted is not None,
            "new semantic TRAIN512 not improved with 6 old groups replay-safe: fail closed")
    alpha,safe_weights,new_metrics,nets=accepted
    safe_file=out/"NRA2_S3BT_SAFE_JOINT_RELATION_TRAIN2560_FP16.safetensors"
    save_file(safe_weights,str(safe_file))
    reread=load_file(str(safe_file),device="cpu")
    require(set(reread)==set(weights)
            and all(torch.equal(reread[k],weights[k]) for k in weights
                    if k.startswith("bridge.graph.") or k.startswith("bridge.query.base.")
                    or k.startswith("normalizer."))
            and all(torch.equal(safe_weights[k],v) for k,v in reread.items()),
            "S3BT safe physical FP16 bitwise reload or graph/old query drift")
    safe_sha=sha_file(safe_file)
    replay={g:{"S3BN_parent":orig["all"]["both_correct"],
               "S3BT_research":new["all"]["both_correct"],
               "net":v} for g,orig,new,v in
            zip(group_names,baseline_metrics,new_metrics,nets)}
    report={
       "stage":"NRA2_S3BT_SEMANTIC_DIRECTIONAL_TRAIN2560_ONLY",
       "S3BN_initialized_joint_pairwise_fp16_sha256":S3BN_SHA,
       "S3BR_newTRAIN512_preregistered_commitment_sha256":S3BR_SHA,
       "S3BS_genuine_newTRAIN512_CPU_success_run":S3BS_RUN,
       "S3BS_newTRAIN512_original43m_cpu_fp16_sha256":S3BS_TOKEN_SHA,
       "S3BA_original_graph_and_old_role_head_sha256":S3BA_SHA,
       "train_count":2560,"historical_six_TRAIN_count":2048,"new_S3BR_TRAIN_count":512,
       "new_S3BR_grammar_styles":8,
       "original_S194_616_J7_unchanged_in_all_seven_CPU_sources":True,
       "new_S3BM_only_pair_params_trained":42869,
       "old_graph_and_original_S3BA_query_frozen_physical_FP16_equal":True,
       "original_normalizer_frozen_physical_FP16_equal":True,
       "optimizer_steps":STEPS,"seed":SEED,"learning_rate":LR,
       "GROUP_WEIGHTS":list(GROUP_WEIGHTS),
       "train_loss_before":float(before),"train_loss_after":float(after),
       "teacher_kl_before":float(kl_before),"teacher_kl_after":float(kl_after),
       "new_relation_symmetry_before":float(sym_before),
       "new_relation_symmetry_after":float(sym_after),
       "physical_fp16_raw_sha256":raw_sha,
       "physical_fp16_safe_sha256":safe_sha,
       "train_only_selected_alpha":alpha,
       "train_only_trust_region_trials":trials,
       "all_six_historical_TRAIN_replay_guard":replay,
       "train_7_group_before":dict(zip(group_names,baseline_metrics)),
       "train_7_group_after":dict(zip(group_names,new_metrics)),
       "training_trace":trace,
       "old_independent_S3AQ_EVAL128_consumed_never_used":True,
       "old_independent_S3BJ_EVAL128_consumed_never_used":True,
       "old_independent_S3BO_EVAL128_consumed_never_used":True,
       "no_heldout_DEV_or_EVAL_training_scoring_or_selection":True,
       "requires_separate_fourth_independent128_before_generalization_claim":True,
       "candidate_accepted":False,"output_gate":0.,
       "full_EOS_new_candidate_tested":False,"all_in_bpp_tested":False,
       "paired_CPU_speed_tested":False,
       "no_Jessica_LIVE_Render_Supabase_APK_main_QA_TEST_changes":True,
    }
    (out/"NRA2_S3BT_DIRECTIONAL_TRAIN2560_AUDIT.json").write_text(
        json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report[k] for k in (
         "stage","train_count","physical_fp16_safe_sha256",
         "train_only_selected_alpha","all_six_historical_TRAIN_replay_guard",
         "train_loss_before","train_loss_after","candidate_accepted","output_gate")},
         sort_keys=True),flush=True)

if __name__=="__main__":
    main()
