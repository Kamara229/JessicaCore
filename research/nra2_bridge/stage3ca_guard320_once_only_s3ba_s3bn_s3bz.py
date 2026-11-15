"""NRA2 S3CA: read-only ONCE-ONLY S3BX GUARD320 diagnostics after S3BZ SHA lock.

Only reuse 320 S3BR TRAIN-source prompts excluded from S3BZ FIT192. The
independent EVAL128 sets #1..#4 are CONSUMED, and NEVER accessed. Since older
S3BT used all 512 S3BR questions, these guards are NOT externally blind
generalization evidence, cannot select an alpha, a model, or a threshold.
Scores may diagnose weaknesses; no training, updates or candidate acceptance.
All forward inputs contain original Calibri43M frozen CPU tokens + no-gold masks.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

from research.nra2_bridge.stage3bx_blocked_family_topology_protocol import (
    split_data, protocol, PRE_S3BR_PARENT_SHA, CONSUMED_EVAL_RUNS,
)
from research.nra2_bridge.stage3br_relation_language_train512_prereg import (
    build_rows as full_s3br_rows,
)
from research.nra2_bridge.stage3bz_fit192_pairwise_replay2240 import (
    bind, require,
)
from research.nra2_bridge.stage3bm_joint_semantic_pair_role_head import (
    S3BMGraphWithJointRoles,
)
from research.nra2_bridge.stage3o_neural_query_roles import LearnedQueryRoleHead
from research.nra2_bridge.stage3k_train_real_entity_binding import (
    sha_file, exact_original_tokenizer,
)
from research.nra2_bridge.stage3f_token_attention import normalize

STAGE="NRA2_S3CA_PREVIOUSLY_USED_TRAIN_ONLY_GUARD320_ONCE_DIAGNOSTIC"
SAFE_S3BZ_SHA="6084061cb1f9ea46ef3b504179bee09b392f4427ac52ec58d768645da98872ee"
PARENT_S3BA_SHA="4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
S3BS_TOKEN_SHA="95912db37d26077f4abb2e1e4050aad13155c3988aa7ee592e6da8d81c3eec5c"
S3BX_SHA="f020319c9cfaa39becb03dca1aed90debf25632a90737707b0c1ff1aaa7ce6ee"
GUARDS=("grammar_guard","topology_guard","double_guard")
EXPECTED={"grammar_guard":192,"topology_guard":64,"double_guard":64}


def pairwise_correct(baseline, candidate):
    x, y=baseline.bool(), candidate.bool()
    both=int((x&y).sum())
    old_only=int((x&~y).sum())
    new_only=int((~x&y).sum())
    neither=int((~x&~y).sum())
    require(both+old_only+new_only+neither==len(x),"paired guard length drift")
    n=old_only+new_only
    p=min(1.0,2.0*sum(math.comb(n,i) for i in range(min(old_only,new_only)+1))
          /2**n) if n else 1.0
    return dict(both_correct=both,old_only_correct=old_only,
                new_only_correct=new_only,both_wrong=neither,
                net_new_minus_old=new_only-old_only,
                exact_two_sided_mcnemar_p=p)


def read_model(path,sha):
    require(sha_file(path)==sha,"pinned physically frozen FP16 SHA changed")
    return load_file(str(path),device="cpu")


def query_head(weights):
    m=S3BMGraphWithJointRoles().eval().requires_grad_(False)
    m.load_state_dict({k.removeprefix("bridge."):v for k,v in weights.items()
                       if k.startswith("bridge.")},strict=True)
    return m


def run():
    require(len(sys.argv)==6,"S3CA expects four artifact dirs + NEW output dir")
    bs,ba,bn,bz,out=map(Path,sys.argv[1:])
    require(not out.exists(),"immutable once-only guard output must not exist")
    torch.set_num_threads(2)
    source=full_s3br_rows()
    groups=split_data()
    require(protocol(groups)["commitment_sha256"]==S3BX_SHA,
            "TRAIN-only S3BX grammar/topology partition drift")
    require(sum(len(groups[k]) for k in GUARDS)==320
            and {k:len(groups[k]) for k in GUARDS}==EXPECTED,
            "S3BX guard320 row counts changed")
    full={r.prompt_sha256:i for i,r in enumerate(source)}
    require(len(full)==512,"full original S3BR 512 prompt hashes changed")
    require(not set(r.prompt_sha256 for r in groups["fit"]) &
            set(r.prompt_sha256 for k in GUARDS for r in groups[k]),
            "S3CA leaked fitting prompts into diagnostic guards")

    tokfile=bs/"NRA2_S3BS_REAL_CPU_TRAIN512_ALL_TOKENS_FP16.safetensors"
    goldfile=bs/"NRA2_S3BS_TRAIN512_STRUCTURAL_LABELS.safetensors"
    bs_audit=json.loads((bs/"NRA2_S3BS_TRAIN512_REAL_CPU_J7_AUDIT.json").read_text())
    require(bs_audit["full_token_fp16_sha256"]==S3BS_TOKEN_SHA
            and bs_audit["physical_graph_labels_sha256"]==sha_file(goldfile)
            and bs_audit["all_616_token_digest_matches_S194"]
            and bs_audit["physical_reload_bitwise_equal"],
            "genuine original 43M CPU TRAIN512 and 616 original J7 drift")
    token_pack=read_model(tokfile,S3BS_TOKEN_SHA)
    labels=load_file(str(goldfile),device="cpu")
    sba=read_model(ba/"NRA2_S3BA_S3AW_INIT_ROLE1536_FP16.safetensors",PARENT_S3BA_SHA)
    sbn=read_model(bn/"NRA2_S3BN_JOINT_ROLE_SAFE_TRAIN2048_FP16.safetensors",PRE_S3BR_PARENT_SHA)
    sbz=read_model(bz/"NRA2_S3BZ_SAFE_JOINT_RELATION_TRAIN2240_FP16.safetensors",SAFE_S3BZ_SHA)
    frozen_keys=[k for k in sbn if
                 k.startswith("bridge.graph.") or
                 k.startswith("bridge.query.base.") or k.startswith("normalizer.")]
    require(all(torch.equal(sbn[k],sbz[k]) for k in frozen_keys),
            "S3BZ changed S3BN original graph, query teacher or normalizer")
    require(all(torch.equal(sba[k],sbn["bridge.query.base."+k.removeprefix("bridge.query.")])
                for k in sba if k.startswith("bridge.query.")),
            "S3BN baseline does not match physically original S3BA")
    require(all(torch.equal(sba[k],sbn[k]) for k in sba
                if k.startswith("bridge.graph.") or k.startswith("normalizer.")),
            "original teacher graph / normalizer changed")

    bhead=LearnedQueryRoleHead().eval().requires_grad_(False)
    bhead.load_state_dict({k.removeprefix("bridge.query."):v for k,v in sba.items()
                           if k.startswith("bridge.query.")},strict=True)
    parent=query_head(sbn)
    candidate=query_head(sbz)
    tokenizer=exact_original_tokenizer()
    mean=sbn["normalizer.mean"].float()
    scale=sbn["normalizer.scale"].float()
    subsets={}
    for name in GUARDS:
        rows=groups[name]
        indices=[full[r.prompt_sha256] for r in rows]
        require(len(set(indices))==len(rows)==EXPECTED[name],"guard indices incorrect")
        tok=token_pack["token_hidden320"][indices]
        mask=token_pack["token_mask"][indices]
        lengths=token_pack["token_lengths"][indices]
        q,_=bind(tokenizer,[r.question for r in rows],tok,mask,lengths)
        t=normalize(tok,mean,scale,mask)
        src=labels["source"][indices].argmax(1).long()
        dst=labels["target"][indices].argmax(1).long()
        require(torch.equal(torch.sort(q.node_order,dim=1).values,
                            torch.sort(torch.stack((src,dst),dim=1),dim=1).values),
                "diagnostic role gold and no-gold literal mask disagree")
        with torch.inference_mode():
            results={}
            for model_name,head in (("S3BA",bhead),("S3BN",parent.query),
                                    ("S3BZ",candidate.query)):
                logits_src,logits_dst=head(t,mask,q)
                right=(logits_src.argmax(1)==src)&(logits_dst.argmax(1)==dst)
                results[model_name]=right
        part={"n":len(rows),"yes":sum(r.reachable for r in rows),
              "source_first":sum(r.query_style%2==0 for r in rows)}
        part["both_roles_correct"]={name_:int(val.sum()) for name_,val in results.items()}
        part["S3BA_vs_S3BZ"]=pairwise_correct(results["S3BA"],results["S3BZ"])
        part["S3BN_vs_S3BZ"]=pairwise_correct(results["S3BN"],results["S3BZ"])
        subsets[name]=part
    report={
        "stage":STAGE,"S3BX_protocol_sha256":S3BX_SHA,
        "safe_new_S3BZ_FP16_sha256":SAFE_S3BZ_SHA,
        "parent_S3BN_SHA256":PRE_S3BR_PARENT_SHA,
        "old_S3BA_SHA256":PARENT_S3BA_SHA,
        "S3BS_true_43M_CPU_token_sha256":S3BS_TOKEN_SHA,
        "all_four_real_independent_EVAL128_permanently_consumed":list(CONSUMED_EVAL_RUNS),
        "source_S3BR_TRAIN512_previously_used_by_rejected_S3BT":True,
        "S3BZ_fit192_tokens_inference_scoring_and_training_excluded_this_step":True,
        "three_guard_groups_NEVER_used_for_candidate_training_alpha_or_selection":True,
        "diagnostic_guard_is_NOT_independent_evidence_of_generalization":True,
        "fifth_independent_external128_required_for_future_quality_claim":True,
        "model_weights_frozen_no_gradient_no_new_CPU_no_NEW_eval":True,
        "original_Calibri_GreenTunnel_NRA_and_616_J7_unchanged":True,
        "candidate_accepted":False,"output_gate":0.0,
        "subsets":subsets,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3CA_GUARD320_ONCE_DIAGNOSTIC_NO_RETUNE.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({"stage":STAGE,"subsets":subsets,
                      "candidate_accepted":False,"output_gate":0.0},sort_keys=True),flush=True)


if __name__=="__main__":
    run()
