"""S3AT: post-score read-only role-order failure diagnostic on scored S3AQ DEV128.

Strictly no EVAL128 CPU prefill, inference or training. Do not modify any
model checkpoint; original 616 S194 CPU J7 and output gate zero protected.
This follows SUCCESS S3AS CI 43581666491, whose DEV128 has already been
opened after S3AS checkpoint FP16 SHA lock. Characterize error by wording.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows, commitment, TRAIN, DEV, EVAL, REVERSED_STYLES, STYLES,
)
from research.nra2_bridge.stage3as_s3z_init_train_dev_roles import (
    S3AR_SHA, S3AQ_COMMIT_SHA, S3Z_SHA, S3AR_RUN, S3Z_RUN,
    bind, gold_roles, require,
)
from research.nra2_bridge.stage3o_neural_query_roles import S3KGraphWithLearnedRoles
from research.nra2_bridge.stage3k_train_real_entity_binding import exact_original_tokenizer, sha_file
from research.nra2_bridge.stage3f_token_attention import normalize

S3AS_RUN=43581666491
S3AS_SHA="06420e22270dd082116086a5174e95c04e9ff2b52664583bc8d5a1b4a074d488"
BASE="NRA2_S3AS_S3Z_INIT_TRAIN512_DEV128_ROLE_AUDIT.json"
EXACT_STAGE="NRA2_S3AS_S3Z_INITIALIZED_ROLE512_TRAIN_DEV128_EXPLORATORY_FROZEN_GRAPH"


def role_details(checkpoint,token,mask,q,src,dst,rows):
    weights=load_file(str(checkpoint),device="cpu")
    net=S3KGraphWithLearnedRoles().eval()
    net.load_state_dict({k.removeprefix("bridge."):v for k,v in weights.items()
                         if k.startswith("bridge.")},strict=True)
    net.requires_grad_(False)
    x=normalize(token,weights["normalizer.mean"].float(),
                weights["normalizer.scale"].float(),mask)
    with torch.inference_mode():
        sl,tl=net.query(x,mask,q)
    pred_s=sl.argmax(1)
    pred_d=tl.argmax(1)
    good=(pred_s==src)&(pred_d==dst)
    swapped=(pred_s==dst)&(pred_d==src)
    require(bool((good|swapped).all()) and not bool((good&swapped).any()),
            "two literal query mentions must yield only exact/crossed role assignments")
    out=[]
    for i,row in enumerate(rows):
        out.append({
            "style":int(row.query_style),
            "family":row.family,
            "positive":int(row.reachable),
            "graph_id":row.graph_id,
            "correct":bool(good[i]),
            "swapped":bool(swapped[i]),
            "destination_first":row.query_style in REVERSED_STYLES,
            "predicted_source_is_text_first":int(pred_s[i])==int(q.node_order[i,0]),
            "true_source_is_text_first":int(src[i])==int(q.node_order[i,0]),
        })
    return out


def group_stats(rows,attr):
    groups=defaultdict(list)
    for r in rows:
        groups[str(r[attr])].append(r)
    return {k:{"count":len(v),"correct":sum(r["correct"] for r in v),
               "wrong_role_swap":sum(r["swapped"] for r in v),
               "correct_positive":sum(r["correct"] and r["positive"] for r in v),
               "correct_negative":sum(r["correct"] and not r["positive"] for r in v),
               "text_first_source_predicted":sum(r["predicted_source_is_text_first"] for r in v),
               "text_first_source_true":sum(r["true_source_is_text_first"] for r in v)}
            for k,v in sorted(groups.items())}


def main():
    if len(sys.argv)!=5:
        raise SystemExit("S3AT S3AR_CPU384_DIR S3Z_LOCKED_DIR S3AS_LOCKED_DIR NEW_OUT")
    ar_dir,z_dir,as_dir,out=map(Path,sys.argv[1:])
    require(not out.exists() or not any(out.iterdir()),"refuse overwrite research artifacts")
    audit_ar=json.loads((ar_dir/"NRA2_S3AR_TRAIN256_DEV128_REAL_CPU_J7_AUDIT.json").read_text())
    audit_as=json.loads((as_dir/BASE).read_text())
    ar_t=ar_dir/"NRA2_S3AR_REAL_CPU_TRAIN256_DEV128_ALL_TOKENS_FP16.safetensors"
    ar_y=ar_dir/"NRA2_S3AR_TRAIN256_DEV128_STRUCTURAL_LABELS.safetensors"
    z_fp=z_dir/"NRA2_S3Z_ROLE_REVERSAL_REHEARSAL256_FP16.safetensors"
    as_fp=as_dir/"NRA2_S3AS_S3Z_INIT_ROLE512_FP16.safetensors"
    require(audit_as["stage"]==EXACT_STAGE
            and audit_ar["graph_prefill_count"]==384
            and audit_ar["group_sizes"]=={TRAIN:256,DEV:128}
            and audit_ar["S3AQ_registered_commitment_sha256"]==S3AQ_COMMIT_SHA
            and audit_ar["all_616_token_digest_matches_S194"] is True
            and audit_ar["S3AQ_future_EVAL128_never_prefilled_scored_or_trained"] is True
            and audit_ar["full_token_fp16_sha256"]==S3AR_SHA==sha_file(ar_t)
            and audit_ar["physical_graph_labels_sha256"]==sha_file(ar_y)
            and audit_as["S3AR_real_cpu_successful_run"]==S3AR_RUN
            and audit_as["S3Z_parent_training_successful_run"]==S3Z_RUN
            and audit_as["S3AQ_future_EVAL128_not_prefilled_scored_or_trained"] is True
            and audit_as["DEV128_scored_only_after_FP16_SHA_commit"] is True
            and audit_as["DEV128_excluded_from_gradient_optimizer"] is True
            and audit_as["new_S3AS_physical_fp16_sha256"]==S3AS_SHA==sha_file(as_fp)
            and audit_as["S3Z_parent_fp16_unchanged_sha256"]==S3Z_SHA==sha_file(z_fp)
            and audit_as["original_graph_physical_fp16_bitwise_equal"] is True
            and audit_as["output_gate"]==0.0
            and audit_as["candidate_accepted"] is False,
            "pinned S3AR S3Z S3AS SHA, protected J7, original DEV-only split drift")
    allrows=build_rows()
    locked=commitment(allrows)
    require(locked["commitment_sha256"]==S3AQ_COMMIT_SHA
            and len(allrows)==512,"S3AQ prereg drift")
    devrows=[r for r in allrows if r.split==DEV]
    require(len(devrows)==128 and len({r.graph_id for r in devrows})==32
            and [r.prompt_sha256 for r in devrows] ==
              audit_ar["split_groups"][DEV]["prompt_sha256"],
            "DEV128 exact original prompt order/topology drift")
    tok=load_file(str(ar_t),device="cpu")
    lab=load_file(str(ar_y),device="cpu")
    dt=tok["token_hidden320"][256:384]
    dm=tok["token_mask"][256:384]
    dl=tok["token_lengths"][256:384]
    source,target=gold_roles(lab,256,384)
    q,_=bind(exact_original_tokenizer(),[r.question for r in devrows],dt,dm,dl)
    require(bool(torch.equal(q.node_order.sort(1).values,
                torch.stack([source,target],dim=-1).sort(1).values)),
            "DEV128 role mentions/targets misaligned")
    old=role_details(z_fp,dt,dm,q,source,target,devrows)
    new=role_details(as_fp,dt,dm,q,source,target,devrows)
    old_total=sum(a["correct"] for a in old)
    new_total=sum(a["correct"] for a in new)
    require(old_total==audit_as["DEV128_paired_S3Z_baseline"]["all"]["both_correct"]
            and new_total==audit_as["DEV128_paired_S3AS_candidate"]["all"]["both_correct"]
            and old_total==37 and new_total==64,
            "read-only S3AT diagnostic not consistent with pinned S3AS DEV score")
    flips={"old_correct_new_wrong":0,"old_wrong_new_correct":0,
           "both_correct":0,"both_wrong":0}
    for before,after in zip(old,new):
        require(before["graph_id"]==after["graph_id"]
                and before["style"]==after["style"],"paired DEV order drift")
        key=("both_correct" if before["correct"] and after["correct"] else
             "both_wrong" if not before["correct"] and not after["correct"] else
             "old_correct_new_wrong" if before["correct"] else
             "old_wrong_new_correct")
        flips[key]+=1
    graph_groups=defaultdict(list)
    for r in new:
        graph_groups[r["graph_id"]].append(r)
    quartet={str(n):sum(sum(r["correct"] for r in q)==n for q in graph_groups.values())
             for n in range(5)}
    report={
        "stage":"NRA2_S3AT_POSTSCORE_DEV128_ROLE_ERROR_DIAG_READONLY",
        "S3AS_successful_run":S3AS_RUN,
        "S3AQ_registered_commitment_sha256":S3AQ_COMMIT_SHA,
        "S3AR_original_CPU_384_frozen_616_J7":True,
        "S3AQ_future_EVAL128_never_prefilled_scored_or_trained":True,
        "scored_DEV_only_not_sealed_blind":True,
        "S3Z_parent_fp16_sha256":S3Z_SHA,
        "S3AS_candidate_fp16_sha256":S3AS_SHA,
        "DEV128_baseline_both":old_total,
        "DEV128_candidate_both":new_total,
        "DEV128_paired_flips":flips,
        "S3Z_by_style":group_stats(old,"style"),
        "S3AS_by_style":group_stats(new,"style"),
        "S3Z_by_family":group_stats(old,"family"),
        "S3AS_by_family":group_stats(new,"family"),
        "S3AS_by_truth":group_stats(new,"positive"),
        "S3AS_correct_per_graph_quartet":quartet,
        "S3AQ_DEV_style_templates":list(STYLES[DEV]),
        "output_gate":0.0,"candidate_accepted":False,
        "new_weights_or_optimizer_steps":0,
        "no_original_LIVE_Render_Supabase_APK_main_QA_TEST_changes":True,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3AT_DEV128_ROLE_ERROR_DIAGNOSTIC.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:report[k] for k in (
        "stage","DEV128_baseline_both","DEV128_candidate_both",
        "DEV128_paired_flips","S3Z_by_style","S3AS_by_style",
        "S3AS_by_truth","S3AS_correct_per_graph_quartet")},
        sort_keys=True),flush=True)


if __name__=="__main__":
    main()
