"""NRA2 S3X: TRAIN-only reversed-mention grammar precommit, no CPU or training.

This is a targeted post-hoc *remediation training* set motivated by S3V/S3W
failure; it is NOT a blind benchmark or an independent new research result.
The 32 S3M TRAIN graph topologies are intentionally reused, to isolate
new wording while preserving existing source/target and graph semantics.
The original S3T 64 scored questions are excluded from TRAIN.
"""
from __future__ import annotations
import hashlib
import json
from collections import defaultdict
from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    build_rows as s3m_rows,GraphPrompt,FAMILIES,QUERY_STYLES as S3M_STYLES,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import (
    build_rows as s3t_rows,QUERY_STYLES as S3T_STYLES,query_char_offsets,
)
from research.nra2_bridge.stage3q_independent_soft_hard_prereg import (
    build_rows as s3q_rows,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle

GROUP="S3X_TRAIN128_NEWGRAMMAR_REVERSE_ORDER"
COUNT=128
GRAPH_COUNT=32
S3N_SUCCESS_RUN=43522555326
S3O_SUCCESS_RUN=43522663612
S3V_SUCCESS_RUN=43529457952
S3W_SUCCESS_RUN=43529926564
S3T_HELDOUT_SHA="034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"
S3O_FROZEN_SHA="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
# Two DST-first and two SRC-first; unlike the previous S3M patterns, no
# query style allows a shortcut to the reachability yes/no label.
NEW_STYLES=(
    "With {dst} as the intended finish, could a directed route originate at {src}?",
    "For a trip setting off from {src}, is {dst} among the vertices one can reach?",
    "Looking toward endpoint {dst}, can one trace a one-way path back to origin {src}?",
    "Taking {src} as the initial station, can the path end at {dst}?",
)
REVERSED_STYLES=(0,2)
PREVIOUS_STYLES=tuple(S3M_STYLES)+tuple(S3T_STYLES)


def build_rows():
    old=[r for r in s3m_rows() if r.split=="TRAIN128"]
    if len(old)!=COUNT or len(NEW_STYLES)!=4:
        raise AssertionError("S3X original TRAIN128 or new-grammar count drift")
    out=[]
    for r in old:
        qmark=r.question.index("?")
        dot=r.question.rfind(". ",0,qmark)
        if dot<0:
            raise AssertionError("S3X unexpected native S3M prompt delimiter")
        opening=r.question[:dot+2]
        closing=r.question[qmark+1:]
        q=NEW_STYLES[r.query_style].format(src=f"R{r.src}",dst=f"R{r.dst}")
        out.append(GraphPrompt(
            graph_id=r.graph_id,split=GROUP,family=r.family,
            active_nodes=r.active_nodes,edges=r.edges,src=r.src,dst=r.dst,
            reachable=r.reachable,query_style=r.query_style,
            question=opening+q+closing))
    validate(out)
    return out


def validate(rows):
    old=s3m_rows()
    train=[r for r in old if r.split=="TRAIN128"]
    unseen=s3t_rows()
    known=s3q_rows()
    if len(rows)!=COUNT or len({r.prompt_sha256 for r in rows})!=COUNT:
        raise AssertionError("S3X 128 unique TRAIN grammar questions required")
    if set(NEW_STYLES)&set(PREVIOUS_STYLES):
        raise AssertionError("S3X query template overlaps old or already-scored S3T")
    prior_hashes={r.prompt_sha256 for r in old}
    prior_hashes.update(r.prompt_sha256 for r in unseen)
    prior_hashes.update(r.prompt_sha256 for r in known)
    if {r.prompt_sha256 for r in rows}&prior_hashes:
        raise AssertionError("S3X previously seen question SHA overlaps")
    by_graph=defaultdict(list)
    for r,t in zip(rows,train):
        if (r.split!=GROUP or r.family not in FAMILIES
            or (r.graph_id,r.edges,r.src,r.dst,r.reachable,r.query_style)!=(t.graph_id,t.edges,t.src,t.dst,t.reachable,t.query_style)
            or oracle(r.edges,r.src,r.dst)!=r.reachable):
            raise AssertionError("S3X reused graph semantic labels drift")
        offsets=query_char_offsets(r.question)
        if (set(offsets["text_order_nodes"])!={r.src,r.dst}
            or (offsets["text_order_nodes"][0]==r.dst)!=(r.query_style in REVERSED_STYLES)):
            raise AssertionError("S3X semantic role/order binding drift")
        by_graph[r.graph_id].append(r)
    if len(by_graph)!=GRAPH_COUNT or sum(r.reachable for r in rows)!=64:
        raise AssertionError("S3X graph count or global label imbalance")
    for group in by_graph.values():
        if (len(group)!=4 or {r.query_style for r in group}!={0,1,2,3}
            or sum(r.reachable for r in group)!=2
            or len({r.topology_key for r in group})!=1):
            raise AssertionError("S3X per graph 4 controlled wording labels drift")
    for family in FAMILIES:
        fam=[r for r in rows if r.family==family]
        if len(fam)!=32 or sum(r.reachable for r in fam)!=16:
            raise AssertionError("S3X per graph family truth imbalance")
        for style in range(4):
            sub=[r for r in fam if r.query_style==style]
            if len(sub)!=8 or sum(r.reachable for r in sub)!=4:
                raise AssertionError("S3X family-wording shortcut")
    for style in range(4):
        for n in (4,5,6):
            sub=[r for r in rows if r.query_style==style and r.active_nodes==n]
            if not sub or sum(r.reachable for r in sub)*2!=len(sub):
                raise AssertionError("S3X length-style truth shortcut")
        for header in ("Connection log:", "One-way route list:"):
            sub=[r for r in rows if r.query_style==style and r.question.startswith(header)]
            if not sub or 2*sum(r.reachable for r in sub)!=len(sub):
                raise AssertionError("S3X header-style truth shortcut")
    return True


def commitment(rows):
    validate(rows)
    prior_s3t=s3t_rows()
    audit={
        "stage":"NRA2_S3X_REVERSAL_ROLE_TRAIN128_GRAMMAR_PREREG_ONLY",
        "group":GROUP,"train_count":COUNT,
        "unique_train_graphs":GRAPH_COUNT,
        "TRAIN_reuses_original_S3M_32_TRAIN_graphs":True,
        "TRAIN_has_64_yes_and_64_no":True,
        "four_novel_wordings":True,
        "two_destination_first_two_source_first":True,
        "original_S3T64_excluded_from_new_TRAIN":True,
        "original_S3T64_seen_during_failure_analysis_not_blind":True,
        "S3T_64_prior_scored_sha256":S3T_HELDOUT_SHA,
        "S3T_64_prior_prompt_sha256":[r.prompt_sha256 for r in prior_s3t],
        "S3W_successful_diagnosis_run":S3W_SUCCESS_RUN,
        "S3V_original_scoring_run":S3V_SUCCESS_RUN,
        "S3O_frozen_original_fp16_sha256":S3O_FROZEN_SHA,
        "original_S3N_frozen_CPU_run":S3N_SUCCESS_RUN,
        "new_training_not_started":True,
        "native_43M_CPU_for_new_TRAIN_not_yet_captured":True,
        "new_checkpoint_not_created":True,
        "original_weights_unchanged":True,
        "616_S194_J7_must_be_reverified_before_and_after_new_CPU":True,
        "original_bridge_and_NRA_must_be_frozen_and_replay_checked":True,
        "output_gate":0.0,"candidate_accepted":False,
        "no_LIVE_Jessica_Render_Supabase_QA_TEST_main_APK":True,
        "future_new_independent_final_blind_required":True,
        "new_train_sha256":[r.prompt_sha256 for r in rows],
        "query_char_offsets":[query_char_offsets(r.question) for r in rows],
        "new_templates":list(NEW_STYLES),
    }
    serialized=json.dumps(audit,sort_keys=True,separators=(",",":")).encode()
    audit["commitment_sha256"]=hashlib.sha256(serialized).hexdigest()
    return audit

if __name__=="__main__":
    report=commitment(build_rows())
    print(json.dumps({k:v for k,v in report.items()
                      if k not in ("new_train_sha256","query_char_offsets","S3T_64_prior_prompt_sha256")},
                     sort_keys=True),flush=True)
