"""NRA2 S3Q: precommit future no-fitting topology-disjoint soft/hard trial.

This creates an *independent research diagnostic*, not a secrecy-sealed
final QA benchmark. It remains unseen by S3K/S3O training and no S3Q
model scores may be inspected until native frozen 43M CPU capture and
FP16 checksum verification. Candidate weights and hardening threshold
must be pinned before the first score.

Scope fixed before evaluation:
 - 16 disjoint directed graphs (4 per family), 64 questions.
 - Two reachability yes/two no per graph, all four source/target
   wording variants, 32 yes/32 no and header+wording counterbalance.
 - EXACT original S3O FP16, S3N frozen 43M CPU profile and independent
   frozen 2241-parameter NRA, with no finetune or threshold tuning.
 - Compare ORIGINAL soft edge probabilities to predetermined >0.5
   hardening, with original soft endpoints in both arms.
 - Report all 64 overall, by graph family and by true yes/no,
   not just a favorable overall score.
 - Retain original J7 616 and output gate 0.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import random
from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    GraphPrompt,FAMILIES,QUERY_STYLES,HEADERS,make_graph,graph_has_cycle,
    build_rows as older_s3m,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import (
    build_rows as older_s3g,oracle,
)

SEED=2026111417
NODES=6
GRAPHS_PER_FAMILY=4
TOTAL_GRAPHS=16
TOTAL_PROMPTS=64
FROZEN_THRESHOLD=0.5
S3O_FP16_SHA256="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3N_FP16_SHA256="b8a375982e556d67dfd5f741050c18c5460249184da9c3fcebc3d174d80db81b"
S3O_RUN=43522663612
S3N_RUN=43522555326
S3P_RUN=43524516584
GROUP="S3Q_INDEPENDENT64"
# Precommitted lengths: each pair of indices shares its graph length.
# Truth direction still alternates within each length and fixed wording.
LENGTHS=(
    (4,4,5,5),  # chain
    (5,5,6,6),  # fork
    (6,6,4,4),  # cycle with one-way sink
    (4,4,6,6),  # diamond
)


def build_rows():
    rng=random.Random(SEED)
    prior=older_s3m()
    used={r.topology_key for r in prior}
    used.update(tuple(sorted(r.edges)) for r in older_s3g())
    old_prompts={r.prompt_sha256 for r in prior}
    old_prompts.update(r.prompt_sha256 for r in older_s3g())
    rows=[]
    for fi,family in enumerate(FAMILIES):
        for j,n in enumerate(LENGTHS[fi]):
            for _ in range(12000):
                order=tuple(rng.sample(range(NODES),n))
                edges=make_graph(order,family)
                key=tuple(sorted(edges))
                possible=[
                    (a,b) for a in order for b in order if a!=b
                    and oracle(edges,a,b) and not oracle(edges,b,a)
                ]
                if key not in used and possible:
                    used.add(key)
                    break
            else:
                raise AssertionError("S3Q independent directed topology exhausted")
            forward=rng.choice(possible)
            backward=(forward[1],forward[0])
            for style,template in enumerate(QUERY_STYLES):
                yes=(style%2)==(j%2)
                src,dst=forward if yes else backward
                if oracle(edges,src,dst)!=yes:
                    raise AssertionError("S3Q expected path orientation disagrees with BFS oracle")
                links=[f"R{a} -> R{b}" for a,b in edges]
                rng.shuffle(links)
                wording=template.format(src=f"R{src}",dst=f"R{dst}")
                # Header cannot act as an answer shortcut: both j in
                # each same-length pair get the same header per style.
                header=HEADERS[(j//2+style)%len(HEADERS)]
                question=header.format(edges="; ".join(links),query=wording)
                rows.append(GraphPrompt(
                    graph_id=f"S3Q-{family}-{j:02}",
                    split=GROUP,family=family,active_nodes=n,edges=edges,
                    src=src,dst=dst,reachable=yes,query_style=style,
                    question=question,
                ))
    validate(rows,prior_sha=old_prompts)
    return rows


def validate(rows,prior_sha=None):
    if len(rows)!=TOTAL_PROMPTS:
        raise AssertionError("S3Q requires exactly 64 new frozen public prompts")
    hashes=[r.prompt_sha256 for r in rows]
    if len(set(hashes))!=TOTAL_PROMPTS:
        raise AssertionError("S3Q duplicated prompt")
    prior=older_s3m()
    prior_keys={r.topology_key for r in prior}
    prior_keys.update(tuple(sorted(r.edges)) for r in older_s3g())
    known_sha={r.prompt_sha256 for r in prior}
    known_sha.update(r.prompt_sha256 for r in older_s3g())
    if prior_sha:
        known_sha.update(prior_sha)
    if known_sha.intersection(hashes):
        raise AssertionError("S3Q prior S3G/S3M prompt leakage")
    graphs=defaultdict(list)
    for r in rows:
        if r.split!=GROUP or r.family not in FAMILIES:
            raise AssertionError("S3Q split or family drift")
        if r.topology_key in prior_keys:
            raise AssertionError("S3Q topology overlaps previously viewed or trained graphs")
        if r.active_nodes!=len({v for pair in r.edges for v in pair}):
            raise AssertionError("S3Q graph node count mismatch")
        if graph_has_cycle(r.edges)!=(r.family=="cycle_sink"):
            raise AssertionError("S3Q directed cycle family drift")
        if oracle(r.edges,r.src,r.dst)!=r.reachable:
            raise AssertionError("S3Q BFS oracle disagrees with frozen answer")
        graphs[r.graph_id].append(r)
    if (len(graphs)!=TOTAL_GRAPHS
        or len({r.topology_key for r in rows})!=TOTAL_GRAPHS
        or sum(r.reachable for r in rows)!=TOTAL_PROMPTS//2):
        raise AssertionError("S3Q disjoint graph count or yes/no balance invalid")
    for group in graphs.values():
        if (len(group)!=4 or sum(r.reachable for r in group)!=2
            or {r.query_style for r in group}!={0,1,2,3}
            or len({r.topology_key for r in group})!=1
            or len({(r.src,r.dst) for r in group})!=2):
            raise AssertionError("S3Q source-target counterbalanced graph pair drift")
        dirs={(r.src,r.dst) for r in group}
        if any((b,a) not in dirs for a,b in dirs):
            raise AssertionError("S3Q opposite query directions missing")
    for family in FAMILIES:
        subset=[r for r in rows if r.family==family]
        if len(subset)!=16 or sum(r.reachable for r in subset)!=8:
            raise AssertionError("S3Q family outcome balance drift")
        for style in range(4):
            same=[r for r in subset if r.query_style==style]
            if len(same)!=4 or sum(r.reachable for r in same)!=2:
                raise AssertionError("S3Q grammar shortcut to truth")
    for style in range(4):
        for header in HEADERS:
            prefix=header.partition("{edges}")[0]
            same=[r for r in rows if r.query_style==style and r.question.startswith(prefix)]
            if not same or sum(r.reachable for r in same)*2!=len(same):
                raise AssertionError("S3Q header shortcut to truth")
        for n in (4,5,6):
            same=[r for r in rows if r.query_style==style and r.active_nodes==n]
            if not same or sum(r.reachable for r in same)*2!=len(same):
                raise AssertionError("S3Q graph length shortcut to truth")
    return True


def commitment(rows):
    validate(rows)
    frozen={
        "stage":"NRA2_S3Q_PREDECLARED_INDEPENDENT_TOPOLOGY64_GRAPH_ONLY",
        "seed":SEED,"prompts":TOTAL_PROMPTS,"topologies":TOTAL_GRAPHS,
        "per_family":{family:16 for family in FAMILIES},
        "yes":32,"no":32,"output_gate":0.0,
        "S3O_frozen_sha256":S3O_FP16_SHA256,
        "S3N_previous_native_CPU_sha256":S3N_FP16_SHA256,
        "S3O_successful_prior_CI":S3O_RUN,
        "S3P_successful_prior_CI":S3P_RUN,
        "S3N_successful_native_CPU_CI":S3N_RUN,
        "predeclared_edge_rule":"hard_edge = (sigmoid(edge_logit) > 0.5).float()",
        "predeclared_role_rule":"source and target remain SOFT in both arms",
        "predeclared_metric":"signed binary NRA logit > 0, count correct by family and yes/no",
        "graph_topologies_disjoint_from_S3G_S3M":True,
        "query_wording_seen_in_prior_S3M":True,
        "new_43M_CPU_features_not_yet_captured":True,
        "S3Q_predictions_not_inspected":True,
        "NRA_model_not_retrained":True,"S3O_model_not_retrained":True,
        "candidate_accepted":False,"independent_final_heldout_passed":False,
        "native_new_FULL_EOS_not_tested":True,
        "all_in_physical_bpp_not_tested":True,
        "same_CPU_speed_not_tested":True,
        "no_LIVE_Jessica_Render_Supabase_APK_main_QA_TEST":True,
        "question_SHA256_in_order":[r.prompt_sha256 for r in rows],
        "topology_SHA256_in_order":[
            hashlib.sha256(json.dumps(sorted(r.edges)).encode()).hexdigest()
            for r in rows[::4]
        ],
    }
    serial=json.dumps(frozen,sort_keys=True,separators=(",",":")).encode()
    frozen["commitment_sha256"]=hashlib.sha256(serial).hexdigest()
    return frozen


if __name__=="__main__":
    data=commitment(build_rows())
    print(json.dumps({k:v for k,v in data.items()
                      if k not in ("question_SHA256_in_order","topology_SHA256_in_order")},
                     sort_keys=True),flush=True)
