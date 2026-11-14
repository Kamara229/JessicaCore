"""NRA2 S3AQ: fresh TRAIN256, research DEV128 and strictly BLIND future EVAL128.

Following S3AO independent EVAL128 (success but negative S3AM generalization),
build entirely disjoint graph topologies and 12 new grammar templates. Explicit
DEV split may support TRAIN-only model selection. Future EVAL remains strictly
unscored, unprefilled, and excluded from training/selection until candidate SHA
is frozen. No model inference, labels delivered to decoder, LIVE or QA/TEST.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import random
from pathlib import Path
import sys

from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    FAMILIES, HEADERS, GraphPrompt, make_graph, graph_has_cycle,
    QUERY_STYLES as S3M_STYLES,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle
from research.nra2_bridge.stage3aa_independent_newgrammar128_prereg import (
    previous_rows,build_rows as s3aa_rows,QUERY_STYLES as S3AA_STYLES,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets
from research.nra2_bridge.stage3x_reversal_train_prereg import NEW_STYLES as S3X_STYLES
from research.nra2_bridge.stage3t_unseen_wording_prereg import QUERY_STYLES as S3T_STYLES
from research.nra2_bridge.stage3ae_train256_future_eval128_prereg import (
    build_rows as s3ae_rows,STYLES as S3AE_STYLES,
)
from research.nra2_bridge.stage3ak_newtrain256_futureeval128_prereg import (
    build_rows as s3ak_rows,STYLES as S3AK_STYLES,TRAIN as S3AK_TRAIN,EVAL as S3AK_EVAL,
)

SEED=2026111491
NODES=6
TRAIN="S3AQ_NEW_TRAIN256"
DEV="S3AQ_RESEARCH_DEV128"
EVAL="S3AQ_FUTURE_INDEPENDENT_EVAL128"
LENGTHS=(4,4,5,5,6,6,4,4)
REVERSED_STYLES=(0,2)
STYLES={
    TRAIN:(
       "Taking {dst} as the finishing station, can a directed walk begin at {src} and reach it?",
       "From starting point {src}, does following the arrows ever bring us to {dst}?",
       "If the endpoint is {dst}, is {src} a valid beginning for an arrow-respecting route?",
       "Depart at {src}; is there any allowed forward-directed route that finishes at {dst}?",
    ),
    DEV:(
       "With the journey ending at {dst}, could one have started at {src} using only forward arrows?",
       "Beginning at {src}, can a traveller ultimately reach {dst} without opposing an arrow?",
       "Treat {dst} as the final stop: is an arrow-directed route possible from {src}?",
       "A traveller leaves {src}; can the directed network lead them to {dst}?",
    ),
    EVAL:(
       "For destination {dst}, can any forward-directed itinerary originate at {src}?",
       "When setting out from {src}, will the directed links permit arrival at {dst}?",
       "Assume {dst} is the destination: can a valid route start from {src}?",
       "If we leave {src} and obey every arrow, is ending at {dst} possible?",
    ),
}
GRAPHS_PER_FAMILY={TRAIN:16,DEV:8,EVAL:8}
ROWS_PER_SPLIT={TRAIN:256,DEV:128,EVAL:128}
ORIGINAL_S3AO_CI=43579515862
PRIOR_S3AK_SHA="e1c8b0f9ad533482c351c5ac392f5923a59759c57c62d55080d692cbebf5354c"
PRIOR_S3AM_SHA="c5c400fc8249ea474263c9388f03e1496320bc8f6b793f93d155a962c3bf0eef"

def old_rows():
    return [r for group in previous_rows() for r in group]+s3aa_rows()+s3ae_rows()+s3ak_rows()

def build_rows():
    old=old_rows()
    used={tuple(sorted(r.edges)) for r in old}
    prior_prompt={r.prompt_sha256 for r in old}
    known_phrasings=set(S3M_STYLES)|set(S3T_STYLES)|set(S3X_STYLES)|set(S3AA_STYLES)|set(S3AE_STYLES["S3AE_NEW_TRAIN256"])|set(S3AE_STYLES["S3AE_FUTURE_INDEPENDENT_EVAL128"])|set(S3AK_STYLES[S3AK_TRAIN])|set(S3AK_STYLES[S3AK_EVAL])
    all_styles=[s for value in STYLES.values() for s in value]
    if len(all_styles)!=12 or len(set(all_styles))!=12 or known_phrasings.intersection(all_styles):
        raise AssertionError("S3AQ 12 new grammar phrasings must be unique and never used")
    rng=random.Random(SEED)
    rows=[]
    for split in (TRAIN,DEV,EVAL):
        for family in FAMILIES:
            count=GRAPHS_PER_FAMILY[split]
            for j in range(count):
                n=LENGTHS[j%len(LENGTHS)]
                for _ in range(30000):
                    order=tuple(rng.sample(range(NODES),n))
                    edges=make_graph(order,family)
                    key=tuple(sorted(edges))
                    possibilities=[(a,b) for a in order for b in order
                                   if a!=b and oracle(edges,a,b) and not oracle(edges,b,a)]
                    if key not in used and possibilities:
                        used.add(key)
                        break
                else:
                    raise AssertionError("S3AQ cannot generate unused directed topology")
                forward=rng.choice(possibilities)
                backward=(forward[1],forward[0])
                for style,template in enumerate(STYLES[split]):
                    yes=(style%2)==(j%2)
                    src,dst=forward if yes else backward
                    if oracle(edges,src,dst)!=yes:
                        raise AssertionError("S3AQ deterministic BFS truth mismatch")
                    arrows=[f"R{a} -> R{b}" for a,b in edges]
                    rng.shuffle(arrows)
                    question=HEADERS[(j//2+style)%len(HEADERS)].format(
                        edges="; ".join(arrows),
                        query=template.format(src=f"R{src}",dst=f"R{dst}"))
                    row=GraphPrompt(
                        graph_id=f"{split}-{family}-{j:02}",
                        split=split,family=family,active_nodes=n,
                        edges=edges,src=src,dst=dst,reachable=yes,
                        query_style=style,question=question)
                    if row.prompt_sha256 in prior_prompt:
                        raise AssertionError("S3AQ prompt reuse from earlier training or evaluation")
                    rows.append(row)
    validate(rows)
    return rows

def validate(rows):
    old=old_rows()
    old_t={tuple(sorted(r.edges)) for r in old}
    old_p={r.prompt_sha256 for r in old}
    if (len(rows)!=512 or len({r.prompt_sha256 for r in rows})!=512
        or {r.split for r in rows}!={TRAIN,DEV,EVAL}
        or any(r.topology_key in old_t or r.prompt_sha256 in old_p for r in rows)):
        raise AssertionError("S3AQ unique TRAIN256/DEV128/blind EVAL128 and prior disjoint")
    groups=defaultdict(list)
    for r in rows:
        groups[(r.split,r.graph_id)].append(r)
        if (r.family not in FAMILIES or oracle(r.edges,r.src,r.dst)!=r.reachable
            or graph_has_cycle(r.edges)!=(r.family=="cycle_sink")
            or len({n for e in r.edges for n in e})!=r.active_nodes):
            raise AssertionError("S3AQ invalid family/topology/BFS oracle")
        order=query_char_offsets(r.question)["text_order_nodes"]
        if (set(order)!={r.src,r.dst}
            or (order[0]==r.dst)!=(r.query_style in REVERSED_STYLES)):
            raise AssertionError("S3AQ source/destination order or S160 query span failure")
    if len(groups)!=128 or len({r.topology_key for r in rows})!=128:
        raise AssertionError("S3AQ 64 TRAIN + 32 DEV + 32 future EVAL disjoint topologies")
    for g in groups.values():
        if (len(g)!=4 or {x.query_style for x in g}!={0,1,2,3}
            or sum(x.reachable for x in g)!=2
            or len({x.topology_key for x in g})!=1
            or len({(x.src,x.dst) for x in g})!=2):
            raise AssertionError("S3AQ paired four-style truth balance per topology")
        pairs={(r.src,r.dst) for r in g}
        if any((b,a) not in pairs for a,b in pairs):
            raise AssertionError("S3AQ opposite direction pair absent")
    for split in (TRAIN,DEV,EVAL):
        subset=[r for r in rows if r.split==split]
        expected=ROWS_PER_SPLIT[split]
        per_family=GRAPHS_PER_FAMILY[split]
        if len(subset)!=expected or sum(r.reachable for r in subset)!=expected//2:
            raise AssertionError("S3AQ split label count mismatch")
        for fam in FAMILIES:
            part=[r for r in subset if r.family==fam]
            if len(part)!=4*per_family or sum(r.reachable for r in part)!=2*per_family:
                raise AssertionError("S3AQ family label imbalance")
        for style in range(4):
            items=[r for r in subset if r.query_style==style]
            if len(items)!=expected//4 or sum(r.reachable for r in items)!=expected//8:
                raise AssertionError("S3AQ style predicts truth")
            for n in (4,5,6):
                part=[r for r in items if r.active_nodes==n]
                if not part or len(part)%2 or sum(r.reachable for r in part)!=len(part)//2:
                    raise AssertionError("S3AQ node count predicts truth")
            for header in HEADERS:
                hprefix=header.split("{edges}")[0]
                part=[r for r in items if r.question.startswith(hprefix)]
                if not part or len(part)%2 or sum(r.reachable for r in part)!=len(part)//2:
                    raise AssertionError("S3AQ header predicts truth")
        for k,styles in (("destination_first",REVERSED_STYLES),("source_first",(1,3))):
            items=[r for r in subset if r.query_style in styles]
            if len(items)!=expected//2 or sum(r.reachable for r in items)!=expected//4:
                raise AssertionError("S3AQ role order predicts truth "+k)
    return True

def commitment(rows):
    validate(rows)
    groups={}
    for split in (TRAIN,DEV,EVAL):
        examples=[r for r in rows if r.split==split]
        graphs=[r for r in examples if r.query_style==0]
        groups[split]={
            "split":split,"count":len(examples),"topology_count":len(graphs),
            "yes_count":sum(r.reachable for r in examples),
            "destination_first_count":sum(r.query_style in REVERSED_STYLES for r in examples),
            "families":{f:sum(r.family==f for r in examples) for f in FAMILIES},
            "grammars":list(STYLES[split]),
            "question_sha256":[r.prompt_sha256 for r in examples],
            "topology_sha256":[hashlib.sha256(json.dumps(sorted(g.edges)).encode()).hexdigest() for g in graphs],
            "source_order_query_offsets":[query_char_offsets(r.question) for r in examples],
        }
    data={
        "stage":"NRA2_S3AQ_PRECOMMIT_TRAIN256_DEV128_FUTURE_EVAL128_NO_SCORE",
        "seed":SEED,
        "original_S3AO_successful_run":ORIGINAL_S3AO_CI,
        "S3AK_previous_independent128_commitment_sha256":PRIOR_S3AK_SHA,
        "S3AM_prior_training_checkpoint_sha256":PRIOR_S3AM_SHA,
        "previous_S3AK_EVAL128_is_already_scored_and_NOT_future_blind":True,
        "S3AO_negative_reachability_S3O_S3Z_S3AG_S3AM":(65,90,69,68),
        "TRAIN256_all_new_graphs_and_4_wordings":True,
        "DEV128_only_for_research_model_selection_not_final_holdout":True,
        "FUTURE_EVAL128_32_unseen_graphs_4_unused_new_wordings":True,
        "train_dev_eval_topology_and_prompt_disjoint":True,
        "new_eval_excluded_from_optimizer_and_checkpoint_selection":True,
        "future_eval_not_to_be_prefilled_before_candidate_physically_locked":True,
        "future_CPU_acquisition_and_training_not_started":True,
        "no_weights_or_training_or_scores":True,
        "original_S194_616_J7_guard_required":True,
        "no_candidate_accepted":True,
        "public_research_prereg_not_sealed_final_QA":True,
        "gate":0.0,
        "Jessica_LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
        "splits":groups,
    }
    payload=json.dumps(data,sort_keys=True,separators=(",",":")).encode()
    data["commitment_sha256"]=hashlib.sha256(payload).hexdigest()
    return data

def main():
    if len(sys.argv)!=2:
        raise SystemExit("S3AQ NEW_EMPTY_OUTPUT_DIR")
    out=Path(sys.argv[1])
    if out.exists() and any(out.iterdir()):
        raise AssertionError("S3AQ output cannot overwrite prior prereg")
    locked=commitment(build_rows())
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3AQ_TRAIN256_DEV128_FUTURE_EVAL128_PREREG.json").write_text(
        json.dumps(locked,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in locked.items() if k!="splits"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
