"""S3AK: preregister fresh TRAIN256 and held-out EVAL128 after S3AI negative score.

S3AA/S3AC and S3AK/S3AI independent evals were already scored: NEVER recycle them.
No model, S160 CPU, neural weights or metrics accessed. All graph topologies
and query wordings disjoint from previous experiment sets. Public research
prereg is NOT sealed final QA; no test scores, no training in this stage.
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

SEED=2026111481
NODES=6
TRAIN="S3AK_NEW_TRAIN256"
EVAL="S3AK_FUTURE_INDEPENDENT_EVAL128"
LENGTHS=(4,4,5,5,6,6,4,4)
REVERSED_STYLES=(0,2)
STYLES={
    TRAIN:(
       "Considering arrival at {dst}, do the directed connections allow departing from {src} to reach it?",
       "Starting the journey at {src}, can one eventually arrive at {dst} by following arrowheads?",
       "If {dst} is our endpoint, can a permitted directed path be traced back to a start at {src}?",
       "Set off at {src}: is reaching {dst} possible without traversing an arrow backward?",
    ),
    EVAL:(
       "With {dst} chosen as the destination, is there a forward-arrow route whose first node is {src}?",
       "A trip begins at {src}; can the marked directions carry it all the way to {dst}?",
       "Suppose the intended arrival is {dst}; can a route following arrows start at {src}?",
       "From {src} as the starting node, does any arrow-following itinerary terminate at {dst}?",
    ),
}
GRAPHS_PER_FAMILY={TRAIN:16,EVAL:8}
ROWS_PER_SPLIT={TRAIN:256,EVAL:128}
ORIGINAL_S3AI_CI=43541542901
PRIOR_S3AE_SHA="0ccbb2b19b3feb93b79f5dcbc892f2e414a564b6e4d83e52997098346e4f480f"
PRIOR_S3AG_SHA="2c23de87efc2369ed6fc0880604420070c116a4ee70b66a4eb0a946f1afeb118"

def old_rows():
    return [r for group in previous_rows() for r in group]+s3aa_rows()+s3ae_rows()

def build_rows():
    old=old_rows()
    used={tuple(sorted(r.edges)) for r in old}
    prior_prompt={r.prompt_sha256 for r in old}
    known_phrasings=set(S3M_STYLES)|set(S3T_STYLES)|set(S3X_STYLES)|set(S3AA_STYLES)|set(S3AE_STYLES["S3AE_NEW_TRAIN256"])|set(S3AE_STYLES["S3AE_FUTURE_INDEPENDENT_EVAL128"])
    all_styles=[s for value in STYLES.values() for s in value]
    if len(all_styles)!=8 or len(set(all_styles))!=8 or known_phrasings.intersection(all_styles):
        raise AssertionError("S3AK 8 new query phrasings must be unique and never used")
    rng=random.Random(SEED)
    rows=[]
    for split in (TRAIN,EVAL):
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
                    raise AssertionError("S3AK cannot generate unused directed topology")
                forward=rng.choice(possibilities)
                backward=(forward[1],forward[0])
                for style,template in enumerate(STYLES[split]):
                    yes=(style%2)==(j%2)
                    src,dst=forward if yes else backward
                    if oracle(edges,src,dst)!=yes:
                        raise AssertionError("S3AK deterministic BFS truth mismatch")
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
                        raise AssertionError("S3AK prompt reuse from earlier training or evaluation")
                    rows.append(row)
    validate(rows)
    return rows

def validate(rows):
    old=old_rows()
    old_t={tuple(sorted(r.edges)) for r in old}
    old_p={r.prompt_sha256 for r in old}
    if (len(rows)!=384 or len({r.prompt_sha256 for r in rows})!=384
        or {r.split for r in rows}!={TRAIN,EVAL}
        or any(r.topology_key in old_t or r.prompt_sha256 in old_p for r in rows)):
        raise AssertionError("S3AK unique research TRAIN256/EVAL128 and prior disjoint")
    groups=defaultdict(list)
    for r in rows:
        groups[(r.split,r.graph_id)].append(r)
        if (r.family not in FAMILIES or oracle(r.edges,r.src,r.dst)!=r.reachable
            or graph_has_cycle(r.edges)!=(r.family=="cycle_sink")
            or len({n for e in r.edges for n in e})!=r.active_nodes):
            raise AssertionError("S3AK invalid family/topology/BFS oracle")
        order=query_char_offsets(r.question)["text_order_nodes"]
        if (set(order)!={r.src,r.dst}
            or (order[0]==r.dst)!=(r.query_style in REVERSED_STYLES)):
            raise AssertionError("S3AK source/destination order or S160 query span failure")
    if len(groups)!=96 or len({r.topology_key for r in rows})!=96:
        raise AssertionError("S3AK 64 TRAIN and 32 heldout disjoint topology requirement")
    for g in groups.values():
        if (len(g)!=4 or {x.query_style for x in g}!={0,1,2,3}
            or sum(x.reachable for x in g)!=2
            or len({x.topology_key for x in g})!=1
            or len({(x.src,x.dst) for x in g})!=2):
            raise AssertionError("S3AK paired four-style truth balance per topology")
        pairs={(r.src,r.dst) for r in g}
        if any((b,a) not in pairs for a,b in pairs):
            raise AssertionError("S3AK opposite direction pair absent")
    for split in (TRAIN,EVAL):
        subset=[r for r in rows if r.split==split]
        expected=ROWS_PER_SPLIT[split]
        per_family=GRAPHS_PER_FAMILY[split]
        if len(subset)!=expected or sum(r.reachable for r in subset)!=expected//2:
            raise AssertionError("S3AK split label count mismatch")
        for fam in FAMILIES:
            part=[r for r in subset if r.family==fam]
            if len(part)!=4*per_family or sum(r.reachable for r in part)!=2*per_family:
                raise AssertionError("S3AK family label imbalance")
        for style in range(4):
            items=[r for r in subset if r.query_style==style]
            if len(items)!=expected//4 or sum(r.reachable for r in items)!=expected//8:
                raise AssertionError("S3AK style predicts truth")
            for n in (4,5,6):
                part=[r for r in items if r.active_nodes==n]
                if not part or len(part)%2 or sum(r.reachable for r in part)!=len(part)//2:
                    raise AssertionError("S3AK node count predicts truth")
            for header in HEADERS:
                hprefix=header.split("{edges}")[0]
                part=[r for r in items if r.question.startswith(hprefix)]
                if not part or len(part)%2 or sum(r.reachable for r in part)!=len(part)//2:
                    raise AssertionError("S3AK header predicts truth")
        for k,styles in (("destination_first",REVERSED_STYLES),("source_first",(1,3))):
            items=[r for r in subset if r.query_style in styles]
            if len(items)!=expected//2 or sum(r.reachable for r in items)!=expected//4:
                raise AssertionError("S3AK role order predicts truth "+k)
    return True

def commitment(rows):
    validate(rows)
    groups={}
    for split in (TRAIN,EVAL):
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
        "stage":"NRA2_S3AK_PRECOMMIT_DISJOINT_NEW_TRAIN256_FUTURE_EVAL128_NO_SCORE",
        "seed":SEED,"original_S3AI_successful_run":ORIGINAL_S3AI_CI,
        "S3AE_previous_independent128_commitment_sha256":PRIOR_S3AE_SHA,
        "previous_S3AE_EVAL128_is_already_scored_and_NOT_future_blind":True,
        "S3AG_previously_locked_checkpoint_sha256":PRIOR_S3AG_SHA,
        "S3AI_negative_reachability_S3O_S3Z_S3AG":(96,91,92),
        "TRAIN256_all_new_graphs_and_4_wordings":True,
        "FUTURE_EVAL128_separate_32_new_graphs_4_distinct_new_wordings":True,
        "new_eval_excluded_from_optimizer_and_checkpoint_selection":True,
        "future_eval_not_to_be_prefilled_before_next_candidate_physically_locked":True,
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
        raise SystemExit("S3AK NEW_EMPTY_OUTPUT_DIR")
    out=Path(sys.argv[1])
    if out.exists() and any(out.iterdir()):
        raise AssertionError("S3AK output cannot overwrite prior prereg")
    locked=commitment(build_rows())
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3AK_NEW_TRAIN256_FUTURE_EVAL128_PREREG.json").write_text(
        json.dumps(locked,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in locked.items() if k!="splits"},sort_keys=True),flush=True)

if __name__=="__main__":
    main()
