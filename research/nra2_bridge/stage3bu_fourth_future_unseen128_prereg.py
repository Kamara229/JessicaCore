"""S3BU: FOURTH and completely untouched future independent128 commitment.

Three previous one-time EVAL128 sets S3AQ, S3BJ, S3BO were scored and
PERMANENTLY consumed; this generator uses their registered graph hashes
only to EXCLUDE their topology from a future independent evaluation.
It never runs models, captures original frozen CPU, fits weights, or scores.
The S3BR new 512 curriculum is original TRAIN topologies only.
"""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import random
import sys

from research.nra2_bridge.stage3bo_third_future_independent128_prereg import (
    ancestors as old_graph_rows,build_rows as third_spent_rows,
    commitment as third_commit,STYLES as THIRD_STYLES,
)
from research.nra2_bridge.stage3br_relation_language_train512_prereg import STYLES as NEW_TRAIN_STYLES
from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import STYLES as EARLY_STYLES
from research.nra2_bridge.stage3bj_future_independent128_prereg import STYLES as SECOND_STYLES
from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    GraphPrompt,FAMILIES,HEADERS,make_graph,graph_has_cycle,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle

SPLIT="S3BU_FOURTH_INDEPENDENT_UNTOUCHED_ROLE_EVAL128"
SEED=2026111594
DESTINATION_FIRST=(0,2,4,6)
SOURCE_FIRST=(1,3,5,7)
LENGTHS=(5,6,6,5)
S3BO_SHA="01858b405fbb034055574fd27feb93d19a34833da342a664ca5daed2b3ad091b"
S3BR_SHA="e941bd45f947b7fc16f0cce4307ed9ea69c0b0e8347da7c0754717430ff0262f"
STYLES=(
    "With {dst} designated as the destination, does an arrow-respecting route exist from {src}?",
    "Given a departure from {src}, is arrival at {dst} achievable along directed connections?",
    "Is delivery to {dst} possible for a packet originally transmitted from {src}?",
    "For a packet outgoing from {src}, might its directed journey eventually deliver to {dst}?",
    "Would endpoint {dst} belong to the forward reachability closure computed from seed {src}?",
    "If the reachability closure is initiated at {src}, does it contain node {dst}?",
    "Could a planned visit to {dst} take place after starting a direction-only traversal at {src}?",
    "When a direction-only traversal starts with {src}, can it ever visit {dst}?",
)

def require(ok,msg):
    if not ok:raise AssertionError("S3BU fourth future EVAL hash only "+msg)

@lru_cache(maxsize=1)
def ancestral_exclusions():
    old=old_graph_rows()
    third=third_spent_rows()
    require(third_commit(third)["commitment_sha256"]==S3BO_SHA,
            "previous consumed third EVAL SHA drift")
    return old+third

def build_rows():
    previous=ancestral_exclusions()
    used={tuple(sorted(r.edges)) for r in previous}
    old_sha={r.prompt_sha256 for r in previous}
    prior=set(THIRD_STYLES)|set(SECOND_STYLES)|set(NEW_TRAIN_STYLES)
    prior|={s for x in EARLY_STYLES.values() for s in x}
    require(len(STYLES)==8 and len(set(STYLES))==8 and not set(STYLES)&prior,
            "eight fourth EVAL syntax styles must be entirely unused")
    rng=random.Random(SEED)
    rows=[]
    for family in FAMILIES:
        for j in range(4):
            n=LENGTHS[j]
            for _ in range(30000):
                order=tuple(rng.sample(range(6),n))
                edges=make_graph(order,family)
                key=tuple(sorted(edges))
                forward=[(a,b) for a in order for b in order
                         if a!=b and oracle(edges,a,b) and not oracle(edges,b,a)]
                if key not in used and forward:
                    used.add(key)
                    break
            else:raise AssertionError("unable to find fourth unseen graph topology")
            src0,dst0=rng.choice(forward)
            for style,form in enumerate(STYLES):
                yes=((style%2)==(j%2))
                src,dst=(src0,dst0) if yes else (dst0,src0)
                require(oracle(edges,src,dst)==yes,
                        "directed graph truth inconsistent")
                edge_text=[f"R{a} -> R{b}" for a,b in edges]
                rng.shuffle(edge_text)
                q=HEADERS[(j//2+style)%len(HEADERS)].format(
                    edges="; ".join(edge_text),
                    query=form.format(src=f"R{src}",dst=f"R{dst}"))
                r=GraphPrompt(graph_id=f"{SPLIT}-{family}-{j:02}",
                              split=SPLIT,family=family,active_nodes=n,
                              edges=edges,src=src,dst=dst,
                              reachable=yes,query_style=style,question=q)
                require(r.prompt_sha256 not in old_sha,
                        "fourth EVAL question already seen elsewhere")
                rows.append(r)
    validate(rows)
    return rows

def validate(rows):
    require(len(rows)==128 and len({r.prompt_sha256 for r in rows})==128,
            "fourth independent128 prompt count or uniqueness")
    past=ancestral_exclusions()
    used={tuple(sorted(r.edges)) for r in past}
    old_sha={r.prompt_sha256 for r in past}
    require(not ({r.topology_key for r in rows}&used) and
            not ({r.prompt_sha256 for r in rows}&old_sha),
            "FOURTH EVAL must have never-seen graphs and phrases")
    graph=defaultdict(list)
    for r in rows:
        require(r.split==SPLIT and r.family in FAMILIES
                and oracle(r.edges,r.src,r.dst)==r.reachable
                and graph_has_cycle(r.edges)==(r.family=="cycle_sink"),
                "FOURTH original graph family or truth changed")
        order=query_char_offsets(r.question)["text_order_nodes"]
        require(order==([r.dst,r.src] if r.query_style in DESTINATION_FIRST
                        else [r.src,r.dst]),
                "FOURTH EVAL literal role offsets mismatched")
        graph[r.graph_id].append(r)
    require(len(graph)==16 and len({r.topology_key for r in rows})==16,
            "FOURTH exactly 16 new graph topologies")
    for variants in graph.values():
        variants=sorted(variants,key=lambda r:r.query_style)
        require([r.query_style for r in variants]==list(range(8))
                and sum(r.reachable for r in variants)==4
                and len({(r.src,r.dst) for r in variants})==2,
                "FOURTH graph eight variants incomplete")
        for k in (0,2,4,6):
            require((variants[k].src,variants[k].dst)==
                    (variants[k+1].dst,variants[k+1].src),
                    "FOURTH paired reversal missing")
    require(sum(r.reachable for r in rows)==64
            and sum(r.query_style in DESTINATION_FIRST for r in rows)==64,
            "FOURTH EVAL role direction/yes-no imbalance")
    for f in FAMILIES:
        sub=[r for r in rows if r.family==f]
        require(len(sub)==32 and sum(r.reachable for r in sub)==16,
                "FOURTH EVAL family imbalance")
    for style in range(8):
        sub=[r for r in rows if r.query_style==style]
        require(len(sub)==16 and sum(r.reachable for r in sub)==8,
                "FOURTH EVAL style predicts graph truth")
    return True

def commitment(rows):
    validate(rows)
    d={
        "stage":"NRA2_S3BU_FOURTH_UNTOUCHED_EVAL128_PREREG_NO_CPU_OR_MODEL",
        "split":SPLIT,"seed":SEED,
        "S3BO_prior_third_consumed_commitment_sha256":S3BO_SHA,
        "S3BR_prior_new_TRAIN512_commitment_sha256":S3BR_SHA,
        "prior_first_independent_EVAL128_consumed_in":43640363396,
        "prior_second_independent_EVAL128_consumed_in":43644153544,
        "prior_third_independent_EVAL128_consumed_in":43648038071,
        "fourth_independent_graphs":16,
        "fourth_independent_new_grammar":8,"future_EVAL128_count":128,
        "yes_count":64,"no_count":64,
        "source_first_count":64,"destination_first_count":64,
        "styles":list(STYLES),
        "sha_prompts_in_order":[r.prompt_sha256 for r in rows],
        "sha_graphs_in_order":[hashlib.sha256(
            json.dumps(sorted(r.edges)).encode()).hexdigest() for r in rows[::8]],
        "sha_yes_no":hashlib.sha256(bytes(int(r.reachable) for r in rows)).hexdigest(),
        "ALL_previous_independent_EVALs_excluded_from_optimizer_and_candidate_selection":True,
        "S3BT_candidate_physically_SHA_locked_before_first_fourth_EVAL_CPU":True,
        "no_model_weights_or_Calibri_CPU_or_scores_or_training_this_stage":True,
        "original_616_S194_J7_preserve":True,
        "output_gate":0.,"candidate_accepted":False,
        "no_LIVE_Jessica_Render_Supabase_APK_main_QA_TEST_changes":True,
    }
    d["commitment_sha256"]=hashlib.sha256(
        json.dumps(d,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return d

def main():
    require(len(sys.argv)==2,"usage: S3BU EMPTY_DIR")
    out=Path(sys.argv[1])
    require(not out.exists(),"fourth independent EVAL protocol cannot be overwritten")
    d=commitment(build_rows())
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3BU_FOURTH_UNTOUCHED128_PREREG.json").write_text(
        json.dumps(d,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in d.items()
                      if k not in ("styles","sha_prompts_in_order","sha_graphs_in_order")},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    main()
