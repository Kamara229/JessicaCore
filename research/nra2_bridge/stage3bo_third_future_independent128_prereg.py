"""NRA2 S3BO: THIRD independently precommitted future EVAL128, data-only.

S3AQ EVAL128 was consumed in once-only S3BH; S3BJ EVAL128 in once-only
S3BL. They are excluded from any optimizer, test selection, or new blind
evaluation. New 16 untrained graph topologies times 8 unseen natural-language
query patterns, 128 novel independent samples, 64/64 source/destination order.
NO neural model or original Calibri CPU invoked in this preregistration.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows as s3aq_rows,old_rows as pre_s3aq_rows,
    commitment as s3aq_commitment,STYLES as S3AQ_STYLES,
)
from research.nra2_bridge.stage3bj_future_independent128_prereg import (
    build_rows as s3bj_rows,commitment as s3bj_commitment,
    STYLES as S3BJ_STYLES,
)
from research.nra2_bridge.stage3au_train512_paraphrase_prereg import STYLES as S3AU_STYLES
from research.nra2_bridge.stage3ay_structural_train512_prereg import STYLES as S3AY_STYLES
from research.nra2_bridge.stage3bc_contrastive_train512_prereg import STYLES as S3BC_STYLES
from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    GraphPrompt,FAMILIES,HEADERS,make_graph,graph_has_cycle,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle

SPLIT="S3BO_FUTURE_THIRD_INDEPENDENT_ROLE_EVAL128"
SEED=2026111552
S3AQ_SHA="c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839"
S3BJ_SHA="e48faef1a8b164cad9641f1b5946932f15c27baeeab47d829fb5ec754dd46265"
S3BL_CONSUMED=43644153544
SOURCE_FIRST=(1,3,5,7)
DESTINATION_FIRST=(0,2,4,6)
NODES=6
FAMILY_GRAPH_COUNT=4
LENGTHS=(4,4,5,6)
STYLES=(
    "Could {dst} be the final stop on an arrow-following trip that began at {src}?",
    "If a walker sets out at {src}, can it eventually get as far as {dst} on directed arcs?",
    "Suppose arrival at {dst} is required; can travel start at {src} and follow the arrowheads to it?",
    "A message starts at {src}; can a sequence of permitted outward steps deliver it to {dst}?",
    "Is {dst} a possible endpoint for a one-way route whose first node is {src}?",
    "Moving only in permitted edge directions from {src}, can the route ever visit {dst}?",
    "If the desired end is {dst}, can a legal directed walk have its origin at {src}?",
    "Beginning with node {src}, might repeated forward moves eventually encounter {dst}?",
)

def require(ok,msg):
    if not ok:
        raise AssertionError("S3BO sealed data-only prereg: "+msg)

def ancestors():
    old=pre_s3aq_rows()
    aq=s3aq_rows()
    bj=s3bj_rows()
    require(s3aq_commitment(aq)["commitment_sha256"]==S3AQ_SHA,
            "old original split commitment drift")
    require(s3bj_commitment(bj)["commitment_sha256"]==S3BJ_SHA,
            "already-spent S3BJ split commitment drift")
    return old+aq+bj

def build_rows():
    old=ancestors()
    used={tuple(sorted(r.edges)) for r in old}
    used_hashes={r.prompt_sha256 for r in old}
    prior_templates={s for arr in S3AQ_STYLES.values() for s in arr}
    prior_templates.update(S3BJ_STYLES)
    prior_templates.update(S3AU_STYLES)
    prior_templates.update(S3AY_STYLES)
    prior_templates.update(S3BC_STYLES)
    require(len(STYLES)==8 and len(set(STYLES))==8
            and not set(STYLES)&prior_templates,
            "third EVAL eight wordings must be fresh and not use previous styles")
    rng=random.Random(SEED)
    rows=[]
    for family in FAMILIES:
        for j in range(FAMILY_GRAPH_COUNT):
            n=LENGTHS[j]
            for _ in range(30000):
                order=tuple(rng.sample(range(NODES),n))
                edges=make_graph(order,family)
                key=tuple(sorted(edges))
                forward=[(a,b) for a in order for b in order
                         if a!=b and oracle(edges,a,b)
                         and not oracle(edges,b,a)]
                if key not in used and forward:
                    used.add(key)
                    break
            else:
                raise AssertionError("S3BO no unused novel topology")
            fsrc,fdst=rng.choice(forward)
            for style,template in enumerate(STYLES):
                yes=(style%2)==(j%2)
                src,dst=(fsrc,fdst) if yes else (fdst,fsrc)
                require(oracle(edges,src,dst)==yes,"third EVAL graph truth invalid")
                arrows=[f"R{a} -> R{b}" for a,b in edges]
                rng.shuffle(arrows)
                question=HEADERS[(j//2+style)%len(HEADERS)].format(
                    edges="; ".join(arrows),
                    query=template.format(src=f"R{src}",dst=f"R{dst}"))
                row=GraphPrompt(graph_id=f"{SPLIT}-{family}-{j:02}",
                                split=SPLIT,family=family,
                                active_nodes=n,edges=edges,
                                src=src,dst=dst,reachable=yes,
                                query_style=style,question=question)
                require(row.prompt_sha256 not in used_hashes,
                        "third EVAL question hash reused in old research")
                rows.append(row)
    validate(rows)
    return rows

def validate(rows):
    require(len(rows)==128 and len({r.prompt_sha256 for r in rows})==128,
            "128 third-blind unique questions required")
    ancient=ancestors()
    oldkeys={tuple(sorted(r.edges)) for r in ancient}
    oldhash={r.prompt_sha256 for r in ancient}
    require(not ({r.topology_key for r in rows}&oldkeys),
            "S3BO third independent graph overlap prior TRAIN or EVAL")
    require(not ({r.prompt_sha256 for r in rows}&oldhash),
            "S3BO third independent literal prompt overlap")
    by_graph=defaultdict(list)
    for r in rows:
        require(r.split==SPLIT and r.family in FAMILIES
                and oracle(r.edges,r.src,r.dst)==r.reachable
                and graph_has_cycle(r.edges)==(r.family=="cycle_sink"),
                "S3BO graph family/direction incorrect")
        order=query_char_offsets(r.question)["text_order_nodes"]
        expected=[r.dst,r.src] if r.query_style in DESTINATION_FIRST else [r.src,r.dst]
        require(order==expected,"S3BO query source/destination literal order incorrect")
        by_graph[r.graph_id].append(r)
    require(len(by_graph)==16 and len({r.topology_key for r in rows})==16,
            "S3BO exactly 16 disjoint graph topologies")
    for group in by_graph.values():
        group.sort(key=lambda r:r.query_style)
        require([r.query_style for r in group]==list(range(8))
                and sum(r.reachable for r in group)==4
                and len({(r.src,r.dst) for r in group})==2,
                "S3BO each graph eight role-balanced novel utterances")
        for k in (0,2,4,6):
            require((group[k].src,group[k].dst)==
                    (group[k+1].dst,group[k+1].src),
                    "S3BO paired reversal missing")
    require(sum(r.reachable for r in rows)==64
            and sum(r.query_style in DESTINATION_FIRST for r in rows)==64,
            "S3BO yes/no or literal-role order balance drift")
    for family in FAMILIES:
        group=[r for r in rows if r.family==family]
        require(len(group)==32 and sum(r.reachable for r in group)==16,
                "S3BO family balance drift")
    for style in range(8):
        group=[r for r in rows if r.query_style==style]
        require(len(group)==16 and sum(r.reachable for r in group)==8,
                "S3BO style predicts graph label")
    return True

def commitment(rows):
    validate(rows)
    data={
        "stage":"NRA2_S3BO_THIRD_FUTURE_INDEPENDENT_EVAL128_NO_MODEL_NO_CPU",
        "split":SPLIT,"seed":SEED,
        "prior_S3AQ_consume_heldout":43640363396,
        "prior_S3BJ_consume_heldout":S3BL_CONSUMED,
        "prior_original_S3AQ_SHA256":S3AQ_SHA,
        "prior_S3BJ_SHA256":S3BJ_SHA,
        "S3BM_architecture_existed_before_third_split":True,
        "S3BN_frozen_candidate_SHA_required_before_CPU_capture_or_scores":True,
        "question_count":128,"new_graph_topology_count":16,
        "new_grammar_style_count":8,
        "yes_count":64,"no_count":64,
        "destination_first_count":64,"source_first_count":64,
        "style_templates":list(STYLES),
        "questions_in_order_sha256":[r.prompt_sha256 for r in rows],
        "graph_sha256_in_order":[hashlib.sha256(
            json.dumps(sorted(r.edges)).encode()).hexdigest()
            for r in rows[::8]],
        "reachability_vector_sha256":hashlib.sha256(
            bytes([int(r.reachable) for r in rows])).hexdigest(),
        "no_spent_EVALs_reused_as_NEW_blind_or_fit":True,
        "no_model_weights_original_CPU_inference_or_adapter_scoring":True,
        "original_S194_616_J7_must_be_preserved":True,
        "output_gate":0.,"candidate_accepted":False,
        "LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    data["commitment_sha256"]=hashlib.sha256(
        json.dumps(data,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return data

def main():
    require(len(sys.argv)==2,"usage: S3BO EMPTY_OUTPUT_PATH")
    out=Path(sys.argv[1])
    require(not out.exists(),"immutable new prereg cannot overwrite")
    data=commitment(build_rows())
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3BO_THIRD_FUTURE_EVAL128_PREREG.json").write_text(
        json.dumps(data,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in data.items()
                      if k not in ("questions_in_order_sha256","graph_sha256_in_order","style_templates") },
                     sort_keys=True),flush=True)

if __name__=="__main__":
    main()
