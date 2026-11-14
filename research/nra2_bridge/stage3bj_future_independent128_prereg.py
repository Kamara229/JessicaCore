"""S3BJ: NEW future blind 128 graph+grammar examples; prereg only, NO model.

The former S3AQ EVAL128 was consumed at S3BH and CANNOT be used again.
Construct disjoint 32 graph topologies and four novel query templates.
DO NOT capture authentic CPU tokens, infer or score candidates in this stage.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import random
import sys

from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows as s3aq_rows, old_rows as all_pre_s3aq_rows,
    STYLES as S3AQ_STYLES, commitment as s3aq_commitment,
)
from research.nra2_bridge.stage3au_train512_paraphrase_prereg import STYLES as S3AU_STYLES
from research.nra2_bridge.stage3ay_structural_train512_prereg import STYLES as S3AY_STYLES
from research.nra2_bridge.stage3bc_contrastive_train512_prereg import STYLES as S3BC_STYLES
from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    GraphPrompt,FAMILIES,HEADERS,make_graph,graph_has_cycle,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle

SPLIT="S3BJ_FUTURE_INDEPENDENT_ROLE_EVAL128"
SEED=2026111517
S3AQ_SHA="c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839"
PREVIOUS_S3BH_EVAL_RUN=43640363396
PARENT_S3BA_SHA="4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09"
FAMILY_COUNT=8
LENGTHS=(4,4,5,5,6,6,4,4)
DESTINATION_FIRST=(0,2)
SOURCE_FIRST=(1,3)
STYLES=(
    "Is reaching the marked terminus {dst} possible if departure occurs at {src} while respecting arrow orientation?",
    "Starting the search at node {src}, can any chain of outward-directed connections end at {dst}?",
    "Imagine {dst} is the required arrival node; could a journey whose origin is {src} get there by arrow-following?",
    "Take {src} as the launch location; will tracing permitted directed arcs ever arrive at {dst}?",
)

def require(flag,why):
    if not flag:
        raise AssertionError("S3BJ prereg fail closed: "+why)

def source_ancestry():
    old=all_pre_s3aq_rows()
    aq=s3aq_rows()
    require(s3aq_commitment(aq)["commitment_sha256"]==S3AQ_SHA,"original S3AQ integrity")
    return old+aq

def build_rows():
    old=source_ancestry()
    used={tuple(sorted(r.edges)) for r in old}
    used_prompt={r.prompt_sha256 for r in old}
    previous_styles={v for ss in S3AQ_STYLES.values() for v in ss}
    previous_styles.update(S3AU_STYLES)
    previous_styles.update(S3AY_STYLES)
    previous_styles.update(S3BC_STYLES)
    require(len(STYLES)==4 and len(set(STYLES))==4 and not set(STYLES)&previous_styles,
            "all four novel EVAL phrasings must be distinct from training")
    rng=random.Random(SEED)
    out=[]
    for family in FAMILIES:
        for j in range(FAMILY_COUNT):
            n=LENGTHS[j]
            for _ in range(30000):
                order=tuple(rng.sample(range(6),n))
                edges=make_graph(order,family)
                key=tuple(sorted(edges))
                poss=[(a,b) for a in order for b in order
                      if a!=b and oracle(edges,a,b) and not oracle(edges,b,a)]
                if key not in used and poss:
                    used.add(key)
                    break
            else:
                raise AssertionError("S3BJ insufficient new directed graphs")
            src_forward,dst_forward=rng.choice(poss)
            for style,tmpl in enumerate(STYLES):
                yes=(style%2)==(j%2)
                src,dst=(src_forward,dst_forward) if yes else (dst_forward,src_forward)
                require(oracle(edges,src,dst)==yes,"contradictory graph truth")
                edge_words=[f"R{a} -> R{b}" for a,b in edges]
                rng.shuffle(edge_words)
                question=HEADERS[(j//2+style)%len(HEADERS)].format(
                    edges="; ".join(edge_words),
                    query=tmpl.format(src=f"R{src}",dst=f"R{dst}"))
                row=GraphPrompt(graph_id=f"{SPLIT}-{family}-{j:02}",
                                split=SPLIT,family=family,active_nodes=n,
                                edges=edges,src=src,dst=dst,reachable=yes,
                                query_style=style,question=question)
                require(row.prompt_sha256 not in used_prompt,
                        "new EVAL question already used")
                out.append(row)
    validate(out)
    return out

def validate(rows):
    require(len(rows)==128 and len({r.prompt_sha256 for r in rows})==128,
            "128 unique queries required")
    ancient=source_ancestry()
    oldkeys={tuple(sorted(r.edges)) for r in ancient}
    oldhash={r.prompt_sha256 for r in ancient}
    require(not ({r.topology_key for r in rows}&oldkeys),"future EVAL graph reused")
    require(not ({r.prompt_sha256 for r in rows}&oldhash),"future EVAL prompt reused")
    groups=defaultdict(list)
    for r in rows:
        require(r.split==SPLIT and r.family in FAMILIES and
                r.reachable==oracle(r.edges,r.src,r.dst) and
                graph_has_cycle(r.edges)==(r.family=="cycle_sink"),
                "graph family/reachability drift")
        order=query_char_offsets(r.question)["text_order_nodes"]
        require(order==([r.dst,r.src] if r.query_style in DESTINATION_FIRST
                        else [r.src,r.dst]),"literal query direction drift")
        groups[r.graph_id].append(r)
    require(len(groups)==32 and len({r.topology_key for r in rows})==32,
            "32 entirely novel topologies required")
    for group in groups.values():
        group.sort(key=lambda r:r.query_style)
        require([r.query_style for r in group]==[0,1,2,3]
                and sum(r.reachable for r in group)==2
                and len({(r.src,r.dst) for r in group})==2,
                "balanced four-style per graph")
        a,b=group[0],group[1]
        require((a.src,a.dst)==(b.dst,b.src),"contradictory source-target pair")
    require(sum(r.reachable for r in rows)==64,"yes/no must be 64/64")
    for style in range(4):
        sub=[r for r in rows if r.query_style==style]
        require(len(sub)==32 and sum(r.reachable for r in sub)==16,
                "style accidentally predicts graph truth")
    require(sum(r.query_style in DESTINATION_FIRST for r in rows)==64,
            "not 64/64 destination/source order")
    for family in FAMILIES:
        sub=[r for r in rows if r.family==family]
        require(len(sub)==32 and sum(r.reachable for r in sub)==16,
                "family imbalance")
    return True

def commitment(rows):
    validate(rows)
    result={
        "stage":"NRA2_S3BJ_FUTURE_INDEPENDENT128_PREREG_NO_CPU_NO_MODEL",
        "protocol":"ONE_TIME_FUTURE_INDEPENDENT_UNOPENED_BLIND",
        "original_S3AQ_registered_sha256":S3AQ_SHA,
        "S3AQ_old_EVAL128_ALREADY_CONSUMED_at_S3BH":PREVIOUS_S3BH_EVAL_RUN,
        "old_EVAL128_must_never_be_used_for_training_or_new_blind":True,
        "original_frozen_S3BA_parent_sha256":PARENT_S3BA_SHA,
        "candidate_S3BI_physical_FP16_must_be_locked_before_new_eval_CPU":True,
        "seed":SEED,"split":SPLIT,"count":128,"graph_count":32,
        "grammar_count":4,"yes":64,"no":64,
        "source_first":64,"destination_first":64,
        "grammar_styles":list(STYLES),
        "question_sha256":[r.prompt_sha256 for r in rows],
        "topology_sha256":[hashlib.sha256(json.dumps(sorted(r.edges)).encode()).hexdigest()
                            for r in rows[::4]],
        "gold_reachability_sha256":hashlib.sha256(bytes([int(r.reachable) for r in rows])).hexdigest(),
        "no_model_weights_or_CPU_capture_or_adaptor_scoring":True,
        "original_616_J7_must_remain_frozen":True,
        "future_candidate_accepted":False,"output_gate":0.0,
        "no_LIVE_Render_Supabase_APK_main_QA_TEST":True,
    }
    b=json.dumps(result,sort_keys=True,separators=(",",":")).encode()
    result["commitment_sha256"]=hashlib.sha256(b).hexdigest()
    return result

def main():
    if len(sys.argv)!=2:
        raise SystemExit("Usage: stage3bj_future_independent128_prereg OUTPUT_EMPTY_DIR")
    out=Path(sys.argv[1])
    require(not out.exists(),"cannot overwrite future blind prereg")
    locked=commitment(build_rows())
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3BJ_FUTURE_INDEPENDENT_EVAL128_PREREG.json").write_text(
        json.dumps(locked,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in locked.items()
                      if k not in ("question_sha256","topology_sha256","grammar_styles")},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    main()
