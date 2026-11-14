"""S3BR: DIRECTIONAL LANGUAGE CURRICULUM TRAIN512, data-only, no EVAL use.

Third one-time independent EVAL128 was consumed by S3BQ #43648038071,
second S3BJ by S3BL, first S3AQ EVAL by S3BH. Their prompts, labels and
original CPU tokens are absolutely excluded from any future training.
This stage only reuses 64 *known original S3AQ TRAIN graph topologies* with
eight entirely new directed-relation constructions and balanced reversals.
Graph truth is used to label TRAIN only; never enters model inference.
No real CPU, weights, role-head scoring, speed or model changes here.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows as original_rows,commitment as parent_commit,
    TRAIN as ORIGINAL_TRAIN, DEV as ORIGINAL_DEV,EVAL as SPENT_OLD_EVAL,
    STYLES as AQ_STYLES,
)
from research.nra2_bridge.stage3bc_contrastive_train512_prereg import STYLES as BC_STYLES
from research.nra2_bridge.stage3au_train512_paraphrase_prereg import STYLES as AU_STYLES
from research.nra2_bridge.stage3ay_structural_train512_prereg import STYLES as AY_STYLES
from research.nra2_bridge.stage3m_role_balanced_graph_data import GraphPrompt,HEADERS,FAMILIES
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle

SPLIT="S3BR_NEW_DIRECTIONAL_RELATION_TRAIN512_ONLY"
SEED=2026111566
OLD_S3AQ_SHA="c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839"
S3BN_FROZEN_INIT_SHA="ea8a81665370cba22c4e8bc75fbe82314089244d4b32278ee62c2c30d88ae811"
FIRST_OLD_EVAL_CONSUMED=43640363396
SECOND_OLD_EVAL_CONSUMED=43644153544
THIRD_OLD_EVAL_CONSUMED=43648038071
SOURCE_FIRST=(0,2,4,6)
DESTINATION_FIRST=(1,3,5,7)
CONSTRUCTIONS=("signal_passing","formal_transition","process_terminal","successor_sequence")
STYLES=(
    "Where a signal begins at {src} and every hop follows an outgoing arrow, could it ultimately be received by {dst}?",
    "Could {dst} eventually receive a signal sent from {src} if every hop obeys the arrow direction?",
    "Treat {src} as the initial state of a directed transition system. Is {dst} ever attainable by repeated transitions?",
    "Does repeated application of directed transitions allow attainment of {dst} from initial state {src}?",
    "Can an outbound process that originates in {src} terminate at {dst} using only permitted one-way moves?",
    "Can {dst} serve as the end state for a one-way process whose origin is {src}?",
    "If one starts with {src} and follows only successors, does any finite sequence finish at {dst}?",
    "Is there a finite predecessor-to-successor sequence ending at {dst} that begins with {src}?",
)

def require(condition,why):
    if not condition:
        raise AssertionError("S3BR TRAIN-only precommit "+why)

def build_rows():
    old=original_rows()
    require(parent_commit(old)["commitment_sha256"]==OLD_S3AQ_SHA,
            "original S3AQ training and heldout genealogy drift")
    earlier={s for pair in AQ_STYLES.values() for s in pair}
    earlier.update(BC_STYLES)
    earlier.update(AU_STYLES)
    earlier.update(AY_STYLES)
    require(len(STYLES)==8 and len(set(STYLES))==8
            and not set(STYLES)&earlier,
            "eight TRAIN-only constructions not genuinely new")
    forbidden={r.prompt_sha256 for r in old}
    by_graph=defaultdict(list)
    for r in old:
        if r.split==ORIGINAL_TRAIN:
            by_graph[r.graph_id].append(r)
    require(len(by_graph)==64 and all(len(g)==4 for g in by_graph.values()),
            "64 original TRAIN-only topologies not available")
    rng=random.Random(SEED)
    output=[]
    for graph_index,(graph_id,old_group) in enumerate(sorted(by_graph.items())):
        grp=sorted(old_group,key=lambda r:r.query_style)
        yes_rows=[r for r in grp if r.reachable]
        no_rows=[r for r in grp if not r.reachable]
        require(len(yes_rows)==len(no_rows)==2
                and len({(r.src,r.dst) for r in yes_rows})==1
                and len({(r.src,r.dst) for r in no_rows})==1
                and yes_rows[0].src==no_rows[0].dst
                and yes_rows[0].dst==no_rows[0].src,
                "old TRAIN graph must contain inverse source/target labels")
        for style,template in enumerate(STYLES):
            is_yes=((style//2)%2)==(graph_index%2)
            a=yes_rows[0] if is_yes else no_rows[0]
            arrows=[f"R{src} -> R{dst}" for src,dst in a.edges]
            rng.shuffle(arrows)
            head=HEADERS[(graph_index+style//2)%len(HEADERS)]
            question=head.format(edges="; ".join(arrows),
                                 query=template.format(src=f"R{a.src}",dst=f"R{a.dst}"))
            row=GraphPrompt(
                graph_id=f"{SPLIT}-{graph_id}",split=SPLIT,
                family=a.family,active_nodes=a.active_nodes,
                edges=a.edges,src=a.src,dst=a.dst,
                reachable=bool(is_yes),query_style=style,
                question=question)
            require(row.prompt_sha256 not in forbidden,
                    "TRAIN prompt overlaps old TRAIN/DEV/used EVAL")
            require(row.reachable==oracle(row.edges,row.src,row.dst),
                    "S3BR TRAIN graph truth mismatch")
            output.append(row)
    validate(output)
    return output

def validate(rows):
    require(len(rows)==512 and len({r.prompt_sha256 for r in rows})==512,
            "only 512 novel grammar TRAIN queries allowed")
    source=original_rows()
    tr={r.topology_key for r in source if r.split==ORIGINAL_TRAIN}
    forbidden_graph={r.topology_key for r in source
                     if r.split in (ORIGINAL_DEV,SPENT_OLD_EVAL)}
    forbidden_hashes={r.prompt_sha256 for r in source}
    require({r.topology_key for r in rows}==tr and not tr&forbidden_graph,
            "S3BR graph data must use ONLY original known TRAIN topologies")
    require(not ({r.prompt_sha256 for r in rows}&forbidden_hashes),
            "training prompt matches old precommitted DEV or spent EVAL")
    by_graph=defaultdict(list)
    for r in rows:
        require(r.split==SPLIT and r.family in FAMILIES
                and r.reachable==oracle(r.edges,r.src,r.dst),
                "TRAIN split or forward oracle incorrect")
        order=query_char_offsets(r.question)["text_order_nodes"]
        require(order==([r.dst,r.src] if r.query_style in DESTINATION_FIRST
                        else [r.src,r.dst]),
                "TRAIN literal role ordering does not match no-gold offsets")
        by_graph[r.graph_id].append(r)
    require(len(by_graph)==64,"S3BR requires all original 64 TRAIN topologies")
    for group in by_graph.values():
        group=sorted(group,key=lambda r:r.query_style)
        require([r.query_style for r in group]==list(range(8))
                and sum(r.reachable for r in group)==4
                and len({(r.src,r.dst) for r in group})==2,
                "TRAIN graph eight grammar constructions not role-balanced")
        for k in (0,2,4,6):
            require((group[k].src,group[k].dst)==(group[k+1].src,group[k+1].dst),
                    "TRAIN near-paraphrase must keep the same roles")
        for k in (0,2):
            require((group[k].src,group[k].dst)==(group[k+2].dst,group[k+2].src),
                    "TRAIN opposite relation directions not present")
    for i in range(8):
        chunk=[r for r in rows if r.query_style==i]
        require(len(chunk)==64 and sum(r.reachable for r in chunk)==32,
                "TRAIN style must never predict reachability label")
    require(sum(r.reachable for r in rows)==256
            and sum(r.query_style in DESTINATION_FIRST for r in rows)==256,
            "TRAIN class and entity-order balance incorrect")
    return True

def commitment(rows):
    validate(rows)
    c={
        "stage":"NRA2_S3BR_RELATION_LANGUAGE_TRAIN512_PREREG_ONLY",
        "seed":SEED,"split":SPLIT,
        "source_original_TRAIN_graphs_only":True,
        "S3AQ_original_parent_commitment_sha256":OLD_S3AQ_SHA,
        "S3BN_frozen_initial_candidate_sha256":S3BN_FROZEN_INIT_SHA,
        "prior_EVAL1_consumed_run":FIRST_OLD_EVAL_CONSUMED,
        "prior_EVAL2_consumed_run":SECOND_OLD_EVAL_CONSUMED,
        "prior_EVAL3_consumed_run":THIRD_OLD_EVAL_CONSUMED,
        "prior_EVALs_NEVER_model_input_nor_target_for_train":True,
        "new_relation_templates":list(STYLES),
        "new_relation_constructions":list(CONSTRUCTIONS),
        "TRAIN512_count":512,"TRAIN_graphs":64,
        "source_first_count":256,"destination_first_count":256,
        "yes_count":256,"no_count":256,
        "query_sha256_in_order":[r.prompt_sha256 for r in rows],
        "graph_sha256_in_order":[hashlib.sha256(
            json.dumps(sorted(r.edges)).encode()).hexdigest() for r in rows[::8]],
        "gold_sha256_in_order":hashlib.sha256(
            bytes([int(r.reachable) for r in rows])).hexdigest(),
        "no_cpu_tokens_or_weights_or_training_or_scoring_this_step":True,
        "protected_616_original_S194_J7_must_remain_frozen":True,
        "new_fourth_independent_EVAL_must_be_sealed_before_future_scoring":True,
        "output_gate":0.0,"candidate_accepted":False,
        "LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    c["commitment_sha256"]=hashlib.sha256(
        json.dumps(c,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return c

def main():
    require(len(sys.argv)==2,"CLI requires EMPTY output directory")
    out=Path(sys.argv[1])
    require(not out.exists(),"never overwrite S3BR TRAIN-only prereg")
    c=commitment(build_rows())
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3BR_DIRECTIONAL_TRAIN512_PREREG.json").write_text(
        json.dumps(c,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in c.items()
                      if k not in ("new_relation_templates","query_sha256_in_order",
                                   "graph_sha256_in_order")},
                     sort_keys=True),flush=True)

if __name__=="__main__":
    main()
