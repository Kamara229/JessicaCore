"""NRA2 S3M: counterbalanced multi-query directed graphs for endpoint roles.

Entirely synthetic PUBLIC RESEARCH: no model weights, inference, labels to
any neural forward pass, LIVE/QA/TEST. 32 TRAIN graph topologies x 4 queries
and 16 exploratory DEV topologies x 4 queries. The 64 DEV queries have NOT
yet been evaluated, but they are not cryptographically blinded. The later
independent FINAL must be generated/frozen separately, not declared solved
from this data. Graphs include chains, branches, cycles and diamonds.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

from research.nra2_bridge.stage3g_factorized_graph_controls import (
    build_rows as s3g_rows, oracle,
)

SEED=2026111409
NODE_POOL=6
FAMILIES=("chain","fork","cycle_sink","diamond")
TRAIN_TOPOLOGIES_PER_FAMILY=8
DEV_TOPOLOGIES_PER_FAMILY=4
QUERY_STYLES=(
    "Is there a directed path from {src} to {dst}?",
    "Can {dst} be reached from {src} following the arrows?",
    "Starting at {src}, can one travel to {dst}?",
    "Is {dst} downstream of {src} along one-way links?",
)
HEADERS=(
    "Connection log: {edges}. {query} Reply yes or no.",
    "One-way route list: {edges}. {query} Answer yes or no.",
)


@dataclass(frozen=True)
class GraphPrompt:
    graph_id: str
    split: str
    family: str
    active_nodes: int
    edges: tuple[tuple[int,int],...]
    src: int
    dst: int
    reachable: bool
    query_style: int
    question: str

    @property
    def prompt_sha256(self):
        return hashlib.sha256(self.question.encode("utf-8")).hexdigest()

    @property
    def topology_key(self):
        return tuple(sorted(self.edges))


def make_graph(order, family):
    n=len(order)
    if n not in (4,5,6):
        raise AssertionError("S3M active node count must be 4,5 or 6")
    if family=="chain":
        edges=[(order[i],order[i+1]) for i in range(n-1)]
    elif family=="fork":
        edges=[(order[0],order[1]),(order[0],order[2]),
               (order[1],order[3])]
        edges.extend((order[i],order[i+1]) for i in range(3,n-1))
    elif family=="cycle_sink":
        edges=[(order[0],order[1]),(order[1],order[2]),
               (order[2],order[0]),(order[2],order[3])]
        edges.extend((order[i],order[i+1]) for i in range(3,n-1))
    elif family=="diamond":
        edges=[(order[0],order[1]),(order[0],order[2]),
               (order[1],order[3]),(order[2],order[3])]
        edges.extend((order[i],order[i+1]) for i in range(3,n-1))
    else:
        raise AssertionError("S3M unknown graph family")
    if len(edges)!=len(set(edges)) or any(a==b for a,b in edges):
        raise AssertionError("S3M duplicate edge or self loop")
    return tuple(edges)


def graph_has_cycle(edges):
    nodes={n for edge in edges for n in edge}
    seen=set()
    active=set()
    def dfs(x):
        if x in active:
            return True
        if x in seen:
            return False
        seen.add(x)
        active.add(x)
        for a,b in edges:
            if a==x and dfs(b):
                return True
        active.remove(x)
        return False
    return any(dfs(n) for n in nodes if n not in seen)


def build_rows():
    rng=random.Random(SEED)
    previous={tuple(sorted(r.edges)) for r in s3g_rows()}
    used=set(previous)
    out=[]
    for split,per_family in (("TRAIN128",TRAIN_TOPOLOGIES_PER_FAMILY),
                             ("DEV64",DEV_TOPOLOGIES_PER_FAMILY)):
        for family_index,family in enumerate(FAMILIES):
            for j in range(per_family):
                # Strict anti-shortcut counterbalance: every active-node
                # length occurs with BOTH yes and no in each split.
                # TRAIN: paired j even/odd hold the same n in each family.
                # DEV: rotated family schedules give n4/n5/n6 all both
                # truth values across the four topology families.
                if per_family==TRAIN_TOPOLOGIES_PER_FAMILY:
                    n=(4,4,5,5,6,6,4,4)[j]
                else:
                    n=((4,5,6,4) if family_index%2==0 else (4,6,5,4))[j]
                for _ in range(5000):
                    order=tuple(rng.sample(range(NODE_POOL),n))
                    edges=make_graph(order,family)
                    key=tuple(sorted(edges))
                    if key not in used:
                        possible=[
                            (a,b) for a in order for b in order if a!=b
                            and oracle(edges,a,b)
                            and not oracle(edges,b,a)
                        ]
                        if possible:
                            used.add(key)
                            break
                else:
                    raise AssertionError("S3M unique directed graph topology exhausted")
                forward=rng.choice(possible)
                reverse=(forward[1],forward[0])
                graph_id=f"{split}-{family}-{j:02}"
                # Alternating parity makes every fixed query grammar occur
                # equally often with yes and no, including within family.
                parity=j%2
                for style,template in enumerate(QUERY_STYLES):
                    yes=(style%2)==parity
                    src,dst=forward if yes else reverse
                    if oracle(edges,src,dst)!=yes:
                        raise AssertionError("S3M independent BFS oracle failed")
                    strings=[f"R{a} -> R{b}" for a,b in edges]
                    rng.shuffle(strings)
                    q=template.format(src=f"R{src}",dst=f"R{dst}")
                    # IMPORTANT: (j+style)%2 was exactly the TRUE label.
                    # Never let a header alone give away reachability.
                    heading=HEADERS[(j//2+style)%len(HEADERS)]
                    full=heading.format(edges="; ".join(strings),query=q)
                    out.append(GraphPrompt(graph_id,split,family,n,edges,
                                           src,dst,yes,style,full))
    validate(out)
    return out


def validate(rows):
    if len(rows)!=192 or len({x.question for x in rows})!=192:
        raise AssertionError("S3M requires 192 distinct 128/64 balanced prompts")
    by_graph={}
    for row in rows:
        by_graph.setdefault(row.graph_id,[]).append(row)
        if oracle(row.edges,row.src,row.dst)!=row.reachable:
            raise AssertionError("S3M independent BFS reachability label mismatch")
        if row.family=="cycle_sink" and not graph_has_cycle(row.edges):
            raise AssertionError("S3M cycle family missing directed cycle")
        if row.family!="cycle_sink" and graph_has_cycle(row.edges):
            raise AssertionError("S3M acyclic family unexpectedly cyclic")
        if row.active_nodes!=len(set(v for edge in row.edges for v in edge)):
            raise AssertionError("S3M wrong active node cardinality")
    if len(by_graph)!=48:
        raise AssertionError("S3M must have exactly 48 graph topologies")
    for name,examples in by_graph.items():
        if (len(examples)!=4 or len({x.query_style for x in examples})!=4
            or sum(x.reachable for x in examples)!=2
            or len({x.topology_key for x in examples})!=1
            or len({x.split for x in examples})!=1):
            raise AssertionError("S3M graph pairs must have independent counterbalanced questions")
        pairs={(x.src,x.dst) for x in examples}
        if len(pairs)!=2 or not all((b,a) in pairs for a,b in pairs):
            raise AssertionError("S3M graphs must have opposite query directions")
    if len({example[0].topology_key for example in by_graph.values()})!=48:
        raise AssertionError("S3M topology leakage across splits")
    for split,total,per_family in (("TRAIN128",128,8),("DEV64",64,4)):
        group=[x for x in rows if x.split==split]
        if len(group)!=total or sum(x.reachable for x in group)!=total//2:
            raise AssertionError("S3M split counts or label balance invalid")
        for n in (4,5,6):
            # Literal prompt length must not itself determine truth.
            for style in range(4):
                subset=[x for x in group if x.active_nodes==n and x.query_style==style]
                if (not subset or len(subset)%2
                    or sum(x.reachable for x in subset)!=len(subset)//2):
                    raise AssertionError("S3M graph length / query template is a yes/no shortcut")
        for style in range(4):
            for header in HEADERS:
                prefix=header.split("{edges}")[0]
                subset=[x for x in group if x.query_style==style and x.question.startswith(prefix)]
                if (not subset or len(subset)%2
                    or sum(x.reachable for x in subset)!=len(subset)//2):
                    raise AssertionError("S3M header / query template leaks yes/no label")
        for family in FAMILIES:
            subset=[x for x in group if x.family==family]
            if len(subset)!=4*per_family or {x.active_nodes for x in subset}!={4,5,6}:
                raise AssertionError("S3M family or length diversity missing")
            for style in range(4):
                positives=[x.reachable for x in subset if x.query_style==style]
                if len(positives)!=per_family or sum(positives)!=per_family//2:
                    raise AssertionError("S3M template-label shortcut detected")
    return True


if __name__=="__main__":
    rows=build_rows()
    print("S3M_PREDECLARED_RESEARCH_ONLY_128TRAIN_64DEV",
          {"prompts":len(rows),
           "train":sum(r.split=="TRAIN128" for r in rows),
           "dev":sum(r.split=="DEV64" for r in rows),
           "topologies":len({r.topology_key for r in rows}),
           "families":FAMILIES,
           "final_blind_created":False})
