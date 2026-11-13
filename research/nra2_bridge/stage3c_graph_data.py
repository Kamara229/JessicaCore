"""S3C declared TRAIN-only directed graph pairs, original frozen 43M CPU.

Generates independent relational tasks with source, target and adjacency
labels. 32 training graph prompts (4/5 nodes) plus 16 *structurally new*
held-out graphs (six active nodes, disjoint edge arrangements and templates).
No gold answer supplied to decoder; no QA/TEST. These are *synthetic* graphs,
not proof of natural-language reasoning or generalization. Source-first
coordinates are canonical R0..R5. No learned model weights are modified.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

import torch

NODE_COUNT = 6
TRAIN_COUNT = 32
HELD_COUNT = 16
TOTAL_COUNT = TRAIN_COUNT + HELD_COUNT
TRAIN_TEMPLATES = (
    "Route book: {links}. Following only these directed links, can {src} reach {dst}? Answer Yes or No.",
    "Ordered relay: {links}. Is there a path from {src} forward to {dst}? Answer Yes or No.",
    "Transfer chain: {links}. If movement follows the recorded arrows, does {src} lead to {dst}? Answer Yes or No.",
    "Dependency notes: {links}. Can you follow one or more arrows from {src} to {dst}? Answer Yes or No.",
)
HELD_TEMPLATES = (
    "Unseen route layout: {links}. With direction enforced, is {dst} downstream of {src}? Reply Yes or No.",
    "Novel succession chain: {links}. Is it possible to get to {dst} by walking forward from {src}? Reply Yes or No.",
)


def sha_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class GraphRecord:
    question: str
    edges: tuple[tuple[int,int], ...]
    source: int
    target: int
    active_count: int
    split: str
    reachable: bool

    @property
    def prompt_sha(self):
        return sha_text(self.question)


def reachable(edges,src,dst):
    stack=[src]
    seen={src}
    while stack:
        n=stack.pop()
        for a,b in edges:
            if a==n and b not in seen:
                if b==dst:
                    return True
                seen.add(b)
                stack.append(b)
    return False


def build_graph_rows()->list[GraphRecord]:
    """Structure-disjoint split fixed BEFORE model inference/training."""
    rng=random.Random(202611130317)
    rows=[]
    seen_prompt=set()
    seen_topology=set()
    for i in range(TOTAL_COUNT):
        held=i>=TRAIN_COUNT
        active=NODE_COUNT if held else 4+i%2
        for _ in range(500):
            order=tuple(rng.sample(range(NODE_COUNT),active))
            topology=tuple((order[j],order[j+1]) for j in range(active-1))
            # Both direction and active-count are part of the topology.
            if topology not in seen_topology:
                seen_topology.add(topology)
                break
        else:
            raise AssertionError("Could not build structurally distinct graph")
        src,dst=(order[0],order[-1]) if i%2==0 else (order[-1],order[0])
        ans=reachable(topology,src,dst)
        if ans != (i%2==0):
            raise AssertionError("Graph target reached against declared direction")
        # Shuffle the prose edge order independently of graph topological order.
        mentions=[f"R{a} -> R{b}" for a,b in topology]
        rng.shuffle(mentions)
        links="; ".join(mentions)
        # Pair each wording template with BOTH labels to prohibit answer
        # prediction from phrasing alone (previous i%2 mapping was unsafe).
        tmpl=(HELD_TEMPLATES[((i-TRAIN_COUNT)//2)%len(HELD_TEMPLATES)]
              if held else TRAIN_TEMPLATES[(i//2)%len(TRAIN_TEMPLATES)])
        question=tmpl.format(links=links,src=f"R{src}",dst=f"R{dst}")
        if question in seen_prompt or sha_text(question) in seen_prompt:
            raise AssertionError("Duplicate graph question")
        seen_prompt.add(question)
        seen_prompt.add(sha_text(question))
        rows.append(GraphRecord(question,topology,src,dst,active,
                                "heldout_novel_structure" if held else "train",
                                ans))
    validate_graph_rows(rows)
    return rows


def validate_graph_rows(rows:list[GraphRecord]):
    if len(rows)!=TOTAL_COUNT or len({r.prompt_sha for r in rows})!=TOTAL_COUNT:
        raise AssertionError("Expected 48 unique paired real graph prompts")
    topology_train={r.edges for r in rows[:TRAIN_COUNT]}
    topology_held={r.edges for r in rows[TRAIN_COUNT:]}
    if topology_train&topology_held:
        raise AssertionError("Graph topology leaked across split")
    if len(topology_train)!=TRAIN_COUNT or len(topology_held)!=HELD_COUNT:
        raise AssertionError("Graphs must have unique relation topology")
    # A shortcut that maps prompt wording directly to answer must fail:
    # every declared wording template contains BOTH reachable outcomes.
    for is_held,templates in ((False,TRAIN_TEMPLATES),(True,HELD_TEMPLATES)):
        subset=rows[TRAIN_COUNT:] if is_held else rows[:TRAIN_COUNT]
        for tmpl in templates:
            # The fixed prefix uniquely identifies each declared format.
            prefix=tmpl.split("{links}")[0]
            selected=[r for r in subset if r.question.startswith(prefix)]
            if not selected or {r.reachable for r in selected}!={True,False}:
                raise AssertionError("Template wording leaked answer direction")
    for i,r in enumerate(rows):
        held=i>=TRAIN_COUNT
        if (r.split!="heldout_novel_structure")!= (not held):
            raise AssertionError("Split identity drift")
        if r.active_count!=(6 if held else 4+i%2):
            raise AssertionError("New structured holdout must require longer paths")
        if len(r.edges)!=r.active_count-1 or len(set(r.edges))!=len(r.edges):
            raise AssertionError("Graph must be a directed acyclic chain")
        if any(not (0<=a<NODE_COUNT and 0<=b<NODE_COUNT) or a==b for a,b in r.edges):
            raise AssertionError("Invalid node relation")
        if reachable(r.edges,r.source,r.target)!=r.reachable:
            raise AssertionError("Independent graph reachability truth drift")
        if r.reachable!=(i%2==0):
            raise AssertionError("Balanced positive and negative")
        if f"R{r.source}" not in r.question or f"R{r.target}" not in r.question:
            raise AssertionError("Source/target never described in question")
    if sum(x.reachable for x in rows[:TRAIN_COUNT])!=16 or sum(
        x.reachable for x in rows[TRAIN_COUNT:])!=8:
        raise AssertionError("TRAIN/heldout exact class balance required")


def build_graph_targets(rows:list[GraphRecord]):
    """Construct labels only AFTER completely frozen CPU prefill capture."""
    validate_graph_rows(rows)
    edges=torch.zeros((TOTAL_COUNT,NODE_COUNT,NODE_COUNT),dtype=torch.float16)
    src=torch.zeros((TOTAL_COUNT,NODE_COUNT),dtype=torch.float16)
    dst=torch.zeros_like(src)
    answers=torch.zeros(TOTAL_COUNT,dtype=torch.int32)
    for i,r in enumerate(rows):
        for a,b in r.edges:
            edges[i,a,b]=1
        src[i,r.source]=1
        dst[i,r.target]=1
        answers[i]=int(r.reachable)
    return {"adjacency":edges,"source":src,"target":dst,
            "reachability":answers}


def no_answer_template_lookup(rows):
    """Declared text never embeds SHA, expected labels or answer words."""
    for r in rows:
        if r.question.count("Answer Yes or No.")+r.question.count("Reply Yes or No.")!=1:
            raise AssertionError("Graph prompt must not reveal answer")
        if r.prompt_sha in r.question or "correct answer" in r.question.lower():
            raise AssertionError("S3C question leaked answer key")
    return True
