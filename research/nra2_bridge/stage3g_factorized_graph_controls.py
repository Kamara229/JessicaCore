"""NRA2 S3G: preregister factorized controlled research graph splits.

No model or weights loaded. Diagnose the confounded S3F shift:
S3F trained on 4/5-node familiar phrasings and evaluated only on
6-node novel phrasings. Here isolate structure size vs wording.
This is RESEARCH DIAGNOSTIC, not untouched independent final evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random

NODES = 6
SEED = 2026111403
GROUP_SIZES = {
    "TRAIN64": 64,
    "CTRL_MATCHED16": 16,      # familiar length + familiar wording
    "CTRL_WORDING16": 16,     # familiar length + novel wording
    "CTRL_LENGTH16": 16,      # 6 nodes + familiar wording
    "CTRL_BOTH16": 16,        # 6 nodes + novel wording
}
FAMILIAR = (
    "Route ledger states: {edges}. Following arrows, can {src} arrive at {dst}? Reply yes or no.",
    "Directed hops: {edges}. Is there a forward path from {src} to {dst}? Reply yes or no.",
)
NOVEL = (
    "Relay history reads: {edges}. Can {dst} be reached from {src} without reversing arrows? Reply yes or no.",
    "Transfer registry lists: {edges}. Is {dst} downstream of {src} under these one-way links? Reply yes or no.",
)


def digest(s: str) -> str:
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def oracle(edges, src, dst):
    seen = {src}
    stack = [src]
    while stack:
        cur = stack.pop()
        for a, b in edges:
            if a == cur and b not in seen:
                if b == dst:
                    return True
                seen.add(b)
                stack.append(b)
    return False


@dataclass(frozen=True)
class Row:
    question: str
    edges: tuple[tuple[int, int], ...]
    source: int
    target: int
    active_nodes: int
    group: str
    reachable: bool

    @property
    def prompt_sha256(self):
        return digest(self.question)


def build_rows():
    rng = random.Random(SEED)
    rows = []
    seen_topology = set()
    seen_questions = set()
    for group, count in GROUP_SIZES.items():
        for i in range(count):
            n = 6 if group in ("CTRL_LENGTH16", "CTRL_BOTH16") else (4 if (i // 2) % 2 == 0 else 5)
            for _ in range(2000):
                order = tuple(rng.sample(range(NODES), n))
                edges = tuple((order[j], order[j+1]) for j in range(n-1))
                identity = tuple(sorted(edges))
                if identity not in seen_topology:
                    seen_topology.add(identity)
                    break
            else:
                raise AssertionError("S3G unique edge topology exhausted")
            positive = (i % 2 == 0)
            src, dst = (order[0], order[-1]) if positive else (order[-1], order[0])
            if oracle(edges, src, dst) != positive:
                raise AssertionError("S3G ground truth invalid")
            links = [f"R{a} -> R{b}" for a, b in edges]
            rng.shuffle(links)
            templates = NOVEL if group in ("CTRL_WORDING16", "CTRL_BOTH16") else FAMILIAR
            tmpl = templates[(i // 2) % len(templates)]
            question = tmpl.format(edges="; ".join(links), src=f"R{src}", dst=f"R{dst}")
            if question in seen_questions:
                raise AssertionError("duplicate prompt")
            seen_questions.add(question)
            rows.append(Row(question, edges, src, dst, n, group, positive))
    validate(rows)
    return rows


def validate(rows):
    if len(rows) != 128 or len({x.prompt_sha256 for x in rows}) != 128:
        raise AssertionError("S3G requires 64 TRAIN and 4 x 16 unique controls")
    if len({tuple(sorted(x.edges)) for x in rows}) != 128:
        raise AssertionError("topology leakage between controlled groups")
    for group, count in GROUP_SIZES.items():
        subset = [x for x in rows if x.group == group]
        if len(subset) != count or sum(x.reachable for x in subset) != count // 2:
            raise AssertionError("S3G group must be balanced 50:50")
        lengths = {x.active_nodes for x in subset}
        if lengths != ({6} if group in ("CTRL_LENGTH16", "CTRL_BOTH16") else {4,5}):
            raise AssertionError("S3G length factor not controlled")
        templates = NOVEL if group in ("CTRL_WORDING16", "CTRL_BOTH16") else FAMILIAR
        for template in templates:
            selected = [x for x in subset if x.question.startswith(template.split("{edges}")[0])]
            if len(selected) != count // 2 or {x.reachable for x in selected} != {True, False}:
                raise AssertionError("S3G template shortcut or count drift")
        for x in subset:
            if oracle(x.edges, x.source, x.target) != x.reachable:
                raise AssertionError("S3G independent graph reachability drift")
            if x.prompt_sha256 in x.question:
                raise AssertionError("S3G prompt leaked SHA")
    return True
