"""S3T: freeze genuinely NEW query grammar and NEW graph topologies.

No CPU/model loaded, no adapter/NRA evaluation, no labels shown to model.
This is an exploratory *open* research generalization check, NOT final
sealed blind QA. Future evaluation is strictly read-only at gate 0.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import random
import re

from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    GraphPrompt, FAMILIES, HEADERS, QUERY_STYLES as OLD_QUERY_STYLES,
    make_graph, graph_has_cycle, build_rows as s3m_rows,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import (
    build_rows as s3g_rows, oracle,
)
from research.nra2_bridge.stage3q_independent_soft_hard_prereg import (
    build_rows as s3q_rows, LENGTHS,
)

SEED = 2026111423
GROUP = "S3T_NOVEL_GRAMMAR64"
NODES = 6
TOTAL_GRAPHS = 16
TOTAL_PROMPTS = 64
FROZEN_THRESHOLD = 0.5
# All four templates are novel relative to the exact S3M/S3Q train/eval.
# Intentional reversal of textual mention order (dst appears before src)
# in styles 0 and 3. No hard-coded semantic roles at inference.
QUERY_STYLES = (
    "Could you get to {dst} by following directed links starting at {src}?",
    "Beginning at {src}, would the arrows ever allow a visit to {dst}?",
    "Suppose the trip departs {src}; is arriving at {dst} possible?",
    "With one-way travel only, is {dst} accessible if departure is {src}?",
)
MENTION = re.compile(r"(?<![A-Za-z0-9_])R([0-5])(?![A-Za-z0-9_])")
S3O_FROZEN_FP16_SHA256 = "40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3R_NATIVE_CPU_FP16_SHA256 = "ff9c24d8ddb8106841caeb62c04301295b54b729466d933f2ac4656732a4e42c"
S3Q_FROZEN_COMMITMENT_SHA256 = "5e9ea9cd66d5dedf0aa2cf574dcf78c8e0a7522c892f775fd8e9e014c46a26cb"
S3S_SUCCESS_RUN = 43525997831
S3R_SUCCESS_RUN = 43525586699


def query_char_offsets(question):
    """Literal alignment-only metadata, never hidden-feature or gold labels."""
    q_end = question.find("?")
    delim = question.rfind(". ", 0, q_end)
    if q_end < 0 or question.find("?", q_end+1) != -1 or delim < 0:
        raise AssertionError("S3T S3O ABI: ambiguous final interrogative")
    start = delim+2
    hits = [m for m in MENTION.finditer(question)
            if start <= m.start() and m.end() <= q_end]
    if len(hits) != 2 or hits[0].group(1) == hits[1].group(1):
        raise AssertionError("S3T S3O ABI: two distinct query mentions required")
    return {"clause": [start, q_end+1],
            "mention_spans": [[m.start(),m.end()] for m in hits],
            "text_order_nodes": [int(m.group(1)) for m in hits]}


def build_rows():
    rng = random.Random(SEED)
    previous = s3m_rows()
    old_q = s3q_rows()
    used = {r.topology_key for r in previous}
    used.update(tuple(sorted(r.edges)) for r in s3g_rows())
    used.update(r.topology_key for r in old_q)
    if set(QUERY_STYLES) & set(OLD_QUERY_STYLES):
        raise AssertionError("S3T query wording not new")
    rows=[]
    for fi,family in enumerate(FAMILIES):
        for j,n in enumerate(LENGTHS[fi]):
            for _ in range(15000):
                order=tuple(rng.sample(range(NODES),n))
                edges=make_graph(order,family)
                key=tuple(sorted(edges))
                possible=[(a,b) for a in order for b in order if a!=b
                          and oracle(edges,a,b) and not oracle(edges,b,a)]
                if key not in used and possible:
                    used.add(key)
                    break
            else:
                raise AssertionError("S3T no unseen topology left")
            forward=rng.choice(possible)
            reverse=forward[1],forward[0]
            for style,template in enumerate(QUERY_STYLES):
                yes=(style%2)==(j%2)
                src,dst=forward if yes else reverse
                if oracle(edges,src,dst)!=yes:
                    raise AssertionError("S3T independent BFS mismatch")
                edge_text=[f"R{a} -> R{b}" for a,b in edges]
                rng.shuffle(edge_text)
                question=HEADERS[(j//2+style)%len(HEADERS)].format(
                    edges="; ".join(edge_text),
                    query=template.format(src=f"R{src}",dst=f"R{dst}"))
                rows.append(GraphPrompt(
                    graph_id=f"S3T-{family}-{j:02}",split=GROUP,
                    family=family,active_nodes=n,edges=edges,
                    src=src,dst=dst,reachable=yes,query_style=style,
                    question=question))
    validate(rows)
    return rows


def validate(rows):
    previous=s3m_rows()
    s3q=s3q_rows()
    old_keys={r.topology_key for r in previous}
    old_keys.update(tuple(sorted(r.edges)) for r in s3g_rows())
    old_keys.update(r.topology_key for r in s3q)
    old_hashes={r.prompt_sha256 for r in previous}
    old_hashes.update(r.prompt_sha256 for r in s3g_rows())
    old_hashes.update(r.prompt_sha256 for r in s3q)
    if len(rows)!=TOTAL_PROMPTS or len({r.prompt_sha256 for r in rows})!=TOTAL_PROMPTS:
        raise AssertionError("S3T expected 64 unique locked questions")
    if {r.prompt_sha256 for r in rows}&old_hashes:
        raise AssertionError("S3T question leakage")
    groups=defaultdict(list)
    for r in rows:
        if r.split!=GROUP or r.family not in FAMILIES or r.topology_key in old_keys:
            raise AssertionError("S3T split/topology overlap")
        if graph_has_cycle(r.edges)!=(r.family=="cycle_sink"):
            raise AssertionError("S3T graph family drift")
        if len({x for e in r.edges for x in e})!=r.active_nodes:
            raise AssertionError("S3T node cardinality drift")
        if oracle(r.edges,r.src,r.dst)!=r.reachable:
            raise AssertionError("S3T BFS label mismatch")
        offsets=query_char_offsets(r.question)
        if (set(offsets["text_order_nodes"])!={r.src,r.dst}
            or (offsets["text_order_nodes"][0]==r.dst)!=(r.query_style in (0,3))):
            raise AssertionError("S3T query roles or text order drift")
        groups[r.graph_id].append(r)
    if (len(groups)!=TOTAL_GRAPHS
        or len({r.topology_key for r in rows})!=TOTAL_GRAPHS
        or sum(r.reachable for r in rows)!=32):
        raise AssertionError("S3T topology/label balance")
    for group in groups.values():
        if (len(group)!=4 or sum(r.reachable for r in group)!=2
            or {r.query_style for r in group}!={0,1,2,3}
            or len({r.topology_key for r in group})!=1):
            raise AssertionError("S3T 4-question paired graph invalid")
        pairs={(r.src,r.dst) for r in group}
        if len(pairs)!=2 or any((b,a) not in pairs for a,b in pairs):
            raise AssertionError("S3T reverse directions missing")
    for family in FAMILIES:
        group=[r for r in rows if r.family==family]
        if len(group)!=16 or sum(r.reachable for r in group)!=8:
            raise AssertionError("S3T per-family balance")
        for style in range(4):
            selection=[r for r in group if r.query_style==style]
            if len(selection)!=4 or sum(r.reachable for r in selection)!=2:
                raise AssertionError("S3T new wording truth shortcut")
    for style in range(4):
        for n in (4,5,6):
            g=[r for r in rows if r.query_style==style and r.active_nodes==n]
            if not g or 2*sum(r.reachable for r in g)!=len(g):
                raise AssertionError("S3T length/grammar truth shortcut")
        for header in HEADERS:
            prefix=header.partition("{edges}")[0]
            g=[r for r in rows if r.query_style==style and r.question.startswith(prefix)]
            if not g or 2*sum(r.reachable for r in g)!=len(g):
                raise AssertionError("S3T header/grammar truth shortcut")
    return True


def commitment(rows):
    validate(rows)
    audit={
        "stage":"NRA2_S3T_NEW_GRAMMAR64_PREDECLARED_ONLY_NO_SCORING",
        "seed":SEED,"split":GROUP,
        "questions":TOTAL_PROMPTS,"graph_topologies":TOTAL_GRAPHS,
        "families":{family:16 for family in FAMILIES},
        "yes":32,"no":32,
        "new_wording_disjoint_from_S3M_S3Q":True,
        "new_graph_topologies_disjoint_from_S3G_S3M_S3Q":True,
        "query_char_spans_frozen_without_tokenizer":True,
        "S3O_actual_native_token_alignment_pending":True,
        "S3O_frozen_adapter_fp16_sha256":S3O_FROZEN_FP16_SHA256,
        "S3R_original_43M_cpu_fp16_sha256":S3R_NATIVE_CPU_FP16_SHA256,
        "S3Q_locked_commitment_sha256":S3Q_FROZEN_COMMITMENT_SHA256,
        "S3S_successful_prior_run":S3S_SUCCESS_RUN,
        "S3R_successful_original_616_J7_run":S3R_SUCCESS_RUN,
        "edge_hard_threshold":FROZEN_THRESHOLD,
        "role_probabilities":"soft in both arms",
        "model_and_NRA_no_fitting_or_tuning":True,
        "candidate_predicted_graph_scores_not_inspected":True,
        "native_original_616_J7_digest_check_required":True,
        "new_cpu_features_not_captured":True,
        "gate":0.0,"no_live_render_supabase_main_apk_jessica":True,
        "not_sealed_final_blind":True,"candidate_accepted":False,
        "new_decoder_full_EOS_not_tested":True,
        "physical_all_in_bpp_not_tested":True,
        "CPU_speed_not_tested":True,
        "question_sha256":[r.prompt_sha256 for r in rows],
        "topology_sha256":[hashlib.sha256(json.dumps(sorted(r.edges)).encode()).hexdigest()
                           for r in rows[::4]],
        "query_char_offsets":[query_char_offsets(r.question) for r in rows],
    }
    data=json.dumps(audit,sort_keys=True,separators=(",",":")).encode()
    audit["commitment_sha256"]=hashlib.sha256(data).hexdigest()
    return audit


if __name__=="__main__":
    result=commitment(build_rows())
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("question_sha256","topology_sha256","query_char_offsets")},
                     sort_keys=True),flush=True)
