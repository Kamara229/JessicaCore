"""S3AA: precommit NEW 128 questions on 32 unseen graph topologies.

Independent topology + wording research check AFTER S3Z role-head TRAIN256.
No weights/tokenizers/CPU loaded, no scores inspected, no model run.
Public research prereg is NOT a sealed secret final QA/blind benchmark.
User's ordinary Jessica chat claim of 'registered' is NOT evidence of
GitHub registration and is never treated as a test result.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
import random

from research.nra2_bridge.stage3m_role_balanced_graph_data import (
    GraphPrompt,FAMILIES,HEADERS,QUERY_STYLES as S3M_STYLES,
    make_graph,graph_has_cycle,build_rows as s3m_rows,
)
from research.nra2_bridge.stage3g_factorized_graph_controls import (
    build_rows as s3g_rows,oracle,
)
from research.nra2_bridge.stage3q_independent_soft_hard_prereg import (
    build_rows as s3q_rows,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import (
    build_rows as s3t_rows,QUERY_STYLES as S3T_STYLES,query_char_offsets,
)
from research.nra2_bridge.stage3x_reversal_train_prereg import (
    build_rows as s3x_rows,NEW_STYLES as S3X_STYLES,
)

SEED=2026111429
GROUP="S3AA_INDEPENDENT_NEWGRAMMAR128"
GRAPH_COUNT=32
PROMPTS=128
PER_FAMILY=8
NODES=6
REVERSED_STYLES=(0,2)
LENGTHS=(4,4,5,5,6,6,4,4)
QUERY_STYLES=(
    "If {dst} is specified as the endpoint, may one depart from {src} and arrive there using directed edges?",
    "Does any arrow-following route begin at {src} and terminate at {dst}?",
    "Could destination {dst} be reached by a one-way journey originating at {src}?",
    "When the journey starts from {src}, is it possible to finish at {dst} by traversing the arrows?",
)
S3Z_FP16_SHA256="60a7121b8270f647909776b58dc21b07c6cb885fd6ebeb6cb3158b23221f6e9f"
S3Z_SUCCESSFUL_CI=43532157094
S3Y_SUCCESSFUL_CI=43530742301
S3X_PREREG_SUCCESSFUL_CI=43530401603
S3O_FROZEN_FP16_SHA256="40a8903d5419cd8641b5d01e75e429e10b449304b3ba140d2245bf7f8a9d00b6"
S3T_PRIOR_SCORED_COMMIT_SHA="034ebd6e1cf6f20ad04ccd00e70c6bc07ee08a46f47da843121524f1d7f0faf3"
HARD_EDGE_THRESHOLD=0.5
# Fixed AFTER seeing S3W+S3Z exploratory outcomes, but BEFORE any S3AA
# original CPU inference, S3AA adapter score, threshold tuning or acceptance.
METRICS=(
    "S3O_vs_S3Z_source_role_exact",
    "S3O_vs_S3Z_destination_role_exact",
    "S3O_vs_S3Z_reachability_hard_edges_soft_roles",
    "original_S3O_vs_S3Z_both_roles_correct",
    "graph_exact_by_original_frozen_S3O_edge_head",
    "original_616_S194_J7_digest_unchanged",
)


def previous_rows():
    return (s3g_rows(),s3m_rows(),s3q_rows(),s3t_rows(),s3x_rows())


def build_rows():
    prior=previous_rows()
    used={r.topology_key for group in prior for r in group}
    old_sha={r.prompt_sha256 for group in prior for r in group}
    old_templates=set(S3M_STYLES)|set(S3T_STYLES)|set(S3X_STYLES)
    if len(set(QUERY_STYLES))!=4 or any(x in old_templates for x in QUERY_STYLES):
        raise AssertionError("S3AA four query phrasings must be novel exact grammar")
    rng=random.Random(SEED)
    out=[]
    for family in FAMILIES:
        for j,n in enumerate(LENGTHS):
            for _ in range(20000):
                order=tuple(rng.sample(range(NODES),n))
                edges=make_graph(order,family)
                key=tuple(sorted(edges))
                possibilities=[(a,b) for a in order for b in order
                               if a!=b and oracle(edges,a,b) and not oracle(edges,b,a)]
                if key not in used and possibilities:
                    used.add(key)
                    break
            else:
                raise AssertionError("S3AA exhausted disjoint directed topologies")
            forward=rng.choice(possibilities)
            backward=(forward[1],forward[0])
            for style,template in enumerate(QUERY_STYLES):
                yes=(style%2)==(j%2)
                src,dst=forward if yes else backward
                if oracle(edges,src,dst)!=yes:
                    raise AssertionError("S3AA independent directed BFS truth mismatch")
                links=[f"R{a} -> R{b}" for a,b in edges]
                rng.shuffle(links)
                question=HEADERS[(j//2+style)%len(HEADERS)].format(
                    edges="; ".join(links),
                    query=template.format(src=f"R{src}",dst=f"R{dst}"))
                row=GraphPrompt(
                    graph_id=f"S3AA-{family}-{j:02}",split=GROUP,
                    family=family,active_nodes=n,edges=edges,
                    src=src,dst=dst,reachable=yes,query_style=style,
                    question=question,
                )
                if row.prompt_sha256 in old_sha:
                    raise AssertionError("S3AA prompt duplicates scored or trained text")
                out.append(row)
    validate(out)
    return out


def validate(rows):
    prior=previous_rows()
    prior_topology={r.topology_key for group in prior for r in group}
    prior_sha={r.prompt_sha256 for group in prior for r in group}
    if (len(rows)!=PROMPTS
        or len({r.prompt_sha256 for r in rows})!=PROMPTS
        or any(r.prompt_sha256 in prior_sha for r in rows)):
        raise AssertionError("S3AA 128 exact unique previously unscored prompts required")
    graphs=defaultdict(list)
    for r in rows:
        if (r.split!=GROUP or r.family not in FAMILIES
            or r.topology_key in prior_topology
            or oracle(r.edges,r.src,r.dst)!=r.reachable
            or graph_has_cycle(r.edges)!=(r.family=="cycle_sink")
            or len({v for e in r.edges for v in e})!=r.active_nodes):
            raise AssertionError("S3AA topology/label/graph-family leakage")
        spans=query_char_offsets(r.question)
        if (set(spans["text_order_nodes"])!={r.src,r.dst}
            or (spans["text_order_nodes"][0]==r.dst)!=(r.query_style in REVERSED_STYLES)):
            raise AssertionError("S3AA semantic destination/source mention order invalid")
        graphs[r.graph_id].append(r)
    if (len(graphs)!=GRAPH_COUNT
        or len({r.topology_key for r in rows})!=GRAPH_COUNT
        or sum(r.reachable for r in rows)!=PROMPTS//2):
        raise AssertionError("S3AA topology counts or globally balanced yes/no invalid")
    for group in graphs.values():
        pairs={(r.src,r.dst) for r in group}
        if (len(group)!=4 or len(pairs)!=2
            or not all((b,a) in pairs for a,b in pairs)
            or len({r.topology_key for r in group})!=1
            or {r.query_style for r in group}!={0,1,2,3}
            or sum(r.reachable for r in group)!=2):
            raise AssertionError("S3AA all four paired questions per unseen graph required")
    for fam in FAMILIES:
        gr=[r for r in rows if r.family==fam]
        if len(gr)!=32 or sum(r.reachable for r in gr)!=16:
            raise AssertionError("S3AA family yes/no imbalance")
        for style in range(4):
            selection=[r for r in gr if r.query_style==style]
            if len(selection)!=8 or sum(r.reachable for r in selection)!=4:
                raise AssertionError("S3AA family/wording answer shortcut")
    for style in range(4):
        for length in (4,5,6):
            group=[r for r in rows if r.query_style==style and r.active_nodes==length]
            if not group or 2*sum(r.reachable for r in group)!=len(group):
                raise AssertionError("S3AA prompt length/wording label shortcut")
        for header in HEADERS:
            prefix=header.split("{edges}")[0]
            group=[r for r in rows if r.query_style==style and r.question.startswith(prefix)]
            if not group or 2*sum(r.reachable for r in group)!=len(group):
                raise AssertionError("S3AA header/wording label shortcut")
    for textorder,styles in (("dst_first",REVERSED_STYLES),
                             ("src_first",(1,3))):
        group=[r for r in rows if r.query_style in styles]
        if (len(group)!=64 or sum(r.reachable for r in group)!=32):
            raise AssertionError("S3AA mention order shortcut: "+textorder)
    return True


def commitment(rows):
    validate(rows)
    data={
        "stage":"NRA2_S3AA_INDEPENDENT_128_NEW_TOPOLOGY_NEW_GRAMMAR_PREREG_NO_CPU_NO_SCORE",
        "seed":SEED,"group":GROUP,"question_count":PROMPTS,
        "graph_topology_count":GRAPH_COUNT,
        "graph_families":{fam:32 for fam in FAMILIES},
        "yes_count":64,"no_count":64,
        "distinct_node_counts":[4,5,6],
        "text_mention_order":{"destination_first":64,"source_first":64},
        "new_four_wordings":list(QUERY_STYLES),
        "novel_grammar_disjoint_from_S3M_S3Q_S3T_S3X":True,
        "novel_graph_topologies_disjoint_from_S3G_S3M_S3Q_S3T_and_S3X_TRAIN":True,
        "S3X_TRAIN_reuses_S3M_topologies_but_S3AA_is_disjoint":True,
        "S3T_already_scored_previously_not_a_blind":True,
        "S3T_prior_scored_SHA":S3T_PRIOR_SCORED_COMMIT_SHA,
        "S3Z_role_head_physical_fp16_sha256_pinned":S3Z_FP16_SHA256,
        "S3O_original_frozen_fp16_sha256_pinned":S3O_FROZEN_FP16_SHA256,
        "S3Z_successful_prior_training_run":S3Z_SUCCESSFUL_CI,
        "S3Y_successful_prior_original_CPU_run":S3Y_SUCCESSFUL_CI,
        "S3X_successful_prior_TRAIN128_prereg_run":S3X_PREREG_SUCCESSFUL_CI,
        "compare_exactly_original_S3O_vs_new_S3Z_same_CPU_same_graph_same_NRA":True,
        "fixed_hard_edge_threshold":HARD_EDGE_THRESHOLD,
        "frozen_edge_graph_exact_same_between_both_candidate_arms":True,
        "role_probabilities_soft_in_both_arms":True,
        "evaluation_metrics_frozen_before_capture":list(METRICS),
        "evaluate_by_family_wording_text_mention_order_and_true_label":True,
        "independent_exact_original_S194_616_J7_digest_required":True,
        "new_real_43M_CPU_capture_not_started":True,
        "new_candidate_S3AA_predictions_not_inspected":True,
        "no_additional_training_on_S3AA":True,
        "no_threshold_selection_or_retune_on_S3AA":True,
        "public_research_precommit_not_cryptographically_hidden_blind":True,
        "original_decoder_full_EOS_gain_not_yet_tested":True,
        "original_all_in_bpp_and_CPU_speed_not_tested":True,
        "candidate_accepted":False,"gate":0.0,
        "LIVE_Render_Supabase_APK_main_QA_TEST_Jessica_unchanged":True,
        "question_sha256":[r.prompt_sha256 for r in rows],
        "topology_sha256":[hashlib.sha256(json.dumps(sorted(r.edges)).encode()).hexdigest()
                           for r in rows[::4]],
        "query_char_offsets":[query_char_offsets(r.question) for r in rows],
    }
    payload=json.dumps(data,sort_keys=True,separators=(",",":")).encode()
    data["commitment_sha256"]=hashlib.sha256(payload).hexdigest()
    return data


if __name__=="__main__":
    c=commitment(build_rows())
    print(json.dumps({k:v for k,v in c.items()
                      if k not in ("question_sha256","topology_sha256","query_char_offsets")},
                     sort_keys=True),flush=True)
