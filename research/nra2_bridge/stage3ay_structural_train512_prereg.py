"""S3AY: preregister independent structural grammar TRAIN512 on S3AQ TRAIN topologies.

Data-only; no original CPU inference or model scoring. The previously
scored research DEV128 may motivate adding broad syntax diversity but is
NEVER placed into the optimizer or copied as a training prompt. The S3AQ
future independent EVAL128 remains unopened and never CPU-prefilled.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows as s3aq_rows, commitment as s3aq_commitment,
    TRAIN as S3AQ_TRAIN, DEV as S3AQ_DEV, EVAL as S3AQ_EVAL, STYLES as S3AQ_STYLES,
)
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets
from research.nra2_bridge.stage3au_train512_paraphrase_prereg import STYLES as S3AU_PREVIOUS_STYLES
from research.nra2_bridge.stage3g_factorized_graph_controls import oracle
from research.nra2_bridge.stage3m_role_balanced_graph_data import GraphPrompt

SPLIT="S3AY_NEW_TRAIN512"
S3AQ_SHA="c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839"
S3AR_REAL_CPU_SUCCESS=43580373587
S3AS_ROLE_SUCCESS=43581666491
S3AT_DEV_DIAG_SUCCESS=43582046510
S3AX_DEV_DIAG_SUCCESS=43587189605
S3AW_SUCCESS=43583928298
S3AW_PHYSICAL_SHA="8541eea24d5074e29b6a4acd8b611d2d29d608dacf2dfe909e8df05eb0fb2130"
S3AU_PRECOMMIT_SHA="ac038f50dab0ace449f1df32ec7e8728476420615f60c85a8d8fce64c787558a"
S3AS_PHYSICAL_SHA="06420e22270dd082116086a5174e95c04e9ff2b52664583bc8d5a1b4a074d488"
STYLES=(
    "Commencing travel at {src}, might one eventually arrive at {dst} by obeying each arrow?",
    "Could arrival at {dst} follow a journey that was launched from {src}?",
    "Upon setting off at {src}, is it possible to finish at {dst} along directed links?",
    "If {dst} marks the terminus, can a trip have originated at {src}?",
    "Leaving from {src} at the outset, would the directed edges allow reaching {dst}?",
    "Could a path that ends at {dst} have its point of departure at {src}?",
    "With {src} as departure point, is {dst} an attainable end along forward arrows?",
    "Given {dst} as the final destination, can a route begin at {src}?",
)
DESTINATION_FIRST=(1,3,5,7)
SOURCE_FIRST=(0,2,4,6)


def build_rows():
    prior=s3aq_rows()
    if s3aq_commitment(prior)["commitment_sha256"]!=S3AQ_SHA:
        raise AssertionError("S3AY parent S3AQ TRAIN/DEV/FUTURE EVAL drift")
    all_legacy_sha={r.prompt_sha256 for r in prior}
    if len(STYLES)!=8 or len(set(STYLES))!=8:
        raise AssertionError("S3AY eight novel independent wordings")
    if set(STYLES)&(set(sum((list(v) for v in S3AQ_STYLES.values()),[]))|set(S3AU_PREVIOUS_STYLES)):
        raise AssertionError("S3AY cannot reuse S3AQ TRAIN/DEV/EVAL or prior S3AU TRAIN wordings")
    training=[r for r in prior if r.split==S3AQ_TRAIN]
    by_graph=defaultdict(list)
    for r in training:
        by_graph[r.graph_id].append(r)
    if len(by_graph)!=64 or any(len(v)!=4 for v in by_graph.values()):
        raise AssertionError("S3AY uses only S3AQ TRAIN64 graph topologies")
    rows=[]
    for graph,items in by_graph.items():
        items.sort(key=lambda r:r.query_style)
        assert [r.query_style for r in items]==[0,1,2,3]
        r0,r1=items[0],items[1]
        if (r0.src,r0.dst)!=(r1.dst,r1.src):
            raise AssertionError("S3AY S3AQ TRAIN roles are not swapped as prereg")
        for i,template in enumerate(STYLES):
            old=items[i%4]
            oldspan=query_char_offsets(old.question)["clause"]
            src,dst=(r0.src,r0.dst) if i%2==0 else (r1.src,r1.dst)
            query=template.format(src=f"R{src}",dst=f"R{dst}")
            question=old.question[:oldspan[0]]+query+old.question[oldspan[1]:]
            x=GraphPrompt(
                graph_id=f"{SPLIT}-{graph}",
                split=SPLIT,family=old.family,active_nodes=old.active_nodes,
                edges=old.edges,src=src,dst=dst,
                reachable=oracle(old.edges,src,dst),query_style=i,question=question)
            if x.prompt_sha256 in all_legacy_sha:
                raise AssertionError("S3AY new TRAIN prompt reused any S3AQ split")
            rows.append(x)
    validate(rows)
    return rows


def validate(rows):
    if len(rows)!=512 or len({r.prompt_sha256 for r in rows})!=512:
        raise AssertionError("S3AY exactly 512 distinct training paraphrases required")
    groups=defaultdict(list)
    for r in rows:
        if r.split!=SPLIT or r.query_style not in range(8):
            raise AssertionError("S3AY strictly TRAIN only with eight styles")
        order=query_char_offsets(r.question)["text_order_nodes"]
        expected=[r.dst,r.src] if r.query_style in DESTINATION_FIRST else [r.src,r.dst]
        if order!=expected or r.reachable!=oracle(r.edges,r.src,r.dst):
            raise AssertionError("S3AY source/destination literal text and BFS oracle drift")
        groups[r.graph_id].append(r)
    if len(groups)!=64 or any(len(v)!=8 for v in groups.values()):
        raise AssertionError("S3AY exactly 64 source TRAIN graphs x 8 novel wordings")
    for r in groups.values():
        r.sort(key=lambda x:x.query_style)
        if [x.query_style for x in r]!=list(range(8)):
            raise AssertionError("S3AY balanced style order")
        if sum(x.reachable for x in r)!=4:
            raise AssertionError("S3AY each TRAIN graph 4 yes / 4 no required")
        if any(r[i].src!=r[i+2].src or r[i].dst!=r[i+2].dst
               for i in (0,1,2,3,4,5) if i+2<8):
            raise AssertionError("S3AY quartet role consistency drift")
    for style in range(8):
        batch=[r for r in rows if r.query_style==style]
        if len(batch)!=64 or sum(r.reachable for r in batch)!=32:
            raise AssertionError("S3AY each grammar balanced yes/no")
    return True


def commitment(rows):
    validate(rows)
    p=s3aq_rows()
    training=[r for r in p if r.split==S3AQ_TRAIN]
    dev=[r for r in p if r.split==S3AQ_DEV]
    ev=[r for r in p if r.split==S3AQ_EVAL]
    by_style={str(i):{
        "count":sum(r.query_style==i for r in rows),
        "yes_count":sum(r.query_style==i and r.reachable for r in rows),
        "destination_first":i in DESTINATION_FIRST
    } for i in range(8)}
    data={
        "stage":"NRA2_S3AY_NEW_TRAIN512_PARAPHRASES_PRECOMMIT_NO_MODEL_NO_EVAL",
        "source_successful_S3AR_real_cpu_run":S3AR_REAL_CPU_SUCCESS,
        "S3AS_successful_role_train_run":S3AS_ROLE_SUCCESS,
        "S3AW_successful_role_train_run":S3AW_SUCCESS,
        "S3AX_successful_DEV_diagnostic_run":S3AX_DEV_DIAG_SUCCESS,
        "S3AW_physical_query_role_sha256":S3AW_PHYSICAL_SHA,
        "S3AU_prior_TRAIN512_commitment_sha256":S3AU_PRECOMMIT_SHA,
        "S3AT_exploratory_DEV_diagnostic_run":S3AT_DEV_DIAG_SUCCESS,
        "S3AQ_parent_preregistered_SHA256":S3AQ_SHA,
        "S3AS_parent_frozen_FP16_SHA256":S3AS_PHYSICAL_SHA,
        "TRAIN512_count":512,"source_TRAIN256_count":len(training),
        "reused_TRAIN_graph_topology_count":64,
        "source_S3AQ_DEV128_prompt_SHA256_only":hashlib.sha256(json.dumps(
            [r.prompt_sha256 for r in dev],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "source_S3AQ_FUTURE_EVAL128_prompt_SHA256_only":hashlib.sha256(json.dumps(
            [r.prompt_sha256 for r in ev],sort_keys=True,separators=(",",":")).encode()).hexdigest(),
        "old_scored_DEV128_NEVER_training_or_candidate_sealed_test":True,
        "original_S194_616_J7_CPU_required_before_all_future_training":True,
        "future_EVAL128_never_prefilled_scored_or_trained":True,
        "future_EVAL128_excluded_from_optimizer_and_selection":True,
        "new_8_train_styles":list(STYLES),
        "source_first_styles":list(SOURCE_FIRST),
        "destination_first_styles":list(DESTINATION_FIRST),
        "per_style_balanced":by_style,
        "graph_prompt_sha256_in_order":[r.prompt_sha256 for r in rows],
        "graph_topology_sha256_in_order":[hashlib.sha256(json.dumps(sorted(r.edges)).encode()).hexdigest() for r in rows],
        "gold_truth_in_order":[bool(r.reachable) for r in rows],
        "train_only_data_not_original_CPU_generated":True,
        "no_new_model_weights_or_training_or_scores":True,
        "output_gate":0.0,"candidate_accepted":False,
        "Jessica_LIVE_Render_Supabase_APK_main_QA_TEST_unchanged":True,
    }
    data["commitment_sha256"]=hashlib.sha256(json.dumps(
        data,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return data


def main():
    if len(sys.argv)!=2:
        raise SystemExit("S3AY EMPTY_OUT_DIR")
    out=Path(sys.argv[1])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("S3AY immutable prereg cannot overwrite")
    report=commitment(build_rows())
    out.mkdir(parents=True,exist_ok=False)
    (out/"NRA2_S3AY_NEW_TRAIN512_PARAPHRASE_PREREG.json").write_text(
        json.dumps(report,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items()
        if k not in ("graph_prompt_sha256_in_order","graph_topology_sha256_in_order",
                     "gold_truth_in_order")},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
