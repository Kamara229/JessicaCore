"""S155: audit the frozen numeric S95 router, not a new model or QA test.

Explain S154's S82_new/82023/left regression by checking the ACTUAL prompt
route on the 456 immutable TRAIN cohort. S82 family labels are not routes.
No model weights are modified and no S82-specific inference exception added.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s87 import guard_all_456_full_cpu as s87
from calibri2.s94.physical_profile_bank import freeze_report
from calibri2.s95.train_and_audit_router import CHOICES, PromptOnlyRouter, prompt_only_vector

LOST_ID = "S82_new/82023/left"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("router",type=Path)
    parser.add_argument("router_report",type=Path)
    parser.add_argument("original_456_report",type=Path)
    parser.add_argument("s154_audit",type=Path)
    parser.add_argument("out",type=Path)
    a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    s95=json.loads(a.router_report.read_text())
    s154=json.loads(a.s154_audit.read_text())
    assert hashlib.sha256(a.router.read_bytes()).hexdigest()==s95["router_sha256"]
    assert s95["candidate_accepted"] is False
    assert s154["model_accepted"] is False and s154["strict_no_forgetting"] is False
    guard=s154["full456_and_new128"]
    assert guard["lost_old_full_ids"]==[LOST_ID],guard["lost_old_full_ids"]
    assert guard["old_full_before"]==362 and guard["old_full_after"]==361
    assert guard["new_full_before"]==17 and guard["new_full_after"]==17

    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    records=freeze_report(a.original_456_report)
    prompts={}
    for family,pairs in s87.families(tok).items():
        for pair in pairs:
            for side in ("left","right"):
                key=f'{family}/{pair["pair_id"]}/{side}'
                row=pair[side]
                assert hashlib.sha256(row["question"].encode("utf8")).hexdigest()==records[key]["question_sha256"]
                prompts[key]=row["question"]
    assert len(prompts)==len(records)==456 and set(prompts)==set(records)
    router=PromptOnlyRouter()
    router.load_state_dict(load_file(str(a.router)),strict=True)
    router.eval().requires_grad_(False)
    names=sorted(prompts)
    features=torch.as_tensor(np.stack([prompt_only_vector(tok,prompts[n]) for n in names]),dtype=torch.float32)
    with torch.inference_mode():
        logits=router(features)
        assert bool(torch.isfinite(logits).all())
        routed=logits.argmax(-1).tolist()
    by_id={k:CHOICES[int(i)] for k,i in zip(names,routed)}
    count=dict(Counter(by_id.values()))
    assert count==s95["frozen_report_routed_TRAIN_prediction"]["choices"],(count,s95["frozen_report_routed_TRAIN_prediction"]["choices"])
    s82_ids=[k for k in names if k.startswith("S82_new/")]
    s82_count=dict(Counter(by_id[k] for k in s82_ids))
    report={
        "stage":"S155_ACTUAL_S95_PROMPT_ROUTER_S82_S93_OVERLAP",
        "no_MODEL_changes_no_LIVE_no_QA":True,
        "source_S95_router_sha256":hashlib.sha256(a.router.read_bytes()).hexdigest(),
        "source_S154_lost_old_full_ids":guard["lost_old_full_ids"],
        "S154_rejected":True,
        "frozen_456_prompt_routes":count,
        "S82_old_family_total":len(s82_ids),
        "S82_family_frozen_routes":s82_count,
        "lost_S82_prompt_route":by_id[LOST_ID],
        "S93_route_overlap_proven":by_id[LOST_ID]=="S93",
        "original_S93_scoped_neural_hook_would_activate_on_lost_id":by_id[LOST_ID]=="S93",
        "next_test":"Train-specific retention for S93-routed protected S82 skills with no inferred-answer lookup; require full J7 before promotion",
        "model_accepted":False,
    }
    (a.out/"S155_ROUTER_OVERLAP_REPORT.json").write_text(
        json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("S155_FROZEN_S95_S82_S93_OVERLAP_AUDIT",json.dumps(report,sort_keys=True,ensure_ascii=False),flush=True)
    if by_id[LOST_ID]!="S93":
        raise AssertionError("S154 lost S82 identity is not S93; investigate prompt feature bridge")
if __name__=="__main__":
    main()
