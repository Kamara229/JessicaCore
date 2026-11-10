"""S87-J4 first and only preregistered 64-case unseen QA: READ-ONLY.
No training, model selection, promotion or re-use of previously inspected QA.
This one-shot comparison is specific to this S87-J4 checkpoint and never
authorizes using these answers to improve a subsequent checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA
from calibri2.s30 import matched_dense_fp32_train as s30
from calibri2.s65 import train_only_new_rank116_and_guard_old as s65
from calibri2.s82 import real_mps_generic_token_context_copy as s82
from calibri2.s87 import guard_all_456_full_cpu as guard
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS

COLORS = ("red", "green", "blue", "yellow")
FRAMES = (
    "In the curator's ledger, {a} recorded {left}, whereas {b} recorded {right}. Return the exact two-word ledger label recorded by {a}.",
    "A safety checklist assigns {b} the code {right} and {a} the code {left}. Answer with the two words in {a}'s code only.",
    "At the equipment desk, the tag for {a} reads {left}; the separate tag for {b} reads {right}. Copy precisely the two-word tag for {a}.",
    "The inventory officer associated {b} with {right}, while the entry for {a} says {left}. Which two words belong to {a}?",
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def preregistered_64(tok):
    _, trained_nouns = s82.newly_authored_train(tok)
    eligible = [
        noun for noun in s82.NOUNS
        if noun not in trained_nouns
        and all(len(tok.encode(f"{color} {noun}", add_special_tokens=False)) == 2
                for color in COLORS)
    ]
    if len(eligible) < 8:
        raise AssertionError("Fewer than eight tokenizer-compatible nouns withheld from S82 TRAIN")
    # The first eight held-out nouns are fixed *before* any candidate inference.
    # Four novel prompt frames x eight noun contrasts x two swaps = 64.
    chosen = eligible[-8:]
    rows = []
    for frame_i, frame in enumerate(FRAMES):
        for i in range(8):
            color = COLORS[(frame_i + i) % 4]
            other_color = COLORS[(frame_i + i + 2) % 4]
            desired = f"{color} {chosen[i]}"
            distractor = f"{other_color} {chosen[(i + 3) % 8]}"
            a, b = f"Kervon{frame_i}{i}", f"Meldis{frame_i}{i}"
            for side in ("left", "right"):
                left, right = ((desired, distractor) if side == "left"
                               else (distractor, desired))
                question = frame.format(a=a, b=b, left=left, right=right)
                gold = tok.encode(left, add_special_tokens=False) + [tok.eos_token_id]
                if len(gold) != 3:
                    raise AssertionError("QA answer must have exactly 2 tokens plus EOS")
                rows.append((f"fresh64/{frame_i}/{i}/{side}", question, left, gold))
    if len(rows) != 64 or len({q for _,q,_,_ in rows}) != 64:
        raise AssertionError("QA 64 uniqueness failed")
    return rows, chosen


def main():
    p=argparse.ArgumentParser()
    p.add_argument("old_cb59", type=Path)
    p.add_argument("old_s82_sidecar", type=Path)
    p.add_argument("new_cb61", type=Path)
    p.add_argument("new_joint_sidecar", type=Path)
    p.add_argument("new_manifest", type=Path)
    p.add_argument("output", type=Path)
    a=p.parse_args()
    torch.set_num_threads(2)
    a.output.mkdir(parents=True, exist_ok=True)
    old_raw=a.old_cb59.read_bytes()
    side_old_bytes=a.old_s82_sidecar.read_bytes()
    side_new_bytes=a.new_joint_sidecar.read_bytes()
    cb61_bytes=a.new_cb61.read_bytes()
    m=json.loads(a.new_manifest.read_text())
    raw_new=guard.open_envelope(cb61_bytes)
    if (len(old_raw)!=SOURCE_BYTES or sha(old_raw)!=s65.S45_SHA
            or sha(side_old_bytes)!=m["old_S82_sidecar_sha256"]
            or sha(side_new_bytes)!=m["prototype_sidecar_sha256"]
            or sha(cb61_bytes)!=m["source_CB61_physical_sha256"]
            or sha(raw_new)!=m["source_sha256"]):
        raise AssertionError("Immutable S82/S87-J4 source chain failed")
    total=len(cb61_bytes)+len(side_new_bytes)+a.new_manifest.stat().st_size
    bpp=8*total/SOURCE_PARAMS
    if total!=m["all_inclusive_physical_bytes"] or bpp>0.7:
        raise AssertionError("Actual physical bpp failed")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    s30.TOK=tok
    qa, withheld_nouns=preregistered_64(tok)
    historic=guard.families(tok)
    seen={p[s]["question"] for family in historic.values() for p in family
          for s in ("left","right")}
    if len(seen)!=456 or any(q in seen for _,q,_,_ in qa):
        raise AssertionError("QA64 overlaps authorized TRAIN or TRAIN provenance changed")
    old_weights=load_file(str(a.old_s82_sidecar))
    new_weights=load_file(str(a.new_joint_sidecar))
    if any(k not in new_weights or not torch.equal(v,new_weights[k])
           for k,v in old_weights.items()):
        raise AssertionError("S82 frozen controller changed")
    old=guard.build_cpu(old_raw, old_weights, False)
    new=guard.build_cpu(raw_new, new_weights, True)
    results=[]
    # The entire preregistered QA is evaluated exactly ONCE, with frozen sources.
    for key,question,answer,gold in qa:
        # Explicit argument order: model, sources, first, later, head, tokenizer, question, pointer.
        old_ids=s82.full_cpu_generate(old[0],old[1],old[2],old[3],old[5],
                                      tok,question,old[4])
        new_ids=s82.full_cpu_generate(new[0],new[1],new[2],new[3],new[5],
                                      tok,question,new[4])
        was=old_ids==gold
        now=new_ids==gold
        results.append({"id":key, "question_sha256":sha(question.encode()),
                        "target_ids":gold,"s82_ids":old_ids,"s87j4_ids":new_ids,
                        "s82_full":was,"s87j4_full":now,
                        "s82_first":bool(old_ids) and old_ids[0]==gold[0],
                        "s87j4_first":bool(new_ids) and new_ids[0]==gold[0]})
    if len(results)!=64:
        raise AssertionError("QA64 incomplete")
    improved=[r["id"] for r in results if r["s87j4_full"] and not r["s82_full"]]
    regressed=[r["id"] for r in results if r["s82_full"] and not r["s87j4_full"]]
    summary={
        "stage":"S87_J4_FRESH64_ONE_SHOT_UNSEEN_CPU_COMPARISON",
        "immutably_frozen_candidate":True,
        "candidate_accepted":False,
        "no_training_or_selection_from_this_QA":True,
        "QA_permanently_embargoed_after_inspection":True,
        "QA_count":len(results),
        "QA_question_sha256s":[r["question_sha256"] for r in results],
        "withheld_S82_train_nouns":withheld_nouns,
        "source_cb61_sha256":sha(cb61_bytes),
        "joint_sidecar_sha256":sha(side_new_bytes),
        "all_inclusive_bpp":bpp,
        "S82_full":sum(r["s82_full"] for r in results),
        "S87J4_full":sum(r["s87j4_full"] for r in results),
        "S82_first":sum(r["s82_first"] for r in results),
        "S87J4_first":sum(r["s87j4_first"] for r in results),
        "gained_full":improved,
        "lost_full":regressed,
        "results":results,
        "GPU_CPU_parity_proven":False,
        "native_C_speed_proven":False,
        "FP32_quality_parity_proven":False,
        "no_LIVE_changes":True,
    }
    (a.output/"S87J4_FRESH64_ONE_SHOT_QA.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps({"event":"S87J4_FRESH64_QA_COMPLETE",
                      "S82_full":summary["S82_full"],
                      "S87J4_full":summary["S87J4_full"],
                      "S82_first":summary["S82_first"],
                      "S87J4_first":summary["S87J4_first"],
                      "gained_full":len(improved),
                      "lost_full":len(regressed),
                      "bpp":bpp,
                      "candidate_accepted":False}),flush=True)


if __name__=="__main__":
    main()
