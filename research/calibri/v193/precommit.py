"""Precommit NEW public original source QA + the exact v192 grader fingerprint.

This is run in CI as a separately completed step BEFORE ANY pretrained
model is loaded and before any tokens are generated. Abort if code changed,
public questions overlap earlier development sets, or gold/pairs drift.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .protocol import GRADER_SHA256,GRADER_VERSION,COUNT,OLD_MODEL,NEW_MODEL,OLD_REVISION,NEW_REVISION
from .questions import cases


def checksum_qa(items):
    return hashlib.sha256(json.dumps(items,sort_keys=True,
            separators=(",",":"),ensure_ascii=False).encode("utf8")).hexdigest()


def grading_fingerprint():
    root=Path(__file__).resolve().parents[1]/"v192"
    checks={}
    for name in ("contract.py","recognize.py","grade.py"):
        data=(root/name).read_bytes()
        checks[name]={"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
    signature=hashlib.sha256(json.dumps(checks,sort_keys=True,
                      separators=(",",":")).encode()).hexdigest()
    if signature!=GRADER_SHA256:
        raise AssertionError("Original v192 grading contract drift: do not run new models")
    return {"version":GRADER_VERSION,"sha256":signature,"files":checks}


def freeze(output):
    from v169.sources import all_cases
    from v190.cases import cases as earlier_public_64
    items=cases()
    if len(items)!=COUNT:raise AssertionError("Missing precommitted QA questions")
    already={x["question"] for x in all_cases()}|{x["question"] for x in earlier_public_64()}
    fresh={x["question"] for x in items}
    if fresh & already:raise AssertionError("A NEW QA duplicates an earlier historical public DEV question")
    grader=grading_fingerprint()
    manifest={"complete_precommit_before_original_model_inference":True,
        "title":"CALIBRI v193 public newly constructed source-only capability control",
        "status":"NEW PUBLIC DEVELOPMENT, NOT SEALED PRIVATE TEST",
        "full_questions_golds":items,"unique_questions":len(fresh),
        "reused_exact_historical_public_questions":0,
        "question_set_sha256":checksum_qa(items),
        "grading_source_frozen_before_inference":grader,
        "original_fp32_135m":{"model":OLD_MODEL,"revision":OLD_REVISION},
        "original_fp32_360m":{"model":NEW_MODEL,"revision":NEW_REVISION},
        "all_gold_labels_precommitted":True,
        "model_source_retraining":False,"compressed_source_test":False,
        "private_v89_TEST_unread":True,
        "Jessica_not_running_tests":True,
        "main_Render_Supabase_APK_untouched":True}
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",
                      encoding="utf8")
    print(json.dumps({"stage":"v193_precommitted","n":len(items),
        "new_public_question_set_sha256":manifest["question_set_sha256"],
        "frozen_grader_sha256":grader["sha256"],"original_FP32_generations":0}),
        flush=True)
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    freeze(args.output)
