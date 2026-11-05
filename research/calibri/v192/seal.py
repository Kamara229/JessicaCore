"""Seal v192 answer grader with SHA256 manifest BEFORE any next QA response.

This workflow does not load old 128 answers, any original checkpoints
or any private TEST. Versioned SHA256 checksums can be required by a
future entirely NEW independent frozen public question experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
from .contract import VERSION,DOMAINS,STATES

def seal(output):
    root=Path(__file__).resolve().parent
    codefiles=("contract.py","recognize.py","grade.py")
    checksums={}
    for filename in codefiles:
        data=(root/filename).read_bytes()
        checksums[filename]={"sha256":hashlib.sha256(data).hexdigest(),
                             "bytes":len(data)}
    unique_frozen=hashlib.sha256(json.dumps(checksums,sort_keys=True,
                            separators=(",",":")).encode()).hexdigest()
    marker={"completed_and_frozen":True,"version":VERSION,
            "python_source_sha256":checksums,
            "aggregate_sha256":unique_frozen,
            "supported_domains":list(DOMAINS),
            "graded_states":list(STATES),
            "v190_precommitted_score_CANNOT_be_replaced":True,
            "v191_previous_posthoc_diagnostic_NOT_an_independent_accuracy_score":True,
            "this_version_locked_BEFORE_future_generation":True,
            "new_independent_public_QA_answers_generated_in_this_run":0,
            "existing_v190_v191_answer_artifacts_opened":False,
            "pretrained_model_source_loaded":False,
            "model_parameters_or_compressed_sources_modified":False,
            "no_private_v89_TEST_or_Jessica_chat_test":True,
            "future_scientific_requirements":[
                "New QAs and golds precommitted before either source model runs",
                "Verify immutable exact grader SHA256 equals this manifest",
                "Archive raw answers, not just parsed grades",
                "Report all 3 grades and 2-of-2 opposing-pair successes",
                "Original FP32 source tests do NOT establish <=0.7 bpp quality",
                "Only after original QA audit compare genuine compressed source"
            ]}
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"v192_frozen_grader_manifest.json").write_text(
        json.dumps(marker,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"stage":"frozen_contract_sealed","version":VERSION,
        "grader_sha256":unique_frozen,
        "new_model_answers":0,"baseline_unchanged":True}),flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    seal(args.output)
