"""Actual pinned original FP32 135M and 360M responses to NEW 96 public QA.

Full original generated text + token IDs; exact v192 grading checksum
was locked BEFORE either model's weights could be loaded here.
No training, CALIBRI quantization, private TEST or Jessica-chat tests.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from v84.sources import original_teacher
from v174.run import save
from v175.evaluate import generate
from v192.grade import summarize
from .precommit import checksum_qa,grading_fingerprint
from .protocol import (
    COUNT,CAP,OLD_MODEL,OLD_REVISION,OLD_UNIQUE,
    NEW_MODEL,NEW_REVISION,NEW_UNIQUE,GRADER_SHA256)
from .questions import cases


@torch.inference_mode()
def run(precommitted_path,output):
    """Same fixed original source comparison, original models only."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(193)
    frozen=json.loads(Path(precommitted_path).read_text(encoding="utf8"))
    expected=cases()
    if not frozen.get("complete_precommit_before_original_model_inference"):
        raise AssertionError("No finalized preregistration; cannot load original models")
    if (frozen["full_questions_golds"]!=expected or
        len(expected)!=COUNT or
        frozen["question_set_sha256"]!=checksum_qa(expected)):
        raise AssertionError("Original QA input/golds mismatch since precommit")
    if grading_fingerprint()["sha256"]!=GRADER_SHA256 or (
        frozen["grading_source_frozen_before_inference"]["sha256"]!=GRADER_SHA256):
        raise AssertionError("Grader source cannot change after QA freeze")
    if frozen["original_fp32_135m"]!={"model":OLD_MODEL,"revision":OLD_REVISION}:
        raise AssertionError("135M source revision drift")
    if frozen["original_fp32_360m"]!={"model":NEW_MODEL,"revision":NEW_REVISION}:
        raise AssertionError("360M source revision drift")
    save(output/"verified_precommitted_96QA_and_grader.json",frozen)

    original135=original_teacher().cpu().eval()
    tokenizer135=AutoTokenizer.from_pretrained(OLD_MODEL,revision=OLD_REVISION)
    original360=AutoModelForCausalLM.from_pretrained(
        NEW_MODEL,revision=NEW_REVISION,torch_dtype=torch.float32).cpu().eval()
    tokenizer360=AutoTokenizer.from_pretrained(NEW_MODEL,revision=NEW_REVISION)
    available={"135M":(original135,tokenizer135,OLD_UNIQUE),
               "360M":(original360,tokenizer360,NEW_UNIQUE)}
    provenance={}
    for arm,(source,tok,expected_unique) in available.items():
        count=sum(p.numel() for p in source.parameters())
        tied=source.get_input_embeddings().weight.data_ptr()==source.get_output_embeddings().weight.data_ptr()
        if count!=expected_unique or not tied:
            raise AssertionError("Original source FP32 param count/tied embeddings diverged: "+arm)
        provenance[arm]={"unique_original_parameters":count,
            "tied_input_output_embeddings":True,
            "tokenizer_vocab_size":len(tok),
            "source":OLD_MODEL if arm=="135M" else NEW_MODEL,
            "original_revision":OLD_REVISION if arm=="135M" else NEW_REVISION}
    save(output/"original_source_provenance.json",provenance)

    raw={arm:[] for arm in available}
    extracted={arm:{} for arm in available}
    for n,case in enumerate(expected):
        for arm,(model,tok,_) in available.items():
            generated=generate(model,tok,case["question"],cap=CAP,stop=True)
            chat=tok.apply_chat_template(
                [{"role":"user","content":case["question"]}],
                tokenize=False,add_generation_prompt=True)
            if generated["prompt_sha"]!=hashlib.sha256(chat.encode()).hexdigest():
                raise AssertionError("Original model saw a different question/chat template")
            response=generated["verbatim_text"]
            if not isinstance(response,str):
                raise AssertionError("No original model answer text for QA case")
            raw[arm].append({"case_id":case["case_id"],"domain":case["domain"],
                "question":case["question"],"gold":case["gold"],
                "pair_id":case.get("pair_id"),
                "question_sha256":case["question_sha256"],
                "complete_generation_original":generated})
            extracted[arm][case["case_id"]]=response
        if (n+1)%12==0:
            save(output/"original_135M_360M_actual_answer_PARTIAL.json",raw)
            print(json.dumps({"stage":"v193_original_new_QA",
                "completed_per_original_source":n+1,
                "all_outputs_preserved":True,
                "no_quantization_test":True}),flush=True)
    if any(len(rows)!=COUNT for rows in raw.values()):
        raise AssertionError("Original response lost before summary")
    scored={arm:summarize(expected,answers)
            for arm,answers in extracted.items()}
    for arm,summary in scored.items():
        if (summary["n"]!=COUNT or
            summary["machine_confirmed"]+
                summary["machine_recognized_wrong"]+
                summary["ungraded_not_known_wrong"]!=COUNT):
            raise AssertionError("Incomplete frozen model-response verdict for "+arm)
        for domain in ("opposite_yesno","fictional_binding"):
            if summary["opposite_pairs"][domain]["total"]!=12:
                raise AssertionError("Paired opposite questions missing")
    complete={"complete":True,
        "frozen_before_any_model_generation":frozen,
        "original_source_provenance":provenance,
        "actual_unfiltered_FP32_generations_96_each":raw,
        "frozen_v192_machine_grading":scored,
        "this_is_public_original_DEVELOPMENT_not_private_independent_TEST":True,
        "no_compressed_source_bpp_or_native_CPU_measured":True,
        "new_original_model_parameters_trained":0,
        "private_v89_TEST_unread":True,
        "Jessica_text_tests_executed":0}
    save(output/"v193_original_FP32_135M_vs_360M_new_96QA_complete.json",complete)
    print(json.dumps({"stage":"v193_complete","original_unique_parameter_counts":{
        arm:source["unique_original_parameters"] for arm,source in provenance.items()},
        "grader_sha256":GRADER_SHA256,
        "graded_outcomes":{arm:{
            "confirmed":z["machine_confirmed"],
            "recognized_wrong":z["machine_recognized_wrong"],
            "ungraded":z["ungraded_not_known_wrong"],
            "domains":z["domain_totals"],
            "paired":{d:{"total":p["total"],"both_correct":p["both_correct"]}
                      for d,p in z["opposite_pairs"].items()}}
            for arm,z in scored.items()}}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("precommitted",type=Path)
    parser.add_argument("output",type=Path)
    options=parser.parse_args()
    run(options.precommitted,options.output)
