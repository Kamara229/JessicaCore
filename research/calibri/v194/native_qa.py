"""Direct native CPU QA from a physically paid, SEALED 360M CB59 source.

No FP32 pretrained teacher loaded here: the only trained matrix source
supplied to this job is the SHA-checked 5-bit physical archive.
Baseline FP32 answers are existing frozen public v193 raw JSON, not
unpaid weights loaded beside the native compact model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

from direct_codebook_v59 import CB59
from direct_runtime_v59 import NativeKernel,bind_into_llama,compile_kernel
from v175.evaluate import generate
from v174.run import save
from v192.grade import grade,summarize
from v193.precommit import grading_fingerprint
from . import protocol as P


@torch.inference_mode()
def run(source_directory,v193_original_record,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source_directory=Path(source_directory)
    record=json.loads((source_directory/"v194_actual_360M_physical_source_seal.json").read_text())
    binary=(source_directory/"CALIBRI_v194_original_360M_complete_native.cb59").read_bytes()
    old=json.loads(Path(v193_original_record).read_text())
    if grading_fingerprint()["sha256"]!=P.FROZEN_GRADER_SHA:
        raise AssertionError("Frozen v192 answer grader CHANGED since independent audit")
    if not old.get("complete") or not old["frozen_before_any_model_generation"].get(
        "complete_precommit_before_original_model_inference"):
        raise AssertionError("Unsealed original FP32 360M QA comparison")
    if old["frozen_before_any_model_generation"]["grading_source_frozen_before_inference"]["sha256"]!=P.FROZEN_GRADER_SHA:
        raise AssertionError("Original FP32 source was not evaluated with same locked grader")
    if len(binary)!=record["physical_whole_file_bytes"] or (
        hashlib.sha256(binary).hexdigest()!=record["physical_whole_SHA256"]):
        raise AssertionError("Physical complete archive does not match stage 1 source SHA")
    if (len(binary)>P.MAX_0P700_PHYSICAL_BYTES or
        record["unique_trained_source_parameter_count"]!=P.ORIGINAL_UNIQUE_PARAMETERS or
        record["hf_original_source_revision_sha"]!=P.REVISION):
        raise AssertionError("Archive source provenance/physical 0.700 budget changed")
    archive=CB59(binary)
    if archive.meta["original_unique_parameter_count"]!=P.ORIGINAL_UNIQUE_PARAMETERS:
        raise AssertionError("Incomplete original 360M source, paid bits accounting not sound")
    if archive.handles["model.embed_tokens.weight"] is not archive.handles["lm_head.weight"]:
        raise AssertionError("Native codebook missing shared 360M embeddings/head")
    locked=old["frozen_before_any_model_generation"]["full_questions_golds"]
    raw_baseline=old["actual_unfiltered_FP32_generations_96_each"]["360M"]
    if len(locked)!=96 or len(raw_baseline)!=96 or any(
        x["question"]!=y["question"] or x["case_id"]!=y["case_id"]
        for x,y in zip(locked,raw_baseline)):
        raise AssertionError("Original FP32 control and QA frozen input mismatch")
    ids=P.SMOKE_IDS
    sampled=[locked[i] for i in ids]
    fp32_original={x["case_id"]:x["complete_generation_original"]["verbatim_text"]
                   for x in raw_baseline if x["case_id"] in ids}
    if set(fp32_original)!=set(ids):
        raise AssertionError("Missing original 360M source answer in public frozen control")
    baseline_graded=summarize(sampled,fp32_original)
    # Seal archive bytes, QA identifiers and original FP32 answers BEFORE
    # any compact native source is bound or any new inference is performed.
    frozen={"frozen_before_native_inference":True,
        "genuine_original_360M_bytes":len(binary),
        "genuine_original_360M_source_sha256":record["physical_whole_SHA256"],
        "genuine_original_360M_unique_parameters":P.ORIGINAL_UNIQUE_PARAMETERS,
        "physical_whole_model_bpp":8*len(binary)/P.ORIGINAL_UNIQUE_PARAMETERS,
        "original_fp32_public_QA_run_id":P.FROZEN_ORIGINAL_QA_RUN,
        "precommitted_original_360M_fp32_baseline_answers":fp32_original,
        "frozen_smoke_case_ids":list(ids),
        "frozen_smoke_questions_and_gold":sampled,
        "baseline_preexisting_original_FP32_scores":baseline_graded,
        "immutable_v192_grader_sha":P.FROZEN_GRADER_SHA,
        "native_compressed_model_inference_NOT_yet_done":True,
        "source_teacher_360M_full_FP32_not_loaded":True,
        "no_source_training_any_public_qa_answers":True,
        "private_v89_TEST_unread":True}
    save(output/"frozen_original_360M_physical_and_QA_protocol_BEFORE_NATIVE_INFERENCE.json",frozen)

    torch.set_num_threads(2)
    config=AutoConfig.from_pretrained(P.MODEL,revision=P.REVISION)
    config._attn_implementation="eager"
    with init_empty_weights(include_buffers=False):
        model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32).eval()
    kernel_file=Path(__file__).resolve().parents[1]/"direct_kernel_v59.c"
    native=NativeKernel(compile_kernel(kernel_file,output/"native_360m_cb59.so"))
    runtime_info=bind_into_llama(model,archive,native)
    if (runtime_info["runtime_dense_multi_dim_source_parameters"]!=0 or
        not runtime_info["tied_embed_and_output_codebook_SHARED_IN_RAM"]):
        raise AssertionError("Unpaid dense model 2D weights leaked into compact native execution")
    tok=AutoTokenizer.from_pretrained(P.MODEL,revision=P.REVISION)
    output_raw={}
    output_time=[]
    for case in sampled:
        result=generate(model,tok,case["question"],cap=P.CAP,stop=True)
        fp32_template_hash=raw_baseline[case["case_id"]]["complete_generation_original"]["prompt_sha"]
        if result["prompt_sha"]!=fp32_template_hash:
            raise AssertionError("Compact 360M native input differs from original 360M FP32 prompt")
        output_raw[case["case_id"]]=result["verbatim_text"]
        output_time.append({"case_id":case["case_id"],
            "seconds_including_prefill":result["wall_seconds"],
            "new_token_count":len(result["token_ids"]),
            "EOS":result["eos"],
            "native_360M_source_SHA256":record["physical_whole_SHA256"]})
        save(output/"in_progress_REAL_PACKED_360M_actual_answers.json",
             {"completed":len(output_raw),"records":output_raw,
              "not_a_final_quality_PASS":True})
        print(json.dumps({"stage":"v194_NATIVE_pack_only",
            "completed":len(output_raw),"case_id":case["case_id"],
            "seconds":result["wall_seconds"],"EOS":result["eos"]}),flush=True)
    compact_graded=summarize(sampled,output_raw)
    same=sum(output_raw[case_id]==fp32_original[case_id] for case_id in ids)
    report={"complete":True,"source_physical":record,
        "native_runtime":runtime_info,
        "entire_360M_model_from_packed_weights_only":True,
        "source_teacher_FP32_weights_loaded_for_native":False,
        "raw_full_original_FP32_QA_baseline":fp32_original,
        "original_360M_FP32_preexisting_gradings":baseline_graded,
        "actual_native_packed_360M_VERBATIM_answers":output_raw,
        "actual_native_packed_360M_frozen_gradings":compact_graded,
        "native_timings_diagnostic":output_time,
        "text_exact_agreement_count_on_16":same,
        "actual_native_binary_SHA256":record["physical_whole_SHA256"],
        "source_bpp":8*len(binary)/P.ORIGINAL_UNIQUE_PARAMETERS,
        "source_bytes":len(binary),
        "actual_0p700_PHYSICAL_SOURCE_gate_PASS":len(binary)<=P.MAX_0P700_PHYSICAL_BYTES,
        "original_fp32_behavioral_parity_NOT_INFERRED":True,
        "native_360M_CPU_speed_relative_to_same_baseline_NOT_YET_MEASURED":True,
        "public_development_smoke_not_final_private_TEST":True,
        "all_failed_cases_preserved":True,
        "private_v89_TEST_not_used":True,"Jessica_ordinary_chat_tests":0}
    save(output/"v194_360m_PHYSICAL_NATIVE_16QA_completed.json",report)
    print(json.dumps({"stage":"v194_NATIVE_complete",
        "physical_bytes":len(binary),
        "actual_bpp":report["source_bpp"],
        "real_original_FP32_machine_correct":baseline_graded["machine_confirmed"],
        "real_packed_native_machine_correct":compact_graded["machine_confirmed"],
        "ungraded_native":compact_graded["ungraded_not_known_wrong"],
        "verbatim_match_count":same,"native_zero_dense_source":True,
        "paired_compact":{k:{"both_correct":v["both_correct"],"total":v["total"]}
                          for k,v in compact_graded["opposite_pairs"].items()}}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("source_directory",type=Path)
    ap.add_argument("original_v193_json",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    run(args.source_directory,args.original_v193_json,args.output)
