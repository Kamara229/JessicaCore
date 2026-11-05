"""Actually write a complete original pretrained SmolLM2 360M paid CB59 file.

The original Hugging Face FP32 source exists here ONLY to form a binary
archive. No evaluation or compressed inference is allowed to borrow
teacher tensors. SHA/CRC, aliases and all UNIQUE source parameters are
verified before evaluating quality or speed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from direct_codebook_v59 import CB59,encode
from v174.run import save
from . import protocol as P


@torch.inference_mode()
def run(output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    precommit={"before_model_loading":True,
        "model":P.MODEL,"revision":P.REVISION,
        "count_original_unique_parameters":P.ORIGINAL_UNIQUE_PARAMETERS,
        "maximum_whole_source_0p700_bytes":P.MAX_0P700_PHYSICAL_BYTES,
        "actual_bits_per_vector":P.BITS_PER_VECTOR,
        "source_weights_per_vector":P.WEIGHTS_PER_VECTOR,
        "codebook_entries":P.CODEBOOK_ENTRIES,
        "source_codec":P.SOURCE_CODEC,
        "no_answer_training":True,
        "no_Jessica_chat_model_tests":True}
    save(output/"v194_360M_protocol_FROZEN_BEFORE_MODEL_LOADING.json",precommit)
    torch.set_num_threads(2)
    torch.manual_seed(194)
    model=AutoModelForCausalLM.from_pretrained(
         P.MODEL,revision=P.REVISION,torch_dtype=torch.float32,
         attn_implementation="eager").cpu().eval()
    original_unique=sum(x.numel() for x in model.parameters())
    if original_unique!=P.ORIGINAL_UNIQUE_PARAMETERS:
        raise AssertionError("Wrong original model unique parameters")
    if model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr():
        raise AssertionError("Original tied pretrained embeddings/head source is not shared")
    original_state=model.state_dict()
    if not original_state:
        raise AssertionError("Empty source state")
    archive,meta,diag=encode(original_state)
    data=CB59(archive,expected_keys=original_state.keys())
    physical=len(archive)
    sha256=hashlib.sha256(archive).hexdigest()
    bpp=8*physical/original_unique
    if (physical>P.MAX_0P700_PHYSICAL_BYTES or bpp>0.70000001 or
        data.meta["original_unique_parameter_count"]!=original_unique):
        raise AssertionError("FULL physical original 360M bpp exceeds source budget")
    if set(data.meta["full_original_state_keys"])!=set(original_state):
        raise AssertionError("Missing original source trained tensor/alias")
    if data.meta["source_alias_keys"].get("lm_head.weight")!="model.embed_tokens.weight":
        raise AssertionError("Original 360M input/output head not shared in physical file")
    descriptors=data.meta["tensor_descriptors"]
    if not descriptors or not any(row["dtype"]=="CB59_K32V8" for row in descriptors.values()):
        raise AssertionError("No actual 5bit compressed original trained matrices")
    codec_matrices=sum(v["dtype"]=="CB59_K32V8" for v in descriptors.values())
    if codec_matrices<100:
        raise AssertionError("Entire original transformer was not physically packed")
    physical_path=output/"CALIBRI_v194_original_360M_complete_native.cb59"
    physical_path.write_bytes(archive)
    if physical_path.stat().st_size!=physical or hashlib.sha256(physical_path.read_bytes()).hexdigest()!=sha256:
        raise AssertionError("Written original 360M physical archive corrupted")
    manifest={
        "complete_original_360m_real_weight_archive":True,
        "full_original_pretrained_model":P.MODEL,
        "hf_original_source_revision_sha":P.REVISION,
        "unique_trained_source_parameter_count":original_unique,
        "physical_whole_file_bytes":physical,
        "MAX_paid_bytes_0p700":P.MAX_0P700_PHYSICAL_BYTES,
        "ACTUAL_entire_source_bits_per_unique_original_parameter":bpp,
        "physical_whole_SHA256":sha256,
        "source_format":"CB59 direct-resident K32/V8 5bit indices",
        "source_group_bits":5,"original_weights_per_group":8,
        "encoded_all_original_trained_matrix_count":codec_matrices,
        "original_source_noncompressed_vectors":sum(v["dtype"]=="f4" for v in descriptors.values()),
        "physical_embedding_head_is_one_tied_alias":True,
        "source_CRC_and_SHA_pass":True,
        "no_FP32_trained_source_sidecar_in_paid_file":True,
        "no_CALIBRI_360M_model_qualitative_claim":True,
        "no_native_CPU_answer_pass_yet":True,
        "no_quality_data_or_gold_used_to_fit_weights":True,
        "private_v89_TEST_unread":True,
        "research_branch_only_Jessica_not_testing":True,
        "diagnostics_are_source_weight_error_not_answer_quality":diag}
    save(output/"v194_actual_360M_physical_source_seal.json",manifest)
    print(json.dumps({"stage":"v194_physical_complete_360M_SOURCE",
        "actual_bpp":bpp,"actual_bytes":physical,"sha256":sha256,
        "original_unique_parameters":original_unique,
        "codec_matrices":codec_matrices,
        "has_native_QA_RESULT_YET":False}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    run(args.output)
