"""Causal original FP32 source family substitutions on real 360M packed weights.

EVERY intervention carries an unpaid original FP32 2D model weight
family and must NEVER be claimed as <=0.7 bpp. The unchanged v194
29,348,539-byte source remains the only genuinely compact model here.
This is diagnostic, uses historical public development questions only.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer

from direct_codebook_v59 import CB59
from direct_runtime_v59 import NativeKernel,bind_into_llama,compile_kernel
from v174.run import save
from v175.evaluate import generate
from v192.grade import summarize
from v193.precommit import grading_fingerprint
from . import protocol as P


def substituted_modules(model,original,arm):
    """Assemble family facades while explicitly charging unpaid dense weights."""
    emb="embed_head" in arm
    attn="attention" in arm or "transformer" in arm
    mlp="mlp" in arm or "transformer" in arm
    if not any((emb,attn,mlp)):raise AssertionError("No valid selected diagnostic source")
    replaced=[]
    if emb:
        model.model.embed_tokens=original.model.embed_tokens
        model.lm_head=original.lm_head
        if model.get_input_embeddings().weight.data_ptr()!=model.get_output_embeddings().weight.data_ptr():
            raise AssertionError("Tied 360M original source embed/head not actually shared")
        replaced.extend(("model.embed_tokens","lm_head"))
    for name,module in original.named_modules():
        if not isinstance(module,nn.Linear):
            continue
        if (attn and ".self_attn." in name) or (mlp and ".mlp." in name):
            parent,_,attr=name.rpartition(".")
            holder=model.get_submodule(parent)
            incumbent=getattr(holder,attr)
            if tuple(module.weight.shape)!=tuple(incumbent.handle.shape):
                raise AssertionError("Original FP32 & real packed source layer shape differs: "+name)
            setattr(holder,attr,module)
            replaced.append(name)
    if (attn or mlp) and len(replaced)<=2:
        raise AssertionError("No original 360M transformer modules swapped")
    dense={name:tuple(p.shape) for name,p in model.named_parameters() if p.ndim>=2}
    if not dense:raise AssertionError("Intervention contains no actual FP32 pretrained source")
    unpaid_2d=sum(p.numel() for p in model.parameters() if p.ndim>=2)
    return {"arm":arm,"replaced_original_FP32_layer_names":replaced,
        "replaced_modules":len(replaced),
        "UNPAID_original_FP32_trained_2D_parameter_count":unpaid_2d,
        "UNPAID_trained_FP32_2D_parameter_BYTES":4*unpaid_2d,
        "original_FP32_source_used_in_runtime":True,
        "physically_packed_only_claim_IS_INVALID":True,
        "actual_model_bpp_without_charging_original_FP32_is_NOT_REPORTED":True,
        "no_new_archive_written":True,
        "all_other_2D_modules_stay_native_packed":True,
        "original_FP32_unpaid_module_names":list(dense)}


def run(binary_file,old_v193_json,output):
    binary_file=Path(binary_file)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(195)
    blob=binary_file.read_bytes()
    if len(blob)!=P.SOURCE_BYTES or hashlib.sha256(blob).hexdigest()!=P.SOURCE_SHA256:
        raise AssertionError("Wrong v194 actual physically paid source")
    archive=CB59(blob)
    if archive.meta["original_unique_parameter_count"]!=P.ORIGINAL_UNIQUE_PARAMETERS:
        raise AssertionError("Original 360M unique source count drift")
    old=json.loads(Path(old_v193_json).read_text())
    if not old.get("complete") or grading_fingerprint()["sha256"]!=P.FROZEN_GRADER_SHA:
        raise AssertionError("Missing proven prior v193 original FP32 or v192 SHA")
    locked=old["frozen_before_any_model_generation"]
    if locked["grading_source_frozen_before_inference"]["sha256"]!=P.FROZEN_GRADER_SHA:
        raise AssertionError("Historical public original FP32 answer graders differ")
    questions=locked["full_questions_golds"]
    actual=old["actual_unfiltered_FP32_generations_96_each"]["360M"]
    if len(questions)!=96 or len(actual)!=96:
        raise AssertionError("Incomplete v193 original source QA")
    selected=[questions[i] for i in P.QA_IDS]
    baseline={int(a["case_id"]):a["complete_generation_original"]["verbatim_text"]
              for a in actual if int(a["case_id"]) in P.QA_IDS}
    if len(baseline)!=len(selected):
        raise AssertionError("Missing precomputed frozen original FP32 case")
    native_script=("same original physically charged CB59 sources for all arms, "
                   "DIAGNOSTIC original FP32 substitutions NOT a compact model")
    frozen={"original_physically_packed_360M_source_sha256":P.SOURCE_SHA256,
        "original_physical_archive_bytes":P.SOURCE_BYTES,
        "original_unique_parameters":P.ORIGINAL_UNIQUE_PARAMETERS,
        "QA_inputs_selected_before_any_new_model_inference":selected,
        "original_FP32_unmodified_verbatim_outputs":baseline,
        "original_original_FP32_baseline":summarize(selected,baseline),
        "arms_precommitted_without_any_TRAIN_fitting":list(P.ARMS),
        "v192_original_frozen_grader_SHA":P.FROZEN_GRADER_SHA,
        "all_arms_with_FP32_original_2D_are_UNPAID_diagnostics":True,
        "not_an_independent_BLIND_TEST":True,
        "no_private_v89_TEST":True,"no_Jessica_chat_test":True,
        "description":native_script}
    save(output/"v195_FROZEN_DIAGNOSTIC_FAMILY_PROTOCOL_BEFORE_MODEL_LOADING.json",frozen)

    # Load true original FP32 source only in this isolated diagnostic job.
    original=AutoModelForCausalLM.from_pretrained(
       P.MODEL,revision=P.REVISION,torch_dtype=torch.float32,
       attn_implementation="eager").cpu().eval()
    if sum(p.numel() for p in original.parameters())!=P.ORIGINAL_UNIQUE_PARAMETERS:
        raise AssertionError("Original FP32 teacher has incorrect unique parameters")
    if original.get_input_embeddings().weight.data_ptr()!=original.get_output_embeddings().weight.data_ptr():
        raise AssertionError("Original pretrained embedding/head tie lost")
    tokenizer=AutoTokenizer.from_pretrained(P.MODEL,revision=P.REVISION)
    config=AutoConfig.from_pretrained(P.MODEL,revision=P.REVISION)
    config._attn_implementation="eager"
    native=NativeKernel(compile_kernel(
          Path(__file__).resolve().parents[1]/"direct_kernel_v59.c",output/"native_360m_v195.so"))
    seen={}
    for arm in P.ARMS:
        # Model shell has no full dense 2D trained source; ONLY the
        # explicitly swapped original FP32 family holds such tensors.
        with init_empty_weights(include_buffers=False):
            model=AutoModelForCausalLM.from_config(config,torch_dtype=torch.float32).eval()
        runtime=bind_into_llama(model,archive,native)
        if runtime["runtime_dense_multi_dim_source_parameters"]!=0:
            raise AssertionError("Native packed base runtime contains unpaid dense source")
        audit=substituted_modules(model,original,arm)
        generated={};details=[]
        for case in selected:
            response=generate(model,tokenizer,case["question"],cap=P.TOKEN_CAP,stop=True)
            reference=actual[int(case["case_id"])]["complete_generation_original"]
            if response["prompt_sha"]!=reference["prompt_sha"]:
                raise AssertionError("Swapped original 360M FP32/native input differs from pretrained baseline")
            generated[case["case_id"]]=response["verbatim_text"]
            details.append({"case_id":case["case_id"],"domain":case["domain"],
                "question":case["question"],"gold":case["gold"],
                "original_preexisting_FP32_answer":baseline[case["case_id"]],
                "diagnostic_hybrid_unpaid_FP32_original_answer":response,
                "original_frozen_prompt_SHA":response["prompt_sha"]})
        scored=summarize(selected,generated)
        seen[arm]={"complete":True,"family_audit":audit,
            "frozen_v192_source_grader_SHA":P.FROZEN_GRADER_SHA,
            "machine_confirmed":scored["machine_confirmed"],
            "recognized_wrong":scored["machine_recognized_wrong"],
            "ungraded_not_known_wrong":scored["ungraded_not_known_wrong"],
            "opposing_pair_scores":{k:{"total":v["total"],"both_correct":v["both_correct"]}
                for k,v in scored["opposite_pairs"].items()},
            "identical_original_FP32_verbatim_count":sum(
                generated[k]==baseline[k] for k in generated),
            "all_real_original_FP32_hybrid_generations":details,
            "explicit_full_GRADES":scored,
            "this_arm_is_UNPAID_DIAGNOSTIC_not_physically_packed":True}
        save(output/"v195_unpaid_source_family_DIAGNOSTIC_partial.json",seen)
        print(json.dumps({"stage":"v195_UNPAID_source_family",
           "arm":arm,"correct":scored["machine_confirmed"],
           "wrong":scored["machine_recognized_wrong"],
           "ungraded":scored["ungraded_not_known_wrong"],
           "UNPAID_FP32_2D_bytes":audit["UNPAID_trained_FP32_2D_parameter_BYTES"],
           "original_verbatim_match":seen[arm]["identical_original_FP32_verbatim_count"]}),flush=True)
        del model
        gc.collect()
    if set(seen)!=set(P.ARMS):
        raise AssertionError("Missing noncompact family diagnostic arm")
    final={"complete":True,"frozen_protocol":frozen,
        "actual_pretrained_source_360M_FP32_loaded_DIAGNOSTIC_ONLY":True,
        "NOT_a_paid_BPP_model_QA_result":True,
        "source_binary_bytes":P.SOURCE_BYTES,"source_SHA":P.SOURCE_SHA256,
        "diagnostic_arms":seen,
        "negative_results_not_hidden":True,
        "private_v89_TEST_not_accessed":True,
        "main_Jessica_Render_Supabase_APK_untouched":True}
    save(output/"v195_360M_causal_original_FP32_unpaid_family_diagnostics.json",final)
    print(json.dumps({"stage":"v195_DIAGNOSTIC_complete",
        "all_arms":{k:{"confirmed":v["machine_confirmed"],
            "ungraded":v["ungraded_not_known_wrong"],
            "unpaid_FP32_bytes":v["family_audit"]["UNPAID_trained_FP32_2D_parameter_BYTES"]}
            for k,v in seen.items()},
        "no_physically_paid_quality_claim":True}),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("physical_source_file",type=Path)
    ap.add_argument("v193_frozen_original_96QA_json",type=Path)
    ap.add_argument("output",type=Path)
    args=ap.parse_args()
    run(args.physical_source_file,args.v193_frozen_original_96QA_json,args.output)
