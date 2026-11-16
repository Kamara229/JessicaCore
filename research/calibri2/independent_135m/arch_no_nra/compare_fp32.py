#!/usr/bin/env python3
"""True pretrained FP32 vs native-width CALIBRI2 architecture (NO NRA) report.

The original input and source do not train on, choose parameters using, or
inspect the sealed test questions.  S0 and S1 are physically pinned artifacts.
A green CI does NOT mean model quality or native CPU speed parity.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
import shutil
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from binary_geometry_v64 import MODEL, MODEL_SHA, EXPECTED_UNIQUE
from direct_codebook_v59 import CB59
from v176.operators import clear
from v196.recover import build_trainable_from_paid_source
from calibri2.independent_135m.arch_no_nra.architecture import (
    Calibri135NoNRA, CompactProfileBank, physical_report
)
from calibri2.independent_135m.s0.run import batched_paths

S0_SHA="50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c"
S1_SHA="55e350047b1485f88c43451f5ae3fbff8481506702e2823a0734567b11cf1fda"
ORIGINAL_BYTES=11229303
MAX_NEW=12
SEED=261116
# ALL freshly authored benchmark items are EVAL ONLY. No gold is used in inference.
# Do not use any of these questions to tune model, GT, slot, router, or checkpoints.
EVAL=(
    ("What is the capital city of Japan? Answer with one word.", "Tokyo", "fact"),
    ("Which planet is known as the Red Planet? One-word answer.", "Mars", "fact"),
    ("Which gas do humans need to breathe to survive? One word.", "oxygen", "fact"),
    ("Which month comes after September? Answer with one word.", "October", "fact"),
    ("Calculate 6 + 7. Return only the number.", "13", "arithmetic"),
    ("What is 15 minus 6? Answer using only a digit.", "9", "arithmetic"),
    ("Calculate 3 times 4. Only the answer.", "12", "arithmetic"),
    ("Compute half of 18. Respond with one number.", "9", "arithmetic"),
    ("Write a one-word antonym of 'empty'.", "full", "language"),
    ("What is the past tense of 'go'? One word.", "went", "language"),
    ("What is the plural of 'mouse'? One word.", "mice", "language"),
    ("Which comes first alphabetically: apple or banana? One word.", "apple", "language"),
)


def sha(data:bytes):
    return hashlib.sha256(data).hexdigest()


def save(path,data):
    Path(path).write_text(json.dumps(data,indent=2,sort_keys=True,
         ensure_ascii=False)+"\n",encoding="utf-8")


def logger(stage,**fields):
    print(json.dumps({"stage":stage,**fields},ensure_ascii=False),flush=True)


def prompt_context(tok,prompt):
    txt=tok.apply_chat_template(
        [{"role":"user","content":prompt}],tokenize=False,
        add_generation_prompt=True)
    ids=tok.encode(txt,add_special_tokens=False)
    if len(ids)>170 or len(ids)<4:
        raise AssertionError("Eval prompt outside frozen context envelope")
    return torch.tensor([ids],dtype=torch.long)


@torch.inference_mode()
def generate(backbone,src,tok,context):
    seq=context.clone()
    generation=[]
    ended=False
    t0=time.perf_counter()
    for step in range(MAX_NEW):
        if src is not None:clear(src)
        outputs=backbone(input_ids=seq,use_cache=False)
        logits=outputs.logits[0,-1]
        if not bool(torch.isfinite(logits).all()):
            raise AssertionError("Nonfinite model outputs")
        tid=int(logits.argmax())
        if tid==tok.eos_token_id:
            ended=True
            break
        generation.append(tid)
        seq=torch.cat((seq,torch.tensor([[tid]],dtype=torch.long)),dim=1)
    if src is not None:clear(src)
    return {"tokens":generation,
            "text":tok.decode(generation,skip_special_tokens=True,
                              clean_up_tokenization_spaces=False),
            "EOS_seen":ended,
            "new_tokens":len(generation),
            "elapsed_wall_s":round(time.perf_counter()-t0,5)}


def normalized_answer(text):
    t=text.strip().casefold().strip(" .!?\n\t\r")
    t=re.sub(r"^(?:(?:the )?answer(?: is|:)\s*)","",t)
    return t.strip(" .!?")


def frozen_evaluation(backbone,src,tok,contexts,reference=""):
    rows=[]
    for i,(question,gold,domain) in enumerate(EVAL):
        got=generate(backbone,src,tok,contexts[i])
        row={"id":i,"domain":domain,"gold":gold,
             "strict_normalized_exact":normalized_answer(got["text"])==gold.casefold(),
             "prompt_sha256":sha(question.encode("utf-8")),
             **got}
        rows.append(row)
        logger("135M_arch_noNRA_evaluation",model=reference,id=i,
               domain=domain,text=got["text"][:90],
               correct=row["strict_normalized_exact"],EOS=got["EOS_seen"])
    return rows


def aggregate(rows):
    counts={}
    for domain in sorted({r["domain"] for r in rows}):
        subset=[r for r in rows if r["domain"]==domain]
        counts[domain]={"correct":sum(r["strict_normalized_exact"] for r in subset),
                        "total":len(subset)}
    return {
        "correct":sum(int(r["strict_normalized_exact"]) for r in rows),
        "total":len(rows),
        "EOS_seen":sum(int(r["EOS_seen"]) for r in rows),
        "by_domain":counts,
        "generation_wall_clock_s_sum":round(sum(r["elapsed_wall_s"] for r in rows),3),
        "avg_generated_tokens":sum(r["new_tokens"] for r in rows)/len(rows)
    }


def check_full_parity(base,architecture):
    differences=[]
    for x,y in zip(base,architecture):
        if (x["tokens"]!=y["tokens"] or x["EOS_seen"]!=y["EOS_seen"]):
            differences.append({"id":x["id"],"B":x["tokens"],"C":y["tokens"]})
    if differences:
        raise AssertionError(
            "Zero-init CALIBRI sidecars changed original CB59 answers! "+
            json.dumps(differences[:2]))
    return True


def shadow_features(wrapper,src,tok,context):
    with torch.inference_mode():
        clear(src)
        result=wrapper.backbone(input_ids=context,use_cache=False,
                                output_hidden_states=True)
        hidden=result.hidden_states[-1]
        proposal=wrapper.slots(hidden,context)
        choice=wrapper.router.suggest(context[0].tolist())
        clear(src)
    return {
        **choice,
        "two_slot_source_span_token_ids":proposal["chosen_source_span_ids"][0].tolist(),
        "source_span_confidence_untrained":float(proposal["shadow_confidence"][0]),
        "slot_not_authorized_to_overwrite_generation":True
    }


def run(s0, s1, out):
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Seal EVAL question/prompt hashes BEFORE downloading/loading model.
    manifest={
      "benchmark_frozen_before_weight_load":True,
      "original_trained_FP32":MODEL,
      "original_HF_revision":MODEL_SHA,
      "questions":[{"id":i,"question":q,"gold":g,
                    "domain":d,"sha256":sha(q.encode())}
                    for i,(q,g,d) in enumerate(EVAL)],
      "sample_count":len(EVAL),
      "tokens_generated_max":MAX_NEW,
      "original_unique_model_parameters":EXPECTED_UNIQUE,
      "gold_never_seen_by_router_tunnel_or_generation":True,
      "NRA_and_S3_role_adapters_disabled_and_not_loaded":True,
      "no_model_training_on_EVAL":True,
      "no_Jessica_chat_tests_no_LIVE_Render_Supabase_APK":True,
      "primary_metric":"strict_normalized_literally_correct_generated_answer",
      "secondary_metrics":("same_exact_output_as_original_FP32",
                           "EOS","first_token","generation_time"),
      "C0_must_match_B_exactly":True,
      "profile_router_and_slots_default_not_armed":True,
      "native_C_speed_not_evaluated":True,
    }
    save(out/"FROZEN_EVAL_12_MANIFEST_BEFORE_WEIGHTS.json",manifest)
    logger("MANIFEST_FROZEN_BEFORE_WEIGHTS",cases=len(EVAL),
           SHA256=sha(json.dumps(manifest,sort_keys=True).encode()))
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    contexts=[prompt_context(tok,q) for q,_,_ in EVAL]
    s0raw=Path(s0).read_bytes();s1raw=Path(s1).read_bytes()
    if (sha(s0raw)!=S0_SHA or sha(s1raw)!=S1_SHA or
        len(s0raw)!=ORIGINAL_BYTES or len(s1raw)!=ORIGINAL_BYTES):
        raise AssertionError("Wrong immutable S0/S1 source artifact")
    orig=CB59(s0raw);alternative=CB59(s1raw)
    if orig.meta["original_unique_parameter_count"]!=EXPECTED_UNIQUE:
        raise AssertionError("Original unique pretrained parameter count drift")
    # Compare original pretrained FP32, not random source or C2 43M teacher.
    fp32=AutoModelForCausalLM.from_pretrained(MODEL,revision=MODEL_SHA,
                                              torch_dtype=torch.float32).cpu().eval()
    if sum(x.numel() for x in fp32.parameters())!=EXPECTED_UNIQUE:
        raise AssertionError("Original FP32 geometry not loaded")
    original=frozen_evaluation(fp32,None,tok,contexts,"FP32_original_trained")
    del fp32;gc.collect()
    logger("FP32_reference_complete",correct=aggregate(original)["correct"])
    student,sources=build_trainable_from_paid_source(orig)
    student.cpu().eval().requires_grad_(False)
    direct=frozen_evaluation(student,sources,tok,contexts,"B_direct_original_CB59")
    logger("B_direct_CB59_complete",correct=aggregate(direct)["correct"])
    wrapper=Calibri135NoNRA(student)
    wrapper.eval().requires_grad_(False)
    c0=frozen_evaluation(wrapper.backbone,sources,tok,contexts,
                         "C0_CALIBRI2_noNRA_internal_GT_fullDepth_G0")
    zero_parity=check_full_parity(direct,c0)
    if tuple(wrapper.full_depth.last_full_coverage)!=tuple(range(
            wrapper.layout["native_depth"])):
        raise AssertionError("Not all native 135M layers traversed by GT")
    bank=CompactProfileBank(orig,alternative)
    bankpath=out/"S1_PROFILE_SPARSE_PAID_DELTA.safetensors"
    bank.write(bankpath)
    # All components are real trainable neural modules; initialized for exact
    # B parity and not yet qualified for being armed to replace answers.
    sidepath=out/"CALIBRI_135M_noNRA_GT_ROUTER_SLOTS_FP16.safetensors"
    n_tensors=wrapper.store(sidepath)
    bank.activate(sources,orig,1)
    clear(sources)
    c1=frozen_evaluation(wrapper.backbone,sources,tok,contexts,
                         "C1_S1_shadow_physical_profile")
    bank.activate(sources,orig,0)
    clear(sources)
    parity_roundtrip=check_full_parity(direct,frozen_evaluation(
        wrapper.backbone,sources,tok,contexts,
        "C0_roundtrip_after_profile_restore"))
    shadow=[shadow_features(wrapper,sources,tok,c) for c in contexts]
    # One frozen core prefill for each slot; separately, real sampled
    # branches (3 actual batched neural computations, no gold/FP32 teacher).
    paths=[]
    for i in range(4):
        row=batched_paths(wrapper.backbone,sources,tok,contexts[i],n=3)
        paths.append({"id":i,"question":EVAL[i][0],**row})
    # Physically save EXACT source and every additional eligible neural
    # parameter. No FP32 model, full S1 duplicate, or NRA sidecar.
    corepath=out/"CALIBRI2_135M_SOURCE_CB59.cb59"
    shutil.copyfile(s0,corepath)
    component_manifest={
      "architecture_name":"CALIBRI2-like pretrained original 135M, NO NRA",
      "native_core_config_sha":MODEL_SHA,
      "original_core_sha":S0_SHA,
      "auxiliary_trainable_parameters":wrapper.auxiliary_parameters(),
      "sidecar_parameter_tensors":n_tensors,
      "components":wrapper.layout,
      "alternate_S1_profile_is_compact_centroid_and_group_index_delta":True,
      "alt_profile_CHANGED_code_groups":bank.total_changes,
      "alt_profile_changed_centroid_matrices":len(bank.changed_centroid_tensors),
      "GT_internal_identity_zero_initialized":True,
      "GT_full_layer_identity_zero_initialized":True,
      "prompt_router_trained_and_armed":False,
      "parallel_slot_head_trained_and_armed":False,
      "S3_or_NRA_adapter_loaded":False,
      "no_FP32_or_two_full_archives_in_inference_package":True,
      "S0_SHA":S0_SHA,"S1_SHA_provenance_only":S1_SHA,
      "requires_full_train_and_independent_eval_before_model_promotion":True
    }
    manpath=out/"CALIBRI2_NO_NRA_PHYSICAL_PACKAGE_MANIFEST.json"
    save(manpath,component_manifest)
    physical=physical_report(corepath,bankpath,sidepath,manpath)
    if not physical["passes_0p700_bpp"]:
        raise AssertionError("Complete architecture sidecar exceeds 0.700 bpp")
    summary={
      "status":"architecture_port_and_readonly_paired_EVAL_completed",
      "NOT_qualified_as_FP32_parity":True,
      "not_native_C_speed":True,
      "physical":physical,
      "architecture":component_manifest,
      "question_count":len(EVAL),
      "FP32":aggregate(original),
      "B_direct_CB59":aggregate(direct),
      "C0_136M_architecture_GT_zero":aggregate(c0),
      "C1_S1_training_shadow_profile":aggregate(c1),
      "B_C0_exact_output_parity":zero_parity,
      "profile_roundtrip_restored_B_parity":parity_roundtrip,
      "C0_vs_FP32_identical_token_sequences":sum(
          int(a["tokens"]==b["tokens"] and a["EOS_seen"]==b["EOS_seen"])
          for a,b in zip(original,c0)),
      "C1_vs_FP32_identical_token_sequences":sum(
          int(a["tokens"]==b["tokens"] and a["EOS_seen"]==b["EOS_seen"])
          for a,b in zip(original,c1)),
      "FP32_first_token_matches_C0":sum(
          int(bool(a["tokens"]) and bool(b["tokens"]) and
              a["tokens"][0]==b["tokens"][0])
          for a,b in zip(original,c0)),
      "neural_slot_shadow_only":shadow,
      "three_real_parallel_branch_tests_no_teacher":paths,
      "all_EVAL_model_full_generated_outputs":{
          "FP32":original,"B_direct_CB59":direct,
          "C0_CALIBRI2_noNRA":c0,"C1_S1_profile":c1,
      },
      "not_a_proof_of_open_ended_language_parity":True,
      "not_claiming_C2_43M_parameter_compatibility":True,
      "training_beyond_S1_not_started_in_this_comparison":True,
    }
    save(out/"ACTUAL_135M_FP32_VS_CALIBRI2_NO_NRA_PAIRED_EVAL.json",summary)
    logger("FINAL_135M_C2_architecture_no_NRA_vs_FP32",
           fp32_correct=summary["FP32"]["correct"],
           direct_correct=summary["B_direct_CB59"]["correct"],
           calibri_correct=summary["C0_136M_architecture_GT_zero"]["correct"],
           s1_profile_correct=summary["C1_S1_training_shadow_profile"]["correct"],
           exact_sequence_parity_against_FP32=
               summary["C0_vs_FP32_identical_token_sequences"],
           zero_init_parity_pass=zero_parity,
           fully_paid_bpp=physical["effective_total_bpp"],
           native_depth=wrapper.layout["native_depth"],
           donor_layers=wrapper.layout["donor_layers"],
           total_auxiliary_params=wrapper.auxiliary_parameters())


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("S0_original_CB59",type=Path)
    p.add_argument("S1_trained_CB59",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    run(a.S0_original_CB59,a.S1_trained_CB59,a.output)
