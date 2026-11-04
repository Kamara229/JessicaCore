"""Top facade: selected 14M roles, same-byte functional feedback (TRAIN only)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from v112.archive import Archive,create_archive
from v112.roles import weights
from v112.metrics import evaluate,patch_model,restore_model,source_digest
from .train import capture,recalibrate
from .protocol import MODEL,PINNED_REVISION,MODES,FIT,VALID,GENERATE,SHRINKAGE

@torch.inference_mode()
def validation_reference(model,tok):
    samples=[];logits=[]
    for prompt in VALID:
        ids=tok(prompt,return_tensors="pt")["input_ids"]
        if ids.shape[1] < 8:
            raise AssertionError("tiny invalid heldout prompt")
        samples.append(ids)
        logits.append(model(input_ids=ids,use_cache=False).logits[0,:-1].detach().float())
    return samples,logits

@torch.inference_mode()
def check_generation(model,tok):
    output=[]
    for prompt in GENERATE:
        tokens=tok(prompt,return_tensors="pt")["input_ids"]
        ids=model.generate(tokens,max_new_tokens=24,do_sample=False,
                           pad_token_id=tok.eos_token_id)
        new=ids[0,len(tokens[0]):]
        output.append({"prompt":prompt,
                       "generated_text":tok.decode(new,skip_special_tokens=True),
                       "generated_ids":new.tolist()})
    return output

def experiment(out):
    torch.set_num_threads(2)
    out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=PINNED_REVISION)
    model=AutoModelForCausalLM.from_pretrained(
        MODEL,revision=PINNED_REVISION,torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    source=weights(model)
    teacher_train=capture(model,tok,source,FIT)
    teacher_valid=capture(model,tok,source,VALID)
    samples,ref_scores=validation_reference(model,tok)
    report={
        "experiment":"CALIBRI v113 TRAIN-only same-byte operator feedback",
        "model":MODEL,"pretrained_revision":PINNED_REVISION,
        "original_selected_weights_sha256":source_digest(source),
        "selected_source_parameters":sum(w.size for w in source.values()),
        "train_sha256":[hashlib.sha256(x.encode()).hexdigest() for x in FIT],
        "valid_sha256":[hashlib.sha256(x.encode()).hexdigest() for x in VALID],
        "selected_only_no_full_070_checkpoint":True,
        "matched_physical_bytes_including_book_scales_manifest_crc_sha":True,
        "no_old_v89_TEST_or_public_QA_used":True,
        "fit_touched_no_valid_or_generation_text":True,
        "shrinkage_frozen":SHRINKAGE,
        "teacher":{"quality":evaluate(model,tok,samples,ref_scores),
                   "greedy":check_generation(model,tok)},
        "candidates":{}
    }
    physical=set()
    for mode in MODES:
        base=create_archive(source,mode)
        base_a=Archive(base)
        try:
            patch_model(model,source,base_a)
            student_train=capture(model,tok,source,FIT)
        finally:
            restore_model(model,source)
        controls={
            "source_only":(base,{}),
        }
        for kind,inputs in (("teacher_signal",teacher_train),
                            ("student_signal",student_train)):
            changed,diag=recalibrate(base,source,teacher_train,inputs)
            controls[kind]=(changed,diag)
        for kind,(blob,train_diag) in controls.items():
            a=Archive(blob)
            phys=a.summary()
            physical.add(phys["whole_source_bytes"])
            if not np.isfinite(phys["selected_bits_per_weight"]):
                raise AssertionError("invalid physical rate")
            label=mode+"__"+kind
            (out/(label+".clb112")).write_bytes(blob)
            from v112.metrics import activation_error
            record={"physical":phys,
                    "fit_diagnostics":train_diag,
                    "teacher_valid_activation_error":activation_error(source,a,teacher_valid)}
            try:
                patch_model(model,source,a)
                record["valid_quality"]=evaluate(model,tok,samples,ref_scores)
                record["greedy"]=check_generation(model,tok)
            finally:
                restore_model(model,source)
            report["candidates"][label]=record
            print(json.dumps({"stage":"v113_small_real","case":label,
                              "physical_bytes":phys["whole_source_bytes"],
                              "selected_bpp":phys["selected_bits_per_weight"],
                              "valid_teacher_KL":record["valid_quality"]["teacher_relative_kl"],
                              "valid_token_NLL":record["valid_quality"]["token_nll"],
                              "valid_teacher_activation_error":record["teacher_valid_activation_error"]["real_teacher_activation_rel_rmse"],
                              "output_snippet":[x["generated_text"][:60] for x in record["greedy"]]}),flush=True)
    if len(physical)!=1 or len(report["candidates"])!=9:
        raise AssertionError("incomplete or unequal physically paid experiment")
    report["all_9_physical_archive_bytes_identical"]=next(iter(physical))
    report["candidate_not_selected_from_valid"]=True
    report["shippable_model_ready"]=False
    (out/"v113_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"stage":"v113_done","real_cases":len(report["candidates"]),
                      "source_model_immutable":True,"no_production_change":True}),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    experiment(p.parse_args().output)
