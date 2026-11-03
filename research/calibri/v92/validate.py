"""Full model paired output metrics; no official independent TEST and no tuning on it."""
from __future__ import annotations
import time
import torch
from torch import nn
from direct_codebook_v59 import CB59
from direct_runtime_v59 import PackedLinear
from v90.generation import generate_diagnostic

QA_VALID_CASES=12

@torch.inference_mode()
def score_validation(model,teacher,rows,yes_no_ids):
    yes,no=map(int,yes_no_ids)
    scans=[]
    for row in rows[:QA_VALID_CASES]:
        ids=torch.tensor(row["tokens"],dtype=torch.long)[None,:]
        raw=model(input_ids=ids,use_cache=False).logits[0,-1,:].float()
        gold=teacher(input_ids=ids,use_cache=False).logits[0,-1,:].float()
        lp=gold.log_softmax(-1)
        lq=raw.log_softmax(-1)
        kl=(lp.exp()*(lp-lq)).sum().item()
        margin=(raw[yes]-raw[no]).item()
        pred=0 if margin>=0 else 1
        p2=torch.log_softmax(raw[torch.tensor([yes,no])],dim=0)
        scans.append({"record_index":row["record_index"],"prompt_SHA":row["prompt_SHA"],
                      "teacher_student_KL":kl,"gold":row["gold"],"pred":pred,
                      "correct":int(pred==row["gold"]), "yes_no_margin":margin,
                      "gold_conditional_two_token_NLL":float(-p2[row["gold"]].item())})
    return {"n":len(scans),"correct":sum(x["correct"] for x in scans),
            "mean_teacher_student_KL":sum(x["teacher_student_KL"] for x in scans)/len(scans),
            "mean_gold_two_token_NLL":sum(x["gold_conditional_two_token_NLL"] for x in scans)/len(scans),
            "items":scans,
            "NLL_scope":"constrained_two_token_BoolQ_gold_NLL_only_NOT_language_NLL"}

def swap_one_block(student,original_archive:bytes,candidate_archive:bytes,module_name:str):
    before=CB59(original_archive)
    after=CB59(candidate_archive)
    mod=student.get_submodule(module_name)
    if not isinstance(mod,PackedLinear):
        raise AssertionError("native compressed block required for exact same-speed swap")
    candidate=after.handles[module_name+".weight"]
    if candidate.shape!=mod.handle.shape:
        raise AssertionError("incompatible operator shape")
    path,_,leaf=module_name.rpartition(".")
    parent=student.get_submodule(path)
    return parent,leaf,mod,PackedLinear(candidate,mod.kernel)

@torch.inference_mode()
def compare_frozen_whole_models(student,teacher,cases,answers,archive_original,archive_rewritten,block_name):
    valid=cases["TRAIN_VALID"]
    orig=score_validation(student,teacher,valid,answers)
    g0=generate_diagnostic(student,None) if False else None
    parent,leaf,old,new=swap_one_block(student,archive_original,archive_rewritten,block_name)
    try:
        setattr(parent,leaf,new)
        corrected=score_validation(student,teacher,valid,answers)
    finally:
        setattr(parent,leaf,old)
    kl_gain=orig["mean_teacher_student_KL"]-corrected["mean_teacher_student_KL"]
    return {"original":orig,"candidate":corrected,
            "teacher_KL_gain":kl_gain,
            "VALID_QA_count_change":corrected["correct"]-orig["correct"],
            "VALID_gold_two_token_NLL_change":corrected["mean_gold_two_token_NLL"]-orig["mean_gold_two_token_NLL"],
            "changes_only_one_full_native_compressed_block":True}

@torch.inference_mode()
def microbenchmark_one_operator(student,archive_original,archive_rewritten,block_name):
    parent,leaf,old,new=swap_one_block(student,archive_original,archive_rewritten,block_name)
    in_features=old.in_features
    x=torch.arange(in_features,dtype=torch.float32).reshape(1,-1)/(in_features+1)
    def measure(mod):
        mod(x)
        samples=[]
        for _ in range(12):
            start=time.perf_counter()
            mod(x)
            samples.append(time.perf_counter()-start)
        samples.sort()
        return sum(samples[3:9])/6
    base=measure(old)
    cand=measure(new)
    return {"original_seconds":base,"candidate_seconds":cand,
            "candidate_speed_ratio":base/max(cand,1e-9),
            "same_native_K32V8_kernel":True,
            "microbench_not_full_model_speed":True}
