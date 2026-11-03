"""Frozen original FP32 single-block swaps: causal probes, not deployable precision."""
from __future__ import annotations
import torch

MAX_SELECTED_BLOCKS=6

def compare_logits(teacher_logits,student_logits,answer_ids):
    t=teacher_logits.float().log_softmax(-1)
    s=student_logits.float().log_softmax(-1)
    result=float((t.exp()*(t-s)).sum().item())
    if result < -1e-4:
        raise AssertionError("negative teacher KL indicates corrupt numeric results")
    yes,no=answer_ids
    return {"teacher_student_KL":result,
            "top1_matches_teacher":int(int(teacher_logits.argmax())==int(student_logits.argmax())),
            "yes_no_margin":float((student_logits[yes]-student_logits[no]).item()),
            "teacher_yes_no_margin":float((teacher_logits[yes]-teacher_logits[no]).item())}

def candidates_by_local_error(local):
    # Highest raw relative error may not be the largest downstream influence.
    # Hence we include first/last down projection and head / embeddings too.
    top=sorted((k for k in local if k.startswith("model.layers.")),
        key=lambda k:local[k]["relative_RMS_error"],reverse=True)
    selected=["lm_head","model.embed_tokens","model.layers.0.mlp.down_proj",
              "model.layers.29.mlp.down_proj"]
    visited=set()
    for name in top:
        layer=name.split(".")[2]
        if layer in visited or name in selected:continue
        visited.add(layer)
        selected.append(name)
        if len(visited)>=MAX_SELECTED_BLOCKS:break
    return list(dict.fromkeys(selected))

@torch.inference_mode()
def causal_one_block_rescue(student,teacher,local,inputs,answer_tokens):
    selected=candidates_by_local_error(local)
    baselines=[compare_logits(x["teacher_logits"],x["baseline_logits"],answer_tokens)
               for x in inputs]
    report=[]
    for name in selected:
        path,_,attr=name.rpartition(".")
        parent=student.get_submodule(path) if path else student
        old=getattr(parent,attr)
        exact=teacher.get_submodule(name)
        if old is exact:raise AssertionError("not swapping quantized module")
        setattr(parent,attr,exact)
        try:
            scores=[]
            for x in inputs:
                ids=torch.tensor(x["tokens"],dtype=torch.long)[None,:]
                logits=student(input_ids=ids,use_cache=False).logits[0,-1,:].float()
                scores.append(compare_logits(x["teacher_logits"],logits,answer_tokens))
        finally:
            setattr(parent,attr,old)
        gain=sum(b["teacher_student_KL"]-s["teacher_student_KL"] for b,s in zip(baselines,scores))/len(scores)
        report.append({"module":name,"mean_KL_gain_from_exact_FP32_one_module":gain,
            "local_same_input_relative_RMS":local[name]["relative_RMS_error"],
            "per_original_TRAIN_prompt_scores":scores})
    report.sort(key=lambda x:x["mean_KL_gain_from_exact_FP32_one_module"],reverse=True)
    return {"original_baseline_kl_by_prompt":baselines,
            "temporary_FP32_one_module_swaps_not_physically_paid":True,
            "ranked_single_module_rescues":report,
            "strongest_gain_not_the_same_as_local_error":True,
            "caveat":"One truncated TRAIN prompt from each original train subset; "
                      "oracle FP32 swaps are upper-bound probes, not deployable models."}
