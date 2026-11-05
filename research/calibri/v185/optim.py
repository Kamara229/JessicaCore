"""Quantization-aware, group-index projected discrete source optimization.

All changes are to PREVIOUSLY PAID 5-bit code indices + FP16 codebooks.
An index proposal is made by model TRAIN gradient, biased toward
original pretrained source weights; it is accepted/rejected using
the *actual* supervised TRAIN forward loss on the same frozen sample.

This is not a claim of converged QAT or recovered language quality.
"""
import json
import random
import numpy as np
import torch
from torch.nn import functional as F

from v176.operators import build,clear
from . import protocol as p


def make_student(original):
    model,sources=build(original)
    if len(sources)!=211:
        raise AssertionError("All 211 real trained source matrices required")
    model.requires_grad_(False)
    for source in sources.values():
        source.centroids.requires_grad_(True)
        source.log_scales.requires_grad_(False)
    trainable={id(s.centroids) for s in sources.values()}
    actual={id(x) for x in model.parameters() if x.requires_grad}
    if actual!=trainable or len(trainable)!=211:
        raise AssertionError("Only already-paid source centroids can be trained")
    return model,sources


def propose_codes(src,weight_gradient,original_fp32):
    """Return up to 512 strict reproducible gradient/projected code swaps."""
    matrix_shape=tuple(src.shape)
    if (weight_gradient.shape!=matrix_shape or
        original_fp32.shape!=matrix_shape or matrix_shape[1]%8):
        raise ValueError("Source matrix and teacher W mismatch")
    scales=src.base_scales.detach().cpu().numpy().astype(np.float32)
    if np.any(scales<=0) or not np.isfinite(scales).all():
        raise AssertionError("Invalid original immutable row scales")
    centers=src.centroids.detach().cpu().numpy().astype("<f2").astype(np.float32)
    if centers.shape!=(32,8):raise ValueError("Wrong saved V8 codebook")
    nrows,ncols=matrix_shape
    groups=ncols//8
    weight_ref=np.asarray(original_fp32,dtype=np.float32)
    grad=np.asarray(weight_gradient,dtype=np.float32)
    if not np.isfinite(weight_ref).all() or not np.isfinite(grad).all():
        raise ValueError("Nonfinite actual TRAIN gradient")
    old=src.codes.detach().cpu().numpy().astype(np.uint8)
    if len(old)!=nrows*groups:raise AssertionError("Incomplete source codes")
    centers2=(centers**2).sum(1)[None,:]
    ranks=[]
    for first in range(0,len(old),4096):
        last=min(len(old),first+4096)
        index=np.arange(first,last,dtype=np.int64)
        rows=index//groups
        columns=(index%groups)*8
        coords=columns[:,None]+np.arange(8)[None,:]
        target=(weight_ref[rows[:,None],coords]/scales[rows,None])
        # Objective gradient transformed from W to normalized codebooks.
        normalized_gradient=(grad[rows[:,None],coords]*scales[rows,None])
        norm=np.linalg.norm(normalized_gradient.astype(np.float64),axis=1)
        good=norm>1e-15
        normalized_gradient/=np.maximum(norm[:,None],1.e-15)
        grad_scores=normalized_gradient@centers.T
        original_energy=(target*target).sum(axis=1,keepdims=True)
        source_square_error=np.maximum(
            (original_energy+centers2-2*target@centers.T),0.0)
        source_penalty=p.SOURCE_GUARD_STRENGTH*(
            source_square_error/np.maximum(original_energy,1.e-5))
        score=grad_scores+source_penalty
        winner=score.argmin(axis=1).astype(np.uint8)
        score_old=score[np.arange(len(index)),old[first:last]]
        gain=score_old-score[np.arange(len(index)),winner]
        good &= winner!=old[first:last]
        good &= gain>p.MIN_SCORE_IMPROVEMENT
        for local in np.flatnonzero(good):
            ranks.append((float(gain[local]),int(index[local]),int(winner[local])))
    ranks.sort(key=lambda x:(-x[0],x[1]))
    chosen=ranks[:p.MAX_INDEX_UPDATES_PER_STEP]
    return {"indices":[item[1] for item in chosen],
            "new_codes":[item[2] for item in chosen],
            "old_codes":[int(old[item[1]]) for item in chosen],
            "score_improvement":[item[0] for item in chosen],
            "candidate_source_groups":len(old),
            "proposed_groups":len(chosen)}


def apply_codes(source,proposal,use_new):
    values=proposal["new_codes" if use_new else "old_codes"]
    if len(values)!=len(proposal["indices"]):
        raise AssertionError("Invalid indexed source update")
    if not values:return
    index=torch.tensor(proposal["indices"],dtype=torch.long)
    code=torch.tensor(values,dtype=torch.long)
    source.codes[index]=code


def train(student,sources,samples,teacher_weights,make_reference):
    if len(samples)!=len(p.PUBLIC_TRAIN_INDICES):
        raise ValueError("Not frozen v185 TRAIN data")
    optimizer=torch.optim.Adam([src.centroids for src in sources.values()],
                               lr=p.LEARNING_RATE_CODEBOOKS)
    rng=random.Random(p.SEED)
    order=list(range(len(samples)))
    objective_history=[]
    attempted=accepted=0
    full_steps=0
    for step,target_name in enumerate(p.STEPS_OPERATORS):
        if step%len(order)==0:rng.shuffle(order)
        sample_index=order[step%len(order)]
        ref=make_reference[sample_index]
        ids=torch.tensor(ref["ids"].tolist(),dtype=torch.long)
        positions=torch.tensor(ref["positions"].tolist(),dtype=torch.long)
        targets=torch.tensor(ref["targets"].tolist(),dtype=torch.long)
        source=sources[target_name]
        retained=[]
        def grad_capture(_module,_args,result):
            if not result.requires_grad:
                raise AssertionError("TRAIN source dense reconstruction unexpectedly frozen")
            result.retain_grad()
            retained.append(result)
        handle=source.register_forward_hook(grad_capture)
        clear(sources)
        optimizer.zero_grad(set_to_none=True)
        try:
            output=student(input_ids=ids,use_cache=False).logits
            loss=F.cross_entropy(output[0,positions],targets)
            if not torch.isfinite(loss):raise ValueError("Bad model TRAIN CE")
            loss.backward()
        finally:
            handle.remove()
        if len(retained)!=1 or retained[0].grad is None:
            raise AssertionError("Missing actual compressed tensor gradient")
        weight_gradient=retained[0].grad.detach().cpu().numpy()
        gradients=[src.centroids for src in sources.values()]
        norm=float(torch.nn.utils.clip_grad_norm_(
            gradients,p.WEIGHT_GRAD_MAX_NORM,error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():
            for src in sources.values():
                src.centroids.clamp_(-16,16)
        clear(sources)
        with torch.inference_mode():
            before=student(input_ids=ids,use_cache=False).logits[0,positions]
            loss_before=float(F.cross_entropy(before,targets))
        proposal=propose_codes(source,weight_gradient,teacher_weights[target_name])
        attempted+=proposal["proposed_groups"]
        if proposal["proposed_groups"]:
            with torch.no_grad():
                apply_codes(source,proposal,True)
            clear(sources)
            with torch.inference_mode():
                after=student(input_ids=ids,use_cache=False).logits[0,positions]
                loss_after=float(F.cross_entropy(after,targets))
            keep=np.isfinite(loss_after) and loss_after+1e-7<loss_before
            if not keep:
                with torch.no_grad():apply_codes(source,proposal,False)
                clear(sources)
            else:
                accepted+=proposal["proposed_groups"]
        else:
            keep=False
            loss_after=loss_before
        item={"step":step+1,"matrix":target_name,
            "sample_index":sample_index,
            "TRAIN_CE_before_centroid_step":float(loss.detach()),
            "TRAIN_CE_before_discrete_candidate":loss_before,
            "TRAIN_CE_after_discrete_candidate":loss_after,
            "candidate_group_swaps":proposal["proposed_groups"],
            "accepted_group_swaps":proposal["proposed_groups"] if keep else 0,
            "accept_actual_TRAIN_loss_improvement":bool(keep),
            "trainable_codebook_gradient_norm":norm}
        objective_history.append(item)
        full_steps+=1
        clear(sources)
        if (step+1)%4==0:
            print(json.dumps({"stage":"quantized_index_training",**item}),flush=True)
    if full_steps!=p.STEPS:
        raise AssertionError("Missing steps")
    return {"actual_end_to_end_full_model_backward_steps":full_steps,
        "candidate_index_swaps":attempted,
        "accepted_index_swaps":accepted,
        "learning_updates_only_prepaid_5bit_codes_or_fp16_centroids":True,
        "no_qa_or_private_TEST_to_choose_indices":True,
        "fixed_training_steps_and_order":True,
        "training_CE_not_answer_acceptance":True,
        "training_curve":objective_history}
