"""Balanced two-prompt whole-model STE and actual source-index proposals.

Every update is judged on the *sum* of the opposing TRAIN answers:
a constant "Yes" output cannot minimize a Yes/No pair. The
candidate has no access to historical public DEV or held-out VALID.
"""
import json
import math
import random
import torch
from torch.nn import functional as F

from v176.operators import clear
from v185.optim import make_student,propose_codes,apply_codes
from . import protocol as p


def paired_loss(model,source,pair,retain=False):
    logits_by_side=[]
    labels_by_side=[]
    cross_entropies=[]
    for side in ("left","right"):
        sample=pair[side]
        # Model-evaluation tensors, not inference-mode teacher views.
        ids=torch.tensor(sample["ids"].tolist(),dtype=torch.long)[None]
        positions=torch.tensor(sample["positions"].tolist(),dtype=torch.long)
        labels=torch.tensor(sample["targets"].tolist(),dtype=torch.long)
        logits=model(input_ids=ids,use_cache=False).logits[0]
        selected=logits[positions]
        loss=F.cross_entropy(selected,labels)
        cross_entropies.append(loss)
        logits_by_side.append(logits[positions[0]])
        labels_by_side.append(int(labels[0]))
    if labels_by_side[0]==labels_by_side[1]:
        raise AssertionError("Opposite paired answers share first token")
    gaps=[logits_by_side[i][labels_by_side[i]]-
          logits_by_side[i][labels_by_side[1-i]] for i in range(2)]
    margin=sum(F.relu(p.MARGIN-gap) for gap in gaps)/2
    ce=sum(cross_entropies)/2
    objective=ce+p.PAIR_MARGIN_WEIGHT*margin
    if not torch.isfinite(objective):
        raise ValueError("Nonfinite balanced TRAIN objective")
    return objective,{"paired_CE":float(ce.detach()),
        "opposite_answer_margin_loss":float(margin.detach()),
        "first_token_logit_gaps":[float(v.detach()) for v in gaps],
        "first_tokens":[int(v.detach().argmax()) for v in logits_by_side],
        "expected_first_tokens":labels_by_side}


def train(student,sources,training_pairs,original_weights,callback=None):
    if len(training_pairs)!=p.TRAIN_PAIR_COUNT:
        raise AssertionError("Incomplete precommitted balanced TRAIN")
    if set(original_weights)!=set(p.SCHEDULE):
        raise AssertionError("Original source gradient/prior target mismatch")
    torch.manual_seed(p.SEED)
    adam=torch.optim.Adam(
        [src.centroids for src in sources.values()],lr=p.CENTROID_LR)
    rng=random.Random(p.SEED)
    order=list(range(len(training_pairs)))
    curve=[]
    accepted=proposed=0
    for step,target_name in enumerate(p.SCHEDULE):
        if step%len(order)==0:rng.shuffle(order)
        train_index=order[step%len(order)]
        pair=training_pairs[train_index]
        source=sources[target_name]
        observed=[]
        def capture(_module,_args,source_tensor):
            if not source_tensor.requires_grad:
                raise AssertionError("No differentiable codebook source")
            if not any(source_tensor is t for t in observed):
                source_tensor.retain_grad()
                observed.append(source_tensor)
        handle=source.register_forward_hook(capture)
        clear(sources);adam.zero_grad(set_to_none=True)
        try:
            loss,before=paired_loss(student,source,pair)
            loss.backward()
        finally:
            handle.remove()
        if not observed or any(t.grad is None for t in observed):
            raise AssertionError("No full-model gradient to projection "+target_name)
        gradient=sum(t.grad.detach().cpu() for t in observed).numpy()
        params=[src.centroids for src in sources.values()]
        norm=float(torch.nn.utils.clip_grad_norm_(
            params,p.MAX_GRAD_NORM,error_if_nonfinite=True))
        adam.step()
        with torch.no_grad():
            for src in sources.values():src.centroids.clamp_(-16,16)
        clear(sources)
        with torch.inference_mode():
            frozen_loss,post_centroids=paired_loss(student,source,pair)
            baseline=float(frozen_loss)
        proposal=propose_codes(source,gradient,original_weights[target_name])
        if proposal["proposed_groups"]>p.MAX_INDEX_SWAPS_PER_STEP:
            raise AssertionError("Unregistered 5-bit index mutation budget")
        proposed+=proposal["proposed_groups"]
        if proposal["proposed_groups"]:
            with torch.no_grad():apply_codes(source,proposal,True)
            clear(sources)
            with torch.inference_mode():
                result,post_code=paired_loss(student,source,pair)
                candidate=float(result)
            improve=math.isfinite(candidate) and candidate+1e-7<baseline
            if not improve:
                with torch.no_grad():apply_codes(source,proposal,False)
                clear(sources)
            else:
                accepted+=proposal["proposed_groups"]
        else:
            improve=False;candidate=baseline;post_code=post_centroids
        data={"step":step+1,"pair_id":pair["pair_id"],"domain":pair["domain"],
              "target_operator":target_name,
              "train_loss_before_codebook_update":float(loss.detach()),
              "train_loss_after_codebook_before_codes":baseline,
              "train_loss_proposed_codes":candidate,
              "candidate_indices":proposal["proposed_groups"],
              "accepted_indices":proposal["proposed_groups"] if improve else 0,
              "accepted_on_pairwise_actual_TRAIN_loss":bool(improve),
              "balanced_pair_gradient_norm":norm,
              "opposing_target_ids":before["expected_first_tokens"],
              "logit_margin_before":before["first_token_logit_gaps"],
              "logit_margin_after":post_code["first_token_logit_gaps"]}
        curve.append(data)
        clear(sources)
        if (step+1)%8==0:
            print(json.dumps({"stage":"balanced_paired_TRAIN",**data}),flush=True)
        if callback and (step+1)%24==0:callback(step+1,curve)
    if len(curve)!=p.STEPS:raise AssertionError("Incomplete 96-step fit")
    for epoch in range(p.EPOCHS):
        selected=curve[epoch*p.TRAIN_PAIR_COUNT:(epoch+1)*p.TRAIN_PAIR_COUNT]
        if sorted(x["pair_id"] for x in selected)!=list(range(p.TRAIN_PAIR_COUNT)):
            raise AssertionError("Lost one opposing behavior pair per epoch")
    return {"total_full_model_gradient_steps":len(curve),
        "TRAIN_pairs_each_seen_exactly_four_times":True,
        "paired_targets_per_step":2,
        "frozen_binary_answer_balance":True,
        "actual_5bit_index_proposals":proposed,
        "actual_5bit_accepted_swaps":accepted,
        "all_FP16_codebooks_trained":len(sources),
        "source_bitwidth_unchanged":5,
        "TRAIN_only_selection_not_QA":True,
        "no_KL_NLL_PPL_quality_gate":True,
        "steps":curve}
