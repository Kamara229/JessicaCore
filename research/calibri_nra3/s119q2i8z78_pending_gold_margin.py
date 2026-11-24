"""Z78 preregistered training-only hardest-invalid margin on pending bytes.
No inference constraints. Planned comparison from the exact revived Z73
checkpoint, same existing head, same data/order/optimizer, one new loss.
Physical execution is separate and conditional on Z77 diagnostic completion.
"""
import torch
from torch.nn import functional as F
from research.calibri_nra3.s119q2i8z65_stable_gate_train import objective_loss,ARMS as JOINT_ARMS
ARMS=("REVIVED_BASE","REVIVED_PENDING_GOLD_MARGIN")
MARGIN=.5
WEIGHT=.5
EPOCHS=4
PARENT_SHA="25acb994cfb8f6e425127da6419b0f47fa5c955b6bbf69d888186e985b1c5256"
GT_SHA="79fd6297ebd350140f15d68bea9146c08e89b767ed6404c7ba311793616d44a6"
def pending_margin(logits,target,mask,normalizer):
    if logits.ndim!=2 or logits.shape[0]!=1 or mask.shape!=(logits.shape[1],) or mask.dtype!=torch.bool:
        raise ValueError("full native vocabulary and bool mask required")
    if not 0<=target<mask.numel() or not bool(mask[target]) or normalizer<1:
        raise ValueError("legal gold and positive pending count required")
    if bool(mask[2]):
        return logits.new_zeros(())
    if not bool((~mask).any()):
        raise ValueError("pending state must contain invalid competitors")
    x=logits[0].float()
    # Gradient acts on the actual strongest invalid competitor and gold;
    # CE still teaches selection among the valid tokens.
    return WEIGHT*F.softplus(x[~mask].max()-x[target]+MARGIN)/normalizer
def row_loss(head,entry,contrast,arm):
    if arm not in ARMS:raise ValueError("fixed matched arms only")
    gold=entry["gold"]
    logits=entry["base"]+.25*head(entry["hidden"])[:,0,:]
    pending=sum(not bool(m[2]) for m in entry["masks"])
    losses=[];ce_sum=0.;margin_sum=0.;aux_sum=0.
    for t in range(len(gold)):
        m=entry["masks"][t]
        total,ce,pair,eos,utf8=objective_loss(logits[t:t+1],gold,t,arm=JOINT_ARMS[1],contrast=contrast,mask=m)
        margin=pending_margin(logits[t:t+1],gold[t],m,max(1,pending)) if arm==ARMS[1] else logits.new_zeros(())
        losses.append(total+margin)
        ce_sum+=float(ce.detach())*len(gold)
        aux_sum+=float((total-ce).detach())
        margin_sum+=float(margin.detach())
    return torch.stack(losses).sum(),{"CE_sum":ce_sum,"existing_aux_sum":aux_sum,"margin_sum":margin_sum,"pending_positions":pending}
def manifest():
    return {"stage":"Z78_PENDING_GOLD_MARGIN_PREREG","arms":ARMS,"parent_joint_SHA256":PARENT_SHA,
        "GT12_SHA256":GT_SHA,"epochs_per_arm":EPOCHS,"TRAIN_rows":98,"native_positions_per_epoch":1618,
        "pending_positions_per_epoch":123,"planned_cached_positions":12944,"planned_Adam_updates":784,
        "trainable_parameters":247040,"projection_LR":.0008,"bias_LR":.000025,"clip":.6,
        "margin":MARGIN,"weight":WEIGHT,"pending_normalizer":"number of pending positions in each answer",
        "control_objective":"unchanged CE+pair+EOS+UTF8","treatment_difference":"add pending_gold_vs_strongest_invalid softplus",
        "physical_execution_requires_Z77_completion":True,"no_DEV_scoring":True,"no_FP32":True,"no_inference_masks":True,
        "new_parameters":0,"candidate_accepted":False,"output_gate":0}
if __name__=="__main__":
    import json
    print("Z78_PREREG",json.dumps(manifest(),sort_keys=True),flush=True)
