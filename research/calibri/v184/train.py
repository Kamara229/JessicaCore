"""Fixed-count, first-token and supervised answer TRAIN, no DEV leakage.

End-to-end gradients across all 30 transformer blocks update every
rank-two A matrix and TWO shared B matrices. CB59 dictionary and 1D
pretrained weights are frozen. This is *training* CE and not an
NLL/PPL quality acceptance criterion. QA is always a separate gate.
"""
import json,random,time
import torch
from torch.nn import functional as F
from . import protocol as p
from .trainable import clear_frozen

@torch.inference_mode()
def freeze_train_references(teacher,samples):
    if len(samples)!=p.TRAIN_CASES:raise ValueError("Incorrect frozen TRAIN set")
    records=[]
    for item in samples:
        ids=item["ids"][None]
        if item["answer_start"] is None:
            top=teacher(input_ids=ids,use_cache=False).logits.argmax(-1)
            length=ids.shape[1]
            positions=torch.arange(max(0,length-5),length,dtype=torch.long)
            targets=top[0,positions].clone()
        else:
            start=item["answer_start"]-1
            positions=torch.arange(start,ids.shape[1]-1,dtype=torch.long)
            targets=ids[0,positions+1].clone()
        if positions.numel()==0:
            raise ValueError("No TRAIN positions to optimize")
        records.append({"ids":ids.clone(),"positions":positions,
                        "targets":targets,"kind":item["kind"]})
    return records

def train(student,sources,alphas,betas,references):
    learned_alpha=list(alphas.values())
    learned_beta=list(betas.values())
    if len(learned_alpha)!=211 or len(learned_beta)!=2:
        raise AssertionError("Missing cross-layer trainable factor")
    optimizer=torch.optim.Adam([
        {"params":learned_alpha,"lr":p.LR_ALPHA},
        {"params":learned_beta,"lr":p.LR_BETA}])
    trained_ids={id(a) for a in learned_alpha+learned_beta}
    if set(id(x) for x in student.parameters() if x.requires_grad)!=trained_ids:
        raise AssertionError("Unaccounted trainable model weights")
    order=list(range(len(references)))
    rng=random.Random(p.SEED)
    curve=[]
    student.eval()
    started=time.perf_counter()
    for step in range(p.STEPS):
        if step%len(references)==0:rng.shuffle(order)
        i=order[step%len(references)]
        sample=references[i]
        clear_frozen(sources)
        optimizer.zero_grad(set_to_none=True)
        # Original teacher references are frozen in inference_mode; indices
        # must be ordinary leaf tensors for gradient-enabled embedding ops.
        ids=torch.tensor(sample["ids"].tolist(),dtype=torch.long)
        positions=torch.tensor(sample["positions"].tolist(),dtype=torch.long)
        targets=torch.tensor(sample["targets"].tolist(),dtype=torch.long)
        output=student(input_ids=ids,use_cache=False).logits
        selected=output[0,positions]
        loss=F.cross_entropy(selected,targets)
        if not torch.isfinite(loss):raise ValueError("Nonfinite supervised TRAIN CE")
        loss.backward()
        if step==0:
            missing=[name for name,a in alphas.items() if a.grad is None]
            if missing:raise AssertionError("No gradient for source operators: "+str(missing))
        norm=float(torch.nn.utils.clip_grad_norm_(
              learned_alpha+learned_beta,p.MAX_GRAD_NORM,error_if_nonfinite=True))
        optimizer.step()
        current={"step":step+1,"sample_index":i,"kind":sample["kind"],
                 "TRAIN_cross_entropy":float(loss.detach()),
                 "global_grad_norm":norm,
                 "elapsed_seconds":time.perf_counter()-started}
        curve.append(current)
        if (step+1)%4==0:print(json.dumps({"stage":"shared_rank2_training",**current}),flush=True)
        del output,selected,loss
        clear_frozen(sources)
    return {"train_steps":p.STEPS,
         "trained_alpha_parameters":sum(x.numel() for x in learned_alpha),
         "shared_basis_parameters":sum(x.numel() for x in learned_beta),
         "gradient_traverses_entire_model":True,
         "fixed_last_iterate_no_development_selection":True,
         "training_loss_not_quality_gate":True,
         "TRAIN_curve":curve}
