"""Properly balanced true/wrong comparisons for ONE shared Green Tunnel.

Each group includes (correct true, plausible wrong, generated wrong CB59,
generated wrong Pythia). Contrast answer semantics relative to question,
NOT absolute 'bad-looking text' and not teacher KL.
"""
import random
import torch
from torch.nn import functional as F
from v120.bytes_adapter import pairs
from .protocol import SEED,STEPS,QUESTION_GROUP_BATCH,LR,RANK_MARGIN

def train(model,rows):
    if len(rows)%4:
        raise AssertionError("same true/plausible/generated/generated group size required")
    for j in range(0,len(rows),4):
        seq=rows[j:j+4]
        if [r[3] for r in seq]!=[
            "independently_correct_reference","plausible_false_reference",
            "actual_generated_CB59_v119","actual_generated_Pythia14M"]:
            raise AssertionError("random or missing same-question group")
        if sum(r[2] for r in seq)!=1:
            raise AssertionError("multiple positive candidates per contrast")
        if len({r[0] for r in seq})!=1:
            raise AssertionError("question alignment broke contrastive supervision")
    torch.manual_seed(SEED)
    rng=random.Random(SEED)
    samples=pairs([(q,a) for q,a,lab,kind in rows]).reshape(-1,4,160)
    questions=len(samples)
    for p in model.parameters():p.requires_grad_(False)
    for p in model.blocks.layers[-1].parameters():p.requires_grad_(True)
    for p in model.score.parameters():p.requires_grad_(True)
    chosen=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(chosen,lr=LR,weight_decay=.001)
    curve=[]
    for step in range(STEPS):
        idx=[rng.randrange(questions) for _ in range(QUESTION_GROUP_BATCH)]
        group=samples[idx].reshape(-1,160)
        raw=model(group).reshape(-1,4)
        winner=raw[:,0]
        distractors=raw[:,1:]
        # Cross-candidate factual alignment and no collapsed accept-all/reject-all.
        pair_loss=F.softplus(distractors-winner[:,None]+RANK_MARGIN).mean()
        labels=torch.zeros_like(raw);labels[:,0]=1.
        balanced_bce=F.binary_cross_entropy_with_logits(
            raw,labels,pos_weight=torch.tensor(3.))
        # Calibrate two operating points, using TRAIN only.
        positive_calibration=F.relu(.6-torch.sigmoid(winner)).square().mean()
        loss=.50*pair_loss+.45*balanced_bce+.05*positive_calibration
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(chosen,1.)
        optimizer.step()
        if step%34==0 or step==STEPS-1:
            curve.append({
                "step":step,
                "pairwise_train_loss":float(pair_loss.detach()),
                "balanced_gold_BCE_train":float(balanced_bce.detach()),
                "train_gold_mean_score":float(torch.sigmoid(winner).mean().detach()),
                "train_false_mean_score":float(torch.sigmoid(distractors).mean().detach())})
    model.eval()
    return {"train_questions":questions,"steps":STEPS,"trace":curve,
            "one_shared_neural_encoder":True,"no_teacher_logit_objective":True}
