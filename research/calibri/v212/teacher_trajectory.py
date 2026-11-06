"""CALIBRI v212: FP32 free-running trajectories frozen exclusively on TRAIN inputs.

The teacher generates its OWN continuations, then the physically compressed
student is trained on the whole-vocabulary teacher behavior at those exact
prefixes. In contrast to static teacher forcing, the model sees contexts the
original FP32 model actually traverses. Not a QA grade or evidence of parity.
No DEV, VALID, TEST, external chat or runtime neural sidecar is used.
"""
import hashlib
import torch
from torch.nn import functional as F

from v196 import protocol as p

TRAJECTORY_STEPS = 4
TEMPERATURE = 2.0
STATIC_TEXT_PREFIX_TOKENS = 24
TEACHER_SOFT_WEIGHT = .70


@torch.no_grad()
def freeze_fp32_train_trajectories(teacher, train_samples, *, eos_id, steps=TRAJECTORY_STEPS):
    if len(train_samples) != len(p.PUBLIC_TRAIN_INDICES) or not (1 <= steps <= 8):
        raise AssertionError("Only frozen twenty public TRAIN source records are permitted")
    if not (0 <= eos_id < p.VOCAB_SIZE):
        raise AssertionError("Invalid original FP32 EOS")
    teacher.eval()
    frozen = []
    for i, row in enumerate(train_samples):
        if row["kind"] == "supervised_chat":
            n = row["answer_start"]
            if n is None or n < 1:
                raise AssertionError("Missing original TRAIN conversation prefix")
        elif row["kind"] in ("wiki", "news"):
            if row["answer_start"] is not None:
                raise AssertionError("Unsupervised text record misclassified")
            n = STATIC_TEXT_PREFIX_TOKENS
        else:
            raise AssertionError("Outside frozen original public TRAIN types")
        ids = row["ids"]
        if not torch.is_tensor(ids) or ids.ndim != 1 or n >= len(ids):
            raise AssertionError("Frozen token data or boundary invalid")
        prefix = ids[:n].long().cpu().clone()
        generated = []
        distributions = []
        context = prefix[None]
        for _ in range(steps):
            logits = teacher(input_ids=context, use_cache=False).logits[0,-1].float()
            if logits.shape != (p.VOCAB_SIZE,) or not torch.isfinite(logits).all():
                raise AssertionError("Unfrozen or nonfinite original FP32 vocab")
            prob = torch.softmax(logits / TEMPERATURE, dim=-1)
            chosen = int(logits.argmax())
            distributions.append(prob.cpu().to(torch.float16).clone())
            generated.append(chosen)
            if chosen == eos_id:
                break
            context = torch.cat([context, context.new_tensor([[chosen]])], dim=1)
        if not generated or len(generated) > steps:
            raise AssertionError("Missing actual FP32 autoregressive TRAIN trajectory")
        frozen.append({
            "split":"TRAIN", "kind":row["kind"],
            "source_public_train_index":p.PUBLIC_TRAIN_INDICES[i],
            "original_prefix_token_sha256":hashlib.sha256(prefix.numpy().tobytes()).hexdigest(),
            "prefix":prefix,
            "teacher_greedy_next_ids":torch.tensor(generated,dtype=torch.long),
            "teacher_full_vocab_probs_fp16":torch.stack(distributions),
            "FP32_teacher_self_generated_not_gold_answer":True,
        })
    if len(frozen)!=20 or len(set(x["source_public_train_index"] for x in frozen))!=20:
        raise AssertionError("Teacher sampled anything other than 20 original TRAIN inputs")
    return frozen


def trajectory_distillation_loss(student, frozen, *, temperature=TEMPERATURE):
    if frozen.get("split")!="TRAIN" or not frozen.get("FP32_teacher_self_generated_not_gold_answer"):
        raise AssertionError("Forbidden DEV/VALID/TEST or untrusted generated record")
    if temperature != TEMPERATURE:
        raise AssertionError("Precommitted FP32 trajectory temperature changed")
    prefix=frozen["prefix"]
    greedy=frozen["teacher_greedy_next_ids"]
    teacher_probs=frozen["teacher_full_vocab_probs_fp16"].float()
    if (prefix.ndim!=1 or greedy.ndim!=1 or not 1<=len(greedy)<=TRAJECTORY_STEPS
            or teacher_probs.shape!=(len(greedy),p.VOCAB_SIZE)):
        raise AssertionError("Missing complete full-vocabulary teacher trajectory")
    if not (torch.isfinite(teacher_probs).all()
            and torch.all((greedy>=0)&(greedy<p.VOCAB_SIZE))):
        raise AssertionError("Invalid full-vocabulary FP32 trajectory")
    # One differentiable student forward over teacher's autoregressive path,
    # and predict each next teacher token from the corresponding left context.
    ids=torch.cat((prefix,greedy))
    positions=torch.arange(len(prefix)-1,len(prefix)+len(greedy)-1)
    logits=student(input_ids=ids[None],use_cache=False).logits[0,positions]
    if logits.shape!=teacher_probs.shape or not torch.isfinite(logits).all():
        raise AssertionError("Unpaid or incomplete student vocabulary trajectory")
    soft=-(teacher_probs*F.log_softmax(logits/temperature,dim=-1)).sum(-1).mean()
    hard=F.cross_entropy(logits,greedy)
    loss=TEACHER_SOFT_WEIGHT*soft+(1-TEACHER_SOFT_WEIGHT)*hard
    if not torch.isfinite(loss):
        raise AssertionError("Nonfinite FP32 autoregressive behavior restoration")
    return loss,{
        "TRAIN_source_index":frozen["source_public_train_index"],
        "FP32_teacher_steps":len(greedy),
        "complete_teacher_49152_vocab":True,
        "FP32_self_generated_sequence":greedy.tolist(),
        "TRAIN_teacher_soft_CE":float(soft.detach()),
        "TRAIN_teacher_greedy_token_CE":float(hard.detach()),
        "no_unseen_QA_or_feedback":True,
    }
