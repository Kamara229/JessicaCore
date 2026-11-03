"""Pure quality calculations, on original TRAIN only."""
import torch

@torch.inference_mode()
def score(logits, teacher_logits, ids):
    if logits.shape != teacher_logits.shape or logits.ndim != 2 or len(ids) != len(logits):
        raise AssertionError("mismatched original token contexts")
    lp = teacher_logits[:-1].double().log_softmax(-1)
    lq = logits[:-1].double().log_softmax(-1)
    kl = float((lp.exp() * (lp - lq)).sum(-1).mean())
    nll = float(-lq.gather(1, ids[1:].long()[:, None]).mean())
    if not (-1e-7 <= kl < float("inf") and 0 <= nll < float("inf")):
        raise AssertionError("nonfinite real original full-model scores")
    return {"full_vocab_KL": kl, "next_token_NLL": nll,
            "teacher_top1": float((lp.argmax(-1) == lq.argmax(-1)).float().mean()),
            "tokens": len(ids)-1}

def mean(rows):
    if not rows:
        raise AssertionError("missing original TRAIN token windows")
    return {k: sum(r[k] for r in rows)/len(rows) if k != "tokens"
            else sum(r[k] for r in rows)
            for k in rows[0]}
