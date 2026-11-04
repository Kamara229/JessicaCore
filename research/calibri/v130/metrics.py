"""Separate free answers, conditional lexical evidence and hidden-state drift."""
import re

import torch
from torch.nn import functional as F
from v128.scoring import matches


def grade(text, answer, eos):
    first = re.match(r"^\s*([A-Za-z]+)\b", text)
    word = first.group(1).casefold() if first else None
    return {"first_word_matches": word == answer.casefold(),
            "exact_text": matches(text, answer),
            "exact_text_with_eos": bool(eos and matches(text, answer))}


@torch.inference_mode()
def likelihood(model, tok, prefix_ids, word):
    """Full teacher-forced conditional log P of the lexical answer only.

    Two distinct first-boundary tokenizations are tested for BOTH choices;
    EOS is deliberately excluded to separate content from answer formatting.
    Not a generation success rate or a calibrated answer probability.
    """
    variants = []
    for string in (word, " " + word):
        suffix = tok.encode(string, add_special_tokens=False)
        if not suffix:
            raise ValueError("empty lexical candidate")
        seq = torch.tensor([prefix_ids + suffix[:-1]], dtype=torch.long)
        logits = model(input_ids=seq, use_cache=False).logits[0]
        selected = logits[len(prefix_ids)-1:len(prefix_ids)-1+len(suffix)]
        if selected.size(0) != len(suffix) or not torch.isfinite(selected).all():
            raise ValueError("bad causal lexical score")
        lp = F.log_softmax(selected.float(), dim=-1)
        score = lp[torch.arange(len(suffix)), torch.tensor(suffix)].sum()
        variants.append({"logp": float(score), "tokens": len(suffix),
                         "leading_space": string.startswith(" ")})
    result = max(variants, key=lambda x: x["logp"])
    return {"best_logp": result["logp"], "variant": result,
            "token_counts": [v["tokens"] for v in variants]}


def similarity(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.shape != b.shape:
        raise ValueError("incomparable activations")
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("nonfinite activations")
    denominator = max(float(a.norm()), 1e-9)
    return {"relative_error": float((a-b).norm()) / denominator,
            "cosine": float(F.cosine_similarity(a, b, dim=0, eps=1e-9))}


def swap_alignment(teacher_a, teacher_b, candidate_a, candidate_b):
    """Compare change caused by reversing the record, not absolute activations."""
    reference = teacher_a - teacher_b
    candidate = candidate_a - candidate_b
    if float(reference.norm()) < 1e-8:
        return None
    return similarity(reference, candidate)
