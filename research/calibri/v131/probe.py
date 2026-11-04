"""Read-only first-logit diagnostic and head/base/bias decomposition.

No decoding and no oracle-conditioned generation; gold tokens are analyzed
only after the forward pass has produced logits for the unmodified question.
"""
from contextlib import contextmanager
import torch
from torch.nn import functional as F


@contextmanager
def head_capture(head):
    result = {}
    def hook(module, args):
        if len(args) != 1:
            raise AssertionError("unexpected head invocation")
        result["last_hidden"] = args[0].detach()[:, -1, :].clone()
    handle = head.register_forward_pre_hook(hook)
    try:
        yield result
    finally:
        handle.remove()


@torch.inference_mode()
def first_logits(model, tokenizer, question):
    from v108.generation import build_chat
    prompt = build_chat(tokenizer, question)
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not 1 < len(ids) < 160:
        raise ValueError("invalid chat prompt")
    head = model.lm_head
    with head_capture(head) as trace:
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0, -1]
    if "last_hidden" not in trace:
        raise AssertionError("head capture missing")
    if not logits.isfinite().all():
        raise ValueError("nonfinite native logits")
    # Only corrected head exposes original, u, v, bias.
    components = None
    if all(hasattr(head, key) for key in ("original", "u", "v", "bias")):
        hidden = trace["last_hidden"]
        baseline = head.original(hidden)[0]
        lowrank = F.linear(F.linear(hidden, head.v), head.u)[0]
        bias = head.bias.detach()
        reconstructed = baseline + lowrank + bias
        if not torch.allclose(reconstructed, logits, rtol=1e-5, atol=1e-4):
            raise AssertionError("lm_head decomposition does not reproduce actual logits")
        eos = tokenizer.eos_token_id
        winner = int(logits.argmax().item())
        components = {
            "eos_base": float(baseline[eos]), "eos_lowrank": float(lowrank[eos]),
            "eos_bias": float(bias[eos]), "eos_sum": float(reconstructed[eos]),
            "winner_id": winner,
            "winner_base": float(baseline[winner]),
            "winner_lowrank": float(lowrank[winner]),
            "winner_bias": float(bias[winner]),
            "winner_sum": float(reconstructed[winner]),
        }
    return logits.detach().clone(), components, len(ids)


def rank(logits, id_):
    if id_ < 0 or id_ >= logits.numel():
        raise ValueError("bad token ID")
    value = logits[id_]
    # Deterministic greedy tie break by ID, matching torch.argmax.
    idx = torch.arange(logits.numel(), device=logits.device)
    return 1 + int(((logits > value) | ((logits == value) & (idx < id_))).sum().item())


@torch.inference_mode()
def summarize_logits(logits, tokenizer, gold, foil, components, prompt_tokens):
    eos = tokenizer.eos_token_id
    p = torch.softmax(logits.to(torch.float64), dim=0)
    indices = logits.topk(8).indices.tolist()
    answer_ranks = {}
    for label, text in (("expected", gold), ("swapped", foil)):
        variants = []
        for prefix in ("", " "):
            encoded = tokenizer.encode(prefix + text, add_special_tokens=False)
            if encoded:
                variants.append({"first_token_id": int(encoded[0]),
                                 "rank": rank(logits, encoded[0]),
                                 "probability": float(p[encoded[0]])})
        answer_ranks[label] = variants
    others = logits.clone()
    others[eos] = -torch.inf
    return {
        "prompt_tokens": prompt_tokens,
        "argmax_id": int(indices[0]),
        "argmax_text": tokenizer.decode([int(indices[0])]),
        "eos_id": int(eos), "eos_rank": rank(logits, eos),
        "eos_probability": float(p[eos]),
        "eos_vs_best_non_eos_margin": float(logits[eos] - others.max()),
        "top8": [{"id": int(i), "text": tokenizer.decode([int(i)]),
                  "logit": float(logits[i])} for i in indices],
        "answer_first_token_probe_NOT_full_answer_accuracy": answer_ranks,
        "head_terms": components,
    }


@torch.inference_mode()
def controller_probe(policy, question, opposite, fixed):
    if policy is None:
        return None
    current, swapped, constant = (policy(text).detach().clone()
                                  for text in (question, opposite, fixed))
    def out(c):
        return [float(v) for v in c]
    return {
        "six_coefficients_current": out(current),
        "six_coefficients_swapped": out(swapped),
        "six_coefficients_constant": out(constant),
        "current_vs_swapped_l2": float(torch.linalg.vector_norm(current-swapped)),
        "current_vs_constant_l2": float(torch.linalg.vector_norm(current-constant)),
        "attention_gates_current": [float(policy.gates(current, depth, 0))
                                    for depth in (0.0167, 0.5, 0.9833)],
        "mlp_gates_current": [float(policy.gates(current, depth, 1))
                              for depth in (0.0167, 0.5, 0.9833)],
    }
