"""v206 Green Tunnel: first-divergence repair using ONLY frozen opposing TRAIN.

Find the first step at which the compressed model's greedy continuation
disagrees with the complete original TRAIN answer, then backprop a margin
between the correct gold token and the strongest competing token there.
The student continues with its own generated prefixes; this is not teacher-
forced alignment of a fictitious good continuation or a DEV-answer oracle.
Original teacher-forced complete-answer and full-vocabulary objectives remain
in v206.run. This TRAIN loss is NOT a substitute for held-out QA parity.
"""
import torch
from torch.nn import functional as F
from v200.rollout import _sample


def first_divergence_margin(student, pair, *, steps_cap=6, margin=0.25):
    if not 2 <= steps_cap <= 12 or margin < 0:
        raise ValueError("Unapproved TRAIN rollout length or margin")
    if pair.get("split") != "TRAIN":
        raise AssertionError("Only frozen TRAIN opposing answers can guide repair")
    losses, traces = [], {}
    for side in ("left", "right"):
        prompt, gold = _sample(pair, side)
        context = prompt.detach().clone()[None]
        matched = 0
        last_logits = last_expected = None
        first_wrong = None
        for step in range(min(steps_cap, len(gold))):
            logits = student(input_ids=context, use_cache=False).logits[0, -1]
            if logits.ndim != 1 or logits.numel() < 2 or not torch.isfinite(logits).all():
                raise AssertionError("Full, finite TRAIN vocabulary expected")
            target_id = int(gold[step])
            if not 0 <= target_id < logits.numel():
                raise ValueError("TRAIN gold target outside the output vocabulary")
            proposed = logits.argmax().detach()
            last_logits, last_expected = logits, target_id
            if int(proposed) != target_id:
                first_wrong = step
                break
            matched += 1
            context = torch.cat((context, proposed.reshape(1, 1)), dim=1)
        if last_logits is None:
            raise AssertionError("Missing TRAIN gold answer token")
        # Strongest competing output token, including undesired repetition/EOS.
        # Value and gold IDs come ONLY from the frozen TRAIN sample.
        other = torch.cat((last_logits[:last_expected],last_logits[last_expected + 1:]))
        strongest_competitor = other.max()
        gap = strongest_competitor - last_logits[last_expected] + margin
        loss = F.softplus(gap)
        if not torch.isfinite(loss):
            raise AssertionError("Nonfinite first-divergence repair loss")
        losses.append(loss)
        traces[side] = {"first_mismatch_index": first_wrong,
                        "correct_greedy_prefix_tokens": matched,
                        "TRAIN_gold_target_id": last_expected,
                        "mismatch_was_seen": first_wrong is not None,
                        "gold_and_competitor_margin":float(
                            (last_logits[last_expected]-strongest_competitor).detach())}
    if int(pair["left"]["first_expected_id"]) == int(pair["right"]["first_expected_id"]):
        raise AssertionError("Opposing frozen TRAIN samples must contradict")
    return torch.stack(losses).mean(), {
        "TRAIN_pair_id":pair["pair_id"],"sides":traces,
        "gold_tokens_only_frozen_TRAIN":True,
        "no_public_DEV_VALID_or_private_TEST":True,
        "no_inference_time_sidecar":True}
