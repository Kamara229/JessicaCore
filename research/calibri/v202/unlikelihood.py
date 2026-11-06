"""CALIBRI v202: TRAIN-only greedy exposure and differentiable repeat correction.

Greedy rollouts use the compressed student's OWN argmax prefixes and the full
TRAIN-side reference (including EOS). We do not modify decoding, mask tokens,
add a sidecar or compute any objective from public DEV/heldout VALID/test.
Anti-repetition learning is deliberately local: penalize the prior emitted
token *only if the student emits that token again and TRAIN gold differs*.
Legitimate repeated tokens in gold are never penalized.
"""
import torch
from torch.nn import functional as F
from v200.rollout import _sample


def train_rollout_ce_and_repeat_unlikelihood(student, pair, *, steps_cap=6):
    if not 2 <= steps_cap <= 12:
        raise ValueError("Greedy recovery horizon must be 2..12 TRAIN tokens")
    side_ce, side_unlikelihood = [], []
    traces = {}
    for side in ("left", "right"):
        prompt, gold = _sample(pair, side)
        if len(gold) < 2 or int(gold[-1]) < 0:
            raise AssertionError("Complete TRAIN answer and EOS required")
        context = prompt.detach().clone()[None]
        per_token_ce, repeat_penalties = [], []
        generated = []
        penalized_positions = []
        for step in range(min(steps_cap, len(gold))):
            logits = student(input_ids=context, use_cache=False).logits[0, -1]
            if not torch.isfinite(logits).all():
                raise AssertionError("Nonfinite source-model output")
            target = gold[step].reshape(1)
            per_token_ce.append(F.cross_entropy(logits[None], target))
            predicted = logits.argmax().detach()
            token = int(predicted)
            if (generated and token == generated[-1]
                    and token != int(gold[step])):
                probability = torch.softmax(logits.float(), -1)[token]
                # Effective gradient even at extremely overconfident repeats;
                # no inference-time suppression or externally stored biases.
                repeat_penalties.append(
                    -torch.log1p(-probability.clamp(max=1.0 - 1e-6)))
                penalized_positions.append(step)
            generated.append(token)
            context = torch.cat((context, predicted.reshape(1, 1)), dim=1)
        cross_entropy = torch.stack(per_token_ce).mean()
        # Differentiable exact zero in absence of unwanted repetitions.
        unlikelihood = (torch.stack(repeat_penalties).mean()
                        if repeat_penalties else cross_entropy * 0.0)
        side_ce.append(cross_entropy)
        side_unlikelihood.append(unlikelihood)
        traces[side] = {
            "actual_greedy_tokens": generated,
            "TRAIN_target_tokens_including_EOS": [
                int(t) for t in gold[:steps_cap]],
            "unwanted_repetition_positions": penalized_positions,
            "legitimate_gold_repetition_exempt": True,
        }
    if int(pair["left"]["first_expected_id"]) == int(
            pair["right"]["first_expected_id"]):
        raise AssertionError("Opposing TRAIN pair no longer contrasts")
    ce = torch.stack(side_ce).mean()
    penalty = torch.stack(side_unlikelihood).mean()
    if not torch.isfinite(ce) or not torch.isfinite(penalty):
        raise AssertionError("Invalid TRAIN-only repeat correction objective")
    return ce, penalty, {
        "TRAIN_pair_id": pair["pair_id"],
        "TRAIN_greedy_rollout": traces,
        "no_public_DEV_or_VALID": True,
        "no_inference_time_repeat_mask": True,
    }
