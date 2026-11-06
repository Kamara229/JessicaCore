"""No-network v202 spurious repetition gradient and TRAIN-only safeguards."""
from types import SimpleNamespace
import unittest
import torch
from v202.unlikelihood import train_rollout_ce_and_repeat_unlikelihood


class TinyStudent(torch.nn.Module):
    def __init__(self, winner=7, size=11):
        super().__init__()
        logits = torch.zeros(size)
        logits[winner] = 7.5
        self.logits = torch.nn.Parameter(logits)
    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.logits[None, None, :].expand(
            1, input_ids.shape[1], -1))


def pair(split="TRAIN", next_token=0):
    def arm(first):
        return {"prompt_ids": torch.tensor([1, 2]),
                "targets": torch.tensor([first, next_token, 0]),
                "first_expected_id": first}
    return {"split": split, "domain": "opposite", "pair_id": 23,
            "left": arm(3), "right": arm(4)}


class TestV202RepeatCorrection(unittest.TestCase):
    def test_unwanted_greedy_repeat_creates_gradient(self):
        model = TinyStudent()
        ce, penalty, details = train_rollout_ce_and_repeat_unlikelihood(
            model, pair(), steps_cap=3)
        self.assertGreater(float(penalty.detach()), 0)
        self.assertEqual(
            details["TRAIN_greedy_rollout"]["left"]["unwanted_repetition_positions"],
            [1, 2])
        self.assertEqual(
            details["TRAIN_greedy_rollout"]["right"]["unwanted_repetition_positions"],
            [1, 2])
        penalty.backward()
        self.assertTrue(torch.isfinite(model.logits.grad).all())
        self.assertGreater(float(model.logits.grad.abs().sum()), 0)
        self.assertGreater(float(model.logits.grad[7]), 0)

    def test_legitimate_repeat_in_gold_is_not_penalized(self):
        model = TinyStudent()
        _, penalty, info = train_rollout_ce_and_repeat_unlikelihood(
            model, pair(next_token=7), steps_cap=2)
        self.assertEqual(float(penalty), 0.0)
        self.assertEqual(
            info["TRAIN_greedy_rollout"]["left"]["unwanted_repetition_positions"], [])

    def test_no_unwanted_repeat_is_exact_zero(self):
        class Alternating(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param=torch.nn.Parameter(torch.zeros(11))
            def forward(self, input_ids, use_cache=False):
                out=self.param.clone()
                out[7 if input_ids.shape[1] % 2 == 0 else 8]+=4
                return SimpleNamespace(logits=out[None,None,:].expand(
                    1,input_ids.shape[1],-1))
        model=Alternating()
        _, penalty, _=train_rollout_ce_and_repeat_unlikelihood(model,pair())
        self.assertEqual(float(penalty),0.0)

    def test_explicit_heldout_guard(self):
        with self.assertRaises(AssertionError):
            train_rollout_ce_and_repeat_unlikelihood(TinyStudent(),pair("VALID"))
        with self.assertRaises(AssertionError):
            train_rollout_ce_and_repeat_unlikelihood(TinyStudent(),pair("TEST"))

    def test_missing_eos_and_bad_horizon_rejected(self):
        bad=pair()
        bad["left"]["targets"]=torch.tensor([3])
        with self.assertRaises(AssertionError):
            train_rollout_ce_and_repeat_unlikelihood(TinyStudent(),bad)
        with self.assertRaises(ValueError):
            train_rollout_ce_and_repeat_unlikelihood(TinyStudent(),pair(),steps_cap=13)


if __name__=="__main__":
    unittest.main()
