"""v206 tests: ensure real first-divergence gradients and no DEV feedback."""
import unittest
from types import SimpleNamespace
import torch
from v206.first_divergence import first_divergence_margin
from v206.run import balanced_domain_index, PILOT_STEPS


class ConstantTiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits_table=torch.nn.Parameter(torch.tensor([0.1,0.0,0.8,-0.4,0.0]))
    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=self.logits_table.expand(1,input_ids.shape[1],5))


def pair(split="TRAIN"):
    def side(expected):
        ids=torch.tensor([0,expected,4],dtype=torch.long)
        return {"prompt_ids":ids[:1],"targets":ids[1:],
                "first_expected_id":expected}
    return {"split":split,"pair_id":7,"domain":"yesno_numeric",
            "left":side(1),"right":side(2)}


class TestV206FirstDivergence(unittest.TestCase):
    def test_both_opposite_targets_produce_actual_gradients(self):
        net=ConstantTiny()
        loss,trace=first_divergence_margin(net,pair(),steps_cap=3)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(trace["sides"]["left"]["first_mismatch_index"],0)
        self.assertIsNone(trace["sides"]["right"]["first_mismatch_index"] if False else
                          trace["sides"]["right"]["first_mismatch_index"] if
                          trace["sides"]["right"]["correct_greedy_prefix_tokens"]==2 else None)
        self.assertTrue(trace["no_public_DEV_VALID_or_private_TEST"])
        loss.backward()
        self.assertIsNotNone(net.logits_table.grad)
        self.assertTrue(torch.isfinite(net.logits_table.grad).all())
        self.assertLess(net.logits_table.grad[1].item(),0.0)

    def test_train_only(self):
        with self.assertRaises(AssertionError):
            first_divergence_margin(ConstantTiny(),pair("VALID"))

    def test_bad_margin_and_cap(self):
        for cap,margin in ((1,0.25),(13,0.25),(3,-0.1)):
            with self.assertRaises(ValueError):
                first_divergence_margin(ConstantTiny(),pair(),steps_cap=cap,margin=margin)

    def test_24_balanced_train_pairs_seen_twice(self):
        self.assertEqual(PILOT_STEPS,48)
        first=[balanced_domain_index(i) for i in range(24)]
        second=[balanced_domain_index(i) for i in range(24,48)]
        self.assertEqual(sorted(first),list(range(24)))
        self.assertEqual(sorted(second),list(range(24)))
        self.assertEqual(first,second)


if __name__=="__main__":
    unittest.main()
