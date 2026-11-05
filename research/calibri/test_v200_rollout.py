"""No-network tiny CPU checks for v200 TRAIN-only rollout and full-answer guard."""
import unittest
from types import SimpleNamespace
import torch
from v200.rollout import paired_greedy_rollout_loss,probe_greedy_train,sequence_no_regression_gate

class TinyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits=torch.nn.Parameter(torch.zeros(8))
    def forward(self,input_ids,use_cache=False):
        return SimpleNamespace(logits=self.logits[None,None,:].expand(1,input_ids.shape[1],8))

def training_pair(split="TRAIN"):
    def sample(first):
        return {"prompt_ids":torch.tensor([1,2]),
                "targets":torch.tensor([first,0]),"first_expected_id":first}
    return {"split":split,"domain":"opposite","pair_id":7,
            "left":sample(3),"right":sample(4)}

def fake_probe(exact=False,repeats=1,prefix=0):
    def side(gold,correct,p,r):
        return {"gold_tokens_with_EOS":gold,"generated_tokens":gold if correct else [7,7],
                "exact_with_EOS":correct,"correct_prefix_tokens":p,
                "adjacent_repetition_excess":r,"stopped_at_EOS":correct}
    return {7:{"domain":"opposite","sides":{
        "left":side([3,0],exact,prefix,repeats),
        "right":side([4,0],False,0,0)}}}

class TestV200SequenceRollout(unittest.TestCase):
    def test_genuine_teacher_free_greedy_gradients_both_opposites_and_EOS(self):
        model=TinyStudent()
        loss,trace=paired_greedy_rollout_loss(model,training_pair())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.logits.grad.abs().sum()),0)
        self.assertEqual(trace["TRAIN_gold_complete_sequence_prefixes"]["left"],[3,0])
        self.assertEqual(trace["TRAIN_gold_complete_sequence_prefixes"]["right"],[4,0])
    def test_repeated_generated_sequence_is_visible(self):
        model=TinyStudent()
        with torch.no_grad():model.logits[7]=8
        out=probe_greedy_train(model,{},[training_pair()],max_new_tokens=4)
        self.assertFalse(out[7]["sides"]["left"]["exact_with_EOS"])
        self.assertEqual(out[7]["sides"]["left"]["adjacent_repetition_excess"],3)
    def test_increased_repetition_rejected_despite_prefix_improvement(self):
        g=sequence_no_regression_gate(fake_probe(repeats=1,prefix=0),
                                      fake_probe(repeats=2,prefix=1))
        self.assertFalse(g["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"])
        self.assertTrue(g["repetition_regressed"])
    def test_original_complete_correct_train_answer_never_lost(self):
        g=sequence_no_regression_gate(fake_probe(exact=True,prefix=2),
                                      fake_probe(exact=False,prefix=0))
        self.assertFalse(g["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"])
        self.assertEqual(g["lost_original_complete_TRAIN_answers"],[(7,"left")])
    def test_positive_train_improvement_preserved(self):
        g=sequence_no_regression_gate(fake_probe(repeats=2,prefix=0),
                                      fake_probe(repeats=1,prefix=1))
        self.assertTrue(g["ACCEPT_FULL_SEQUENCE_TRAIN_ONLY"])
    def test_heldout_validation_blocked(self):
        with self.assertRaises(AssertionError):
            paired_greedy_rollout_loss(TinyStudent(),training_pair("VALID"))
        with self.assertRaises(AssertionError):
            probe_greedy_train(TinyStudent(),{},[training_pair("VALID")])
    def test_training_label_tampering_rejected(self):
        a=fake_probe();b=fake_probe()
        b[7]["sides"]["right"]["gold_tokens_with_EOS"]=[6,0]
        with self.assertRaises(ValueError):sequence_no_regression_gate(a,b)

if __name__=="__main__":unittest.main()
