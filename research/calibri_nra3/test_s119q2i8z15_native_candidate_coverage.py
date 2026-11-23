"""Z15 native49152 fixed topK probability/coverage and original physical provenance."""
import inspect
import unittest
import torch
from research.calibri_nra3.s119q2i8z15_native_candidate_coverage_contract import (
    VOCAB,candidate_distribution,posthoc_gold_second,
)
from research.calibri_nra3.s119q2i8z14_fresh14_bpe2_coverage_source import (
    BUDGETS,verify,
)
from research.calibri_nra3 import s119q2i8z15_real43m_second_topk_coverage as physical

class TestZ15TargetBlind(unittest.TestCase):
    def test_source_and_budgets_locked(self):
        self.assertEqual(verify()["new_DEV14"],14)
        self.assertEqual(VOCAB,49152)
        self.assertEqual(BUDGETS,(8,16,32,64,128,512))
        self.assertIn("BPE2_TOPK_COVERAGE",physical.STAGE)
        self.assertIn("z10_dir",inspect.signature(physical.run).parameters)
    def test_native_full49152_topk_then_posthoc_gold(self):
        x=torch.full((1,VOCAB),-40.)
        for rank,token in enumerate((123,124,125,126,127,128,129,130,131)):
            x[0,token]=10.-rank
        distribution=candidate_distribution(x)
        self.assertEqual(distribution["native_greedy_second_id"],123)
        self.assertEqual(distribution["top512_candidate_native_ids"][:5],[123,124,125,126,127])
        scored=posthoc_gold_second(distribution,127,own_first_correct=True)
        self.assertFalse(scored["greedy_SECOND_correct"])
        self.assertTrue(scored["conditional_correct_gold_BPE2_in_topK"]["8"])
        self.assertEqual(scored["native_gold_rank_if_in_top512"],5)
        self.assertTrue(distribution["target_gold_never_used_for_candidate_selection"])
        self.assertLessEqual(distribution["topK_probability_mass"]["8"],1.00001)
    def test_first_wrong_cannot_count_gold_coverage(self):
        x=torch.zeros((1,VOCAB))
        x[0,101]=99.
        c=candidate_distribution(x)
        out=posthoc_gold_second(c,101,own_first_correct=False)
        self.assertFalse(out["greedy_SECOND_correct"])
        self.assertIsNone(out["native_gold_rank_if_in_top512"])
        self.assertEqual(sum(out["conditional_correct_gold_BPE2_in_topK"].values()),0)
    def test_EOS_never_hidden_and_invalid_logits_fail_closed(self):
        x=torch.zeros((1,VOCAB))
        x[0,2]=40.
        c=candidate_distribution(x)
        self.assertEqual(c["rank_of_EOS_native49152"],1)
        self.assertTrue(c["EOS_in_top8"])
        for bad in (torch.ones((1,17)),torch.full((1,VOCAB),float("nan"))):
            with self.assertRaises(ValueError):candidate_distribution(bad)
        with self.assertRaises(ValueError):
            posthoc_gold_second(c,49152,own_first_correct=True)
    def test_no_gradient_and_protected_model_no_training_in_physical_runner(self):
        code=inspect.getsource(physical.run)
        self.assertIn("torch.no_grad()",code)
        self.assertIn("posthoc_gold_second(",code)
        self.assertIn("candidate_distribution(",code)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",code)
        self.assertNotIn("optimizer.step()",code)
        self.assertIn('"output_gate":0',code)
if __name__=="__main__":unittest.main()
