"""S35 source-only decoder contracts. Fake logits are NOT original 43M proof."""
from __future__ import annotations
import inspect
import unittest
import torch

from research.nra3_s19.s19g_paid_semantic_graph import GraphLogits
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    PREV_NEG,LIKELY_DATIVE,
)
from research.nra3_s20.s35_real43m_local_negation_scope_trainonly import (
    S35_FIXED_TARGET_GROUPS,S35_SCOPE_RULE,
    apply_local_ne_dative_scope,verify_logits_unmodified,
)
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    PATTERNS,
)


class S35SourceOnlyLocalNegationScope(unittest.TestCase):
    def setUp(self):
        b,w=2,5
        self.valid=torch.ones((b,w),dtype=torch.bool)
        self.raw=torch.zeros((b,w,338),dtype=torch.float32)
        # The first row contains correct НЕ + dative at target index 1.
        self.raw[0,1,320+PREV_NEG]=1
        self.raw[0,1,320+LIKELY_DATIVE]=1
        # Second row has 'не' adjacent to the verb, not to recipient.
        self.raw[1,1,320+PREV_NEG]=1
        self.raw[1,3,320+LIKELY_DATIVE]=1
        def rand(shape):return torch.randn(*shape)
        self.model=GraphLogits(
            event_status=rand((b,w,6)),
            role=rand((b,w,w,9)),
            argument_polarity=rand((b,w,w,3)),
            coreference=rand((b,w,w,2)),
            event_relation=rand((b,w,w,5)),
        )
        self.model.argument_polarity[...,2]=12.0

    def test_correct_target_scope_preserved_false_scope_suppressed(self):
        new=apply_local_ne_dative_scope(self.model,self.raw,self.valid)
        self.assertTrue(torch.equal(
            new.argument_polarity[0,2,1],
            self.model.argument_polarity[0,2,1]))
        self.assertEqual(float(new.argument_polarity[0,2,2,2]),-1e6)
        self.assertEqual(float(new.argument_polarity[1,2,3,2]),-1e6)
        self.assertEqual(int(new.argument_polarity[0,2,1].argmax()),2)
        self.assertNotEqual(int(new.argument_polarity[0,2,2].argmax()),2)
        self.assertTrue(verify_logits_unmodified(self.model,new,self.raw,self.valid))

    def test_input_only_mask_no_pattern_id_no_gold_in_forward(self):
        s=inspect.getsource(apply_local_ne_dative_scope)
        for forbidden in ("gold","PATTERNS","fold","group","held"):
            self.assertNotIn(forbidden,s)
        self.assertEqual(S35_FIXED_TARGET_GROUPS,(26,28))
        self.assertEqual(PATTERNS[26],"rejected_positive_first")
        self.assertEqual(PATTERNS[28],"negated_female_past")
        self.assertIn("target_has_previous_NE_and_likely_DATIVE",S35_SCOPE_RULE)

    def test_does_not_modify_any_model_output_tensor(self):
        old=torch.clone(self.model.argument_polarity)
        new=apply_local_ne_dative_scope(self.model,self.raw,self.valid)
        self.assertTrue(torch.equal(self.model.argument_polarity,old))
        self.assertFalse(torch.equal(new.argument_polarity,old))
        self.assertTrue(torch.equal(new.role,self.model.role))
        self.assertTrue(torch.equal(new.event_relation,self.model.event_relation))
        self.assertTrue(torch.equal(new.coreference,self.model.coreference))
        self.assertTrue(torch.equal(new.event_status,self.model.event_status))


if __name__=="__main__":
    unittest.main()
