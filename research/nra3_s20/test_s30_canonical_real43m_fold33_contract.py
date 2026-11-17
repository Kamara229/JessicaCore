"""S30 contract: actual canonical SHA authority and TRAIN-only fold33 separation.

This unit test uses no original43M model weights; only GitHub Actions CI
physically verifies the genuine 43M+GT SHA-sealed archives and raw338 shard.
"""
import inspect
import unittest

from research.nra3_s20.s22_new256_counterfactual_train_graph_source import splits
from research.nra3_s20.s30_reproducible_canonical43m_fp16_fold33 import (
    FOLD,HELD_CONSTRUCTION,S29_ARCHIVED_S22D_CANONICAL,S29_ARCHIVED_S26_CANONICAL,
    audit,
)
from research.nra3_s20.s29_archived_original43m_canonical_fp16_identity import (
    audit_actual_archived_model,
)


class S30Original43mPaidFingerprintFold33OnlyContracts(unittest.TestCase):
    def test_fold33_train248_held8_not_part_of_training(self):
        axis,which,train,held=splits()[FOLD]
        self.assertEqual((FOLD,HELD_CONSTRUCTION),(33,25))
        self.assertEqual((axis,which,len(train),len(held)),("CONSTRUCTION",25,248,8))
        self.assertTrue(set(train).isdisjoint(set(held)))
        self.assertEqual(len(set(train)|set(held)),256)

    def test_original_43m_archived_parent_canonical_pins(self):
        self.assertEqual(S29_ARCHIVED_S22D_CANONICAL,
                         "35102946886b9f35d86f57f791b6ab6001b813e4f79397cff8e4d25c04e7a538")
        self.assertEqual(S29_ARCHIVED_S26_CANONICAL,
                         "02d4403ffbce01632d54d66349a01c3f10b81650d26d9540bb86f495cac62807")
        self.assertIn("canonical_paid_semantic_fp16_sha",
                      inspect.getsource(audit_actual_archived_model))

    def test_original_and_new_physical_sha_provenance_no_gold_forward(self):
        code=inspect.getsource(audit)
        self.assertIn("exact_real_report_audit(base_info,base_sha)",code)
        self.assertIn("archived_base=PaidSemanticEventGraph.from_physical",code)
        self.assertIn("archived_trial=EventGroundedInversionGraph.from_physical",code)
        self.assertIn("canon_base==canon_base_replay",code)
        self.assertIn("canon_trial==canon_trial_replay",code)
        self.assertIn("old.head_physical()==old_repeat.head_physical()",code)
        self.assertIn("new.head_physical()==new_repeat.head_physical()",code)
        self.assertIn("trraw,trvalid,trgold=",code)
        self.assertIn("dvraw,dvvalid=",code)
        self.assertNotIn("old(dvraw,dvvalid,trgold)",code)
        self.assertNotIn("new(dvraw,dvvalid,trgold)",code)
        self.assertIn("no_frozen_original43M_GT_weight_update",code)
        self.assertIn("fresh_refit_NOT_equivalent_to_historical_exact_fold",code)


if __name__=="__main__":
    unittest.main()
