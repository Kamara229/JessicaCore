"""S32 physical starting state, exact TRAIN guard and no-QA contracts."""
import unittest
from pathlib import Path
from calibri2.s32.microguard_pcgrad_fp32 import (
    SOURCE_SHA,S29_SEALED_SHA,STEPS,BLOCK_STEPS,LR,
    OLD_GRAD_BLEND,CURRICULUM,guarded_old_ids,
)
from calibri2.s30.matched_dense_fp32_train import (
    SOURCE_SHA as S30_START,S29_FROZEN_SHA
)

class S32Guard(unittest.TestCase):
    def test_single_precommitted_micro_update_program(self):
        self.assertEqual(SOURCE_SHA,S30_START)
        self.assertEqual(S29_SEALED_SHA,S29_FROZEN_SHA)
        self.assertEqual(STEPS,96)
        self.assertEqual(BLOCK_STEPS,2)
        self.assertEqual(LR,0.000025)
        self.assertEqual(OLD_GRAD_BLEND,.65)
        self.assertEqual(CURRICULUM,("new","new","old","new"))

    def test_old_answer_identity_not_count(self):
        before=[
            {"id":"one","full_correct":True},
            {"id":"two","full_correct":True},
            {"id":"three","full_correct":False}]
        after=[
            {"id":"one","full_correct":True},
            {"id":"two","full_correct":False},
            {"id":"three","full_correct":True}]
        original=guarded_old_ids(before)
        replacement=guarded_old_ids(after)
        self.assertEqual(len(original),len(replacement))
        self.assertFalse(original.issubset(replacement))

    def test_training_never_imports_unseen_QA_or_selection(self):
        source=Path(__file__).with_name("microguard_pcgrad_fp32.py").read_text()
        self.assertNotIn("from calibri2.s29.",source)
        self.assertNotIn("import calibri2.s29.",source)
        self.assertNotIn("from calibri2.s26.frozen_new_qa",source)
        self.assertIn("model.load_state_dict(saved_weights)",source)
        self.assertIn("opt.load_state_dict(saved_optimizer)",source)
        self.assertIn("guarded.issubset(retained)",source)

if __name__=="__main__":
    unittest.main()
