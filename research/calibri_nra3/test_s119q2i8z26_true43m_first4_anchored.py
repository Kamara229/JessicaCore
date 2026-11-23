"""Z26 real original43M native full49152 84 past TRAIN and NEW14 first4 auto BPE."""
import inspect
import unittest
from research.calibri_nra3.s119q2i8z25_84historicaltrain_new14dev_hardnegative_source import (
    verify,rows,TOP_K,RANK,TRAIN_EPOCHS,
)
from research.calibri_nra3.s119q2i8z25_native64_anchored_hardnegative_rank16 import (
    MODES,Native64AnchoredHardNegative,
)
from research.calibri_nra3 import s119q2i8z26_true43m_anchored_hardnegative_first4 as physical

class TestZ26RealOriginalHardnegativeFirst4(unittest.TestCase):
    def test_01_train_and_new_dev_are_separate_and_frozen(self):
        source=verify()
        self.assertEqual((source["historical_previously_consumed_TRAIN_count"],
                          source["brand_new_author_synthetic_DEV_count"]),(84,14))
        self.assertEqual(len(rows("TRAIN")),84)
        self.assertEqual(len(rows("DEV")),14)
        self.assertEqual((TOP_K,RANK,TRAIN_EPOCHS),(64,16,30))
        self.assertEqual(len(MODES),2)
        self.assertEqual(physical.NATIVE_VOCAB,49152)
        self.assertIn("z10_dir",inspect.signature(physical.run).parameters)
    def test_02_real_native43m_and_original_weights_check(self):
        c=inspect.getsource(physical.run)
        for needle in (
            "load_actual_original(s93,side,s142)",
            "real_joint_native(", "scoped_exact_native_gain(",
            "capture_existing_full43m_hidden(model)",
            "full.last_forward_coverage==tuple(range(12))",
            "old_original_param_digest(model)==original_digest",
            "native_embedding=torch.cat(native_pieces,dim=0)",
            "no_old_spent_DEV_labels_used_for_any_training",
        ):
            self.assertIn(needle,c)
    def test_03_training_occurs_before_new_DEV_and_no_teacher_forcing(self):
        c=inspect.getsource(physical.run)
        self.assertLess(c.index("for row in fresh_train:"),
                        c.index("train_matched_control_on_historical_train("))
        self.assertLess(c.index("train_matched_control_on_historical_train("),
                        c.index("for row in fresh_dev:"))
        self.assertIn('for mode in ("ORIGINAL_NATIVE_GREEDY",)+MODES:',c)
        self.assertIn("row[\"prefix\"]+continuation",c)
        self.assertIn("correct_prefix_contiguous_BPE",c)
        self.assertIn("autonomous_first_four_native_ids",c)
        self.assertIn("first_four_native_correct",c)
        self.assertIn("witnessed[0]==total_physical",c)
        self.assertIn('"candidate_accepted":False,"output_gate":0',c)
        self.assertNotIn("optimizer.step()",c)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",c)
        self.assertNotIn("target_next3_original_embeddings(",c)
if __name__=="__main__":
    unittest.main()
