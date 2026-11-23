"""Z13 read-only original43M physical wrapper fixed source and no-train contract."""
import inspect
import unittest
from research.calibri_nra3 import s119q2i8z13_true43m_native8x2_parallel_own as runner
from research.calibri_nra3.s119q2i8z12_parallel_native_new14_source import verify

class TestZ13SourceOnlyPhysical(unittest.TestCase):
    def test_01_predeclared_source_not_previous_spent(self):
        v=verify()
        self.assertEqual(v["fresh_DEV_rows"],14)
        self.assertEqual(v["BRANCH_WIDTH"],8)
        self.assertEqual(v["THIRD_WIDTH"],2)
        self.assertEqual(v["HORIZON"],2)
        self.assertIn("8X2_LOOKAHEAD_FRESH14_OWN",runner.STAGE)
        self.assertEqual(runner.NATIVE_VOCAB,49152)
    def test_02_source_physical_provenance_required(self):
        sig=inspect.signature(runner.run)
        self.assertIn("z10_dir",sig.parameters)
        self.assertIn("s142",sig.parameters)
        self.assertIn("q2i8w_dir",sig.parameters)
        self.assertIn("out",sig.parameters)
    def test_03_no_training_forwards_during_archival_probe(self):
        source=inspect.getsource(runner.run)
        self.assertIn("native_lookahead_second_third(",source)
        self.assertIn("torch.no_grad()",source)
        self.assertIn("old_original_param_digest",source)
        self.assertIn("gold_used_for_posthoc_evaluation_NOT_candidate_selection",source)
        self.assertNotIn("train_joint_true43m_context_top8_ab(",source)
        self.assertNotIn("optimizer.step()",source)
        self.assertIn('"output_gate":0',source)
if __name__=="__main__":
    unittest.main()
