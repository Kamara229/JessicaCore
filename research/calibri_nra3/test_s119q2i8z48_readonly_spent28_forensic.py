"""Z48 stdlib-only read-only report data invariants; zero S142 loads."""
import unittest
from pathlib import Path
from research.calibri_nra3 import s119q2i8z48_readonly_z47_spent28_diversity_forensic as z48

class TestZ48ReadOnly(unittest.TestCase):
    def test_crossprompt_entropy_not_conditional_entropy(self):
        a=z48.empirical([4,4,4,4])
        self.assertEqual(a["distinct_native_IDs"],1)
        self.assertEqual(a["crossprompt_empirical_entropy_bits"],0.)
        self.assertEqual(a["most_common_native_ID_frequency"],4)
        self.assertEqual(a["second_BPE_Herfindahl"],1.)
        b=z48.empirical([1,2,3,4])
        self.assertEqual(b["distinct_native_IDs"],4)
        self.assertEqual(b["crossprompt_empirical_entropy_bits"],2.)
        self.assertEqual(b["most_common_native_ID_share"],.25)
        self.assertEqual(b["top3_native_IDs_total"],3)
    def test_invalid_original_token_ids_rejected(self):
        for x in ([49152],[-1],[],[1.5],[True]):
            with self.assertRaises(ValueError):z48.empirical(x)
    def test_only_original_one_prior_spent_report_can_be_opened(self):
        with self.assertRaises((ValueError,KeyError)):
            z48.audit({"stage":"fake","candidate_accepted":False,"output_gate":0})
        with self.assertRaises((ValueError,KeyError)):
            z48.audit({"stage":z48.Z47_STAGE,
                       "Z46_NEW28_before_physical_SHA256":"not correct",
                       "candidate_accepted":False,"output_gate":0})
    def test_source_is_readonly_without_model_or_mutations(self):
        source=Path(z48.__file__).read_text(encoding="utf8")
        self.assertNotIn("import torch",source)
        self.assertNotIn("AutoTokenizer",source)
        self.assertNotIn("optimizer.step()",source)
        self.assertNotIn("real_joint_native(",source)
        self.assertNotIn("load_actual_original(",source)
        self.assertIn("no_DEV_remediation_or_hyperparameter_search",source)
        self.assertIn("candidate_accepted\":False,\"output_gate\":0",source)
        self.assertEqual(len(z48.MODES),3)
        self.assertEqual(len(z48.KINDS),7)
if __name__=="__main__":
    unittest.main()
