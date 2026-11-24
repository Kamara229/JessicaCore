"""Z48 protected spent DEV: stdlib-only test, never re-score original model."""
import unittest
from pathlib import Path
from research.calibri_nra3 import s119q2i8z48_pinned_z47_spent28_collapse_forensics as z

SOURCE=Path(z.__file__).read_text(encoding="utf8")

class TestZ48SourcePinnedNoDevReuse(unittest.TestCase):
    def test_entropy_noncollapse_and_majority_are_distinct_metrics(self):
        collapsed=z.diversity([12]*28)
        self.assertEqual(collapsed["distinct_IDs"],1)
        self.assertEqual(collapsed["empirical_crossprompt_entropy_bits"],0.)
        self.assertEqual(collapsed["majority_fraction"],1.)
        balanced=z.diversity(list(range(28)))
        self.assertEqual(balanced["distinct_IDs"],28)
        self.assertAlmostEqual(balanced["empirical_crossprompt_entropy_bits"],4.807355,places=6)
        self.assertEqual(balanced["majority_count"] if "majority_count" in balanced else
                         balanced["majority_ID_count"],1)
    def test_low_diversity_but_more_unique_than_one(self):
        z1=z.diversity([4]*20+[7]*8)
        self.assertEqual(z1["distinct_IDs"],2)
        self.assertGreater(z1["empirical_crossprompt_entropy_bits"],0)
        self.assertLess(z1["empirical_crossprompt_entropy_bits"],1)
        self.assertEqual(z1["top_native_ID_counts"][0]["count"],20)
    def test_untrusted_and_unpinned_reports_fail_closed(self):
        with self.assertRaises((KeyError,ValueError)):z.summarize({})
        with self.assertRaises((KeyError,ValueError)):
            z.summarize({"stage":"not-actual-Z47", "candidate_accepted":False})
        with self.assertRaises(ValueError):z.diversity([49152])
        with self.assertRaises(ValueError):z.diversity([-1])
        with self.assertRaises(ValueError):z.diversity([])
    def test_fixed_28_precommit_and_no_python_torch_import(self):
        self.assertEqual(len(z.KINDS),7)
        self.assertEqual(len(z.MODES),3)
        self.assertEqual(len(z.SOURCE_SHA),64)
        self.assertNotIn("import torch",SOURCE)
        self.assertNotIn("optimizer.step(",SOURCE)
        self.assertNotIn("real_joint_native(",SOURCE)
        self.assertIn("new_original43M_native_forwards",SOURCE)
        self.assertIn("DEV28_considered_SPENT_never_fit_or_rescore",SOURCE)
        self.assertIn("paired_original_HUMAN_first2_changes",SOURCE)
        self.assertIn("paired_original_HUMAN_premature_EOS_changes",SOURCE)
    def test_original_S142_43M_source_physical_cost_is_only_prior_artifact(self):
        self.assertEqual(z.Z47_STAGE,
            "S119Q2I8Z47_PHYSICAL_REAL43M_FRESH28_DEV_BPE2_ONPOLICY_AND_CROSSPROMPT_DIVERSITY")
        self.assertTrue(z.STAGE.startswith("S119Q2I8Z48_READONLY"))
if __name__=="__main__":
    unittest.main()
