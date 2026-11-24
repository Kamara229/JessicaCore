"""Z44 stdlib source-only tests: native rank bins, source pin, no DEV use."""
import unittest
from research.calibri_nra3.s119q2i8z44_pinned_train84_second_native_rank_readonly import (
    STAGE,SRC_SHA,stats,empirical,audit
)

class TestZ44NoNewModel(unittest.TestCase):
    def test_rank_bins_and_interquartile_counters(self):
        x=stats([1,2,8,32,64,65,100])
        self.assertEqual(x["n"],7)
        self.assertEqual(x["rank_median_lower"],32)
        self.assertEqual(x["rank_p75_nearest_rank"],65)
        self.assertEqual(x["topk_cumulative_counts"]["1"],1)
        self.assertEqual(x["topk_cumulative_counts"]["64"],5)
        self.assertEqual(x["beyond_original_TOP64"],2)
        with self.assertRaises(ValueError):stats([0,4])
    def test_empirical_diversity_not_conditional_lm_entropy(self):
        self.assertEqual(empirical([6,6,6,6])["distinct_IDs"],1)
        self.assertEqual(empirical([6,6,6,6])["empirical_cross_prompt_entropy_bits"],0.)
        self.assertEqual(empirical([1,2,3,4])["empirical_cross_prompt_entropy_bits"],2.)
    def test_wrong_report_cannot_count_as_legal_training(self):
        with self.assertRaises((ValueError,KeyError)):
            audit({"stage":"incorrect","candidate_accepted":False,"output_gate":0})
        with self.assertRaises(ValueError):
            audit({"stage":STAGE,"candidate_accepted":False,"output_gate":0,
                   "Z40_previous_legal_TRAIN84_SHA256":"bad",
                   "no_new_DEV_or_GOLD_evaluation":True,
                   "no_fitted_shadow_training_or_optimizer_steps":True,
                   "previous_Z40_DEV14_remains_SPENT_not_used":True,
                   "no_616J7_EVAL128_GOLD_bpp_CB59_CPU_certification":True,
                   "original_S142_43M_GT12_Core_NRA3_NPA1_and_Z10_A_B_unchanged":True,
                   "real_original43M_native_49152_decoder_forwards":332,
                   "first_native_exact_on_84_TRAIN":77,
                   "first_three_native_exact_on_84_TRAIN":0})
    def test_archived_physical_sha_contract(self):
        self.assertEqual(len(SRC_SHA),64)
        self.assertIn("LEGAL_TRAIN84",STAGE)
if __name__=="__main__":
    unittest.main()
