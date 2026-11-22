"""S119q2i8l unit-only fail-closed native paired outcome audit."""
import unittest
from research.calibri_nra3.s119q2i8l_paired_native_postrun_audit import (
    analyse,wilson_95,paired_ab,paired_native_sequences,score_records,OBS,
)

def one(*,full=False,valid=True):
    return {
        "true_native_full_EOS_correct":full,
        "valid_full_UTF8_bytes":valid,
        "native_EOS":full,
        "first_native_correct_but_wrong_second":not full,
        "first_gold_BPE_correct":True,
        "text_exact_even_if_not_finished":full,
        "real_EOS_but_UNFINISHED_or_INVALID_UTF8":False,
        "original_full_decoder_calls":3,
    }

class TestPairedNativeAudit(unittest.TestCase):
    def test_01_wilson_95_small_n_correct_extremes(self):
        self.assertEqual(wilson_95(0,14)[0],0)
        self.assertEqual(wilson_95(14,14)[1],1)
        self.assertGreater(wilson_95(0,14)[1],0.19)
        self.assertLess(wilson_95(14,14)[0],0.81)
    def test_02_failclosed_unverified_report(self):
        with self.assertRaises(ValueError):
            analyse({})
        with self.assertRaises(ValueError):
            analyse({"stage":"evil","output_gate":1})
    def test_03_count_per_native_event_from_individual_records(self):
        s=score_records([one(full=True),one(full=False)],2)
        self.assertEqual(s["full_correct"],1)
        self.assertEqual(s["valid_utf8"],2)
        self.assertEqual(s["real_native_forward_calls"],6)
    def test_04_paired_win_loss_does_not_assume_significance(self):
        a=[one(full=True),one(full=False)]
        b=[one(full=False),one(full=True)]
        d=paired_ab(a,b,OBS["full_correct"])
        self.assertEqual(d["CE_only_pass_KL_fail"],1)
        self.assertEqual(d["CE_only_fail_KL_pass"],1)
        self.assertTrue(d["no_hypothesis_or_promotion_from_tiny_visible_internal_DEV"])
    def test_05_native_identical_outputs_distinguished_from_same_aggregate(self):
        a=[{**one(full=False),"source_id":"example-a",
            "generated_own_native_BPE_ids":[155,202,2]}]
        b=[{**one(full=False),"source_id":"example-a",
            "generated_own_native_BPE_ids":[155,203,2]}]
        diff=paired_native_sequences(a,b)
        self.assertEqual(diff["identical_greedy_native_BPE_sequences"],0)
        self.assertEqual(diff["different_greedy_native_BPE_sequences"],1)
        self.assertEqual(diff["first_pairwise_disagreement_positions_1based"],[2])
        self.assertEqual(diff["different_first_generated_token"],0)
    def test_06_identical_full_greedy_trace(self):
        a=[{**one(full=False),"source_id":"example-a",
            "generated_own_native_BPE_ids":[155,202,2]}]
        same=paired_native_sequences(a,a)
        self.assertEqual(same["identical_greedy_native_BPE_sequences"],1)
        self.assertEqual(same["different_greedy_native_BPE_sequences"],0)
    def test_05_invalid_boolean_cannot_be_reported_as_measured(self):
        broken=one()
        broken["native_EOS"]="False"
        with self.assertRaises(ValueError):
            score_records([broken],1)
if __name__=="__main__":
    unittest.main()
