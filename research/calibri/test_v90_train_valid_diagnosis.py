"""Small deterministic integrity tests; no weights, no external downloads."""
import unittest
from v90.diagnostics import paired_diagnosis,greedy_summary
from v90.provenance import check_slices,SPLITS,DIAGNOSTIC_PROMPTS
from v89.provenance import BOOL_Q,PROMPTS

def item(ix,gold,pred,margin):
    return {"record_index":ix,"gold":gold,"prompt_SHA":f"sha-{ix}",
            "pred":pred,"correct":int(gold==pred),
            "yes_no_logit_margin":margin,"prediction_confidence":0.7}

class TestV90(unittest.TestCase):
    def test_split_disjoint_and_outside_official_TEST(self):
        self.assertTrue(check_slices())
        self.assertTrue(all(0 <= a < b < 9428 for a,b in SPLITS.values()))
        self.assertTrue(set(DIAGNOSTIC_PROMPTS).isdisjoint(PROMPTS))
        self.assertEqual(BOOL_Q,(3210,3258))
        with self.assertRaises(AssertionError):
            check_slices({"TRAIN":(4,10),"TRAIN_VALID":(9,12)})
        with self.assertRaises(AssertionError):
            check_slices({"TRAIN":(1,3),"TEST":(4,9)})

    def test_paired_record_classification_and_gold_margin(self):
        base=[item(1,0,0,2),item(2,1,1,-2),item(3,0,1,-1),item(4,1,0,1)]
        corr=[item(1,0,1,-1),item(2,1,1,-3),item(3,0,0,2),item(4,1,0,2)]
        stats=paired_diagnosis(base,corr)
        self.assertEqual(stats["categories"],{"both_correct":1,"CB59_only_correct":1,
            "CB88_only_correct":1,"both_wrong":1})
        self.assertEqual([r["record_index"] for r in stats["regressed_records"]],[1])
        self.assertEqual([r["record_index"] for r in stats["recovered_records"]],[3])
        self.assertEqual(stats["CB59_correct"],2)
        self.assertEqual(stats["CB88_correct"],2)
        with self.assertRaises(AssertionError):paired_diagnosis(base,corr[::-1])

    def test_repetition_metric_reports_collapse_without_overclaiming(self):
        scores=greedy_summary([[1,1,1,1],[1,2,3,4]])
        self.assertAlmostEqual(scores["mean_distinct_ratio"],0.625)
        self.assertAlmostEqual(scores["mean_most_frequent_share"],0.625)
        with self.assertRaises(AssertionError):greedy_summary([[]])

if __name__=="__main__":unittest.main()
