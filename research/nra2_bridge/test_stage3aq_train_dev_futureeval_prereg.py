"""S3AQ prereg data-only test; never imports actual CPU model or adapter weights."""
import unittest
from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    TRAIN,DEV,EVAL,STYLES,REVERSED_STYLES,build_rows,validate,commitment,old_rows,
)

class S3AQDatasetPrereg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()
        cls.commit=commitment(cls.rows)

    def test_frozen_512_split_disjointness(self):
        rows=self.rows
        self.assertTrue(validate(rows))
        self.assertEqual(len(rows),512)
        self.assertEqual(len({r.prompt_sha256 for r in rows}),512)
        self.assertEqual(len({r.topology_key for r in rows}),128)
        self.assertTrue({r.prompt_sha256 for r in rows}.isdisjoint({r.prompt_sha256 for r in old_rows()}))

    def test_balanced_role_order_and_labels_all_splits(self):
        for split,count,graphs in ((TRAIN,256,64),(DEV,128,32),(EVAL,128,32)):
            rr=[r for r in self.rows if r.split==split]
            self.assertEqual(len(rr),count)
            self.assertEqual(len({r.topology_key for r in rr}),graphs)
            self.assertEqual(sum(r.reachable for r in rr),count//2)
            self.assertEqual(sum(r.query_style in REVERSED_STYLES for r in rr),count//2)
            for style in range(4):
                sr=[r for r in rr if r.query_style==style]
                self.assertEqual(len(sr),count//4)
                self.assertEqual(sum(r.reachable for r in sr),count//8)

    def test_grammar_and_output_gate_untouched(self):
        rr=[g for v in STYLES.values() for g in v]
        self.assertEqual(len(rr),12)
        self.assertEqual(len(set(rr)),12)
        info=self.commit
        self.assertEqual(info["stage"],"NRA2_S3AQ_PRECOMMIT_TRAIN256_DEV128_FUTURE_EVAL128_NO_SCORE")
        self.assertEqual(info["original_S3AO_successful_run"],43579515862)
        self.assertTrue(info["new_eval_excluded_from_optimizer_and_checkpoint_selection"])
        self.assertTrue(info["future_eval_not_to_be_prefilled_before_candidate_physically_locked"])
        self.assertTrue(info["no_weights_or_training_or_scores"])
        self.assertEqual(info["gate"],0.0)
        self.assertEqual(len(info["commitment_sha256"]),64)

if __name__=="__main__":
    unittest.main()
