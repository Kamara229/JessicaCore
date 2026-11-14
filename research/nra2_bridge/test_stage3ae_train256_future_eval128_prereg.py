"""S3AE new TRAIN256/EVAL128 prereg integrity only; no model inference."""
import unittest
from collections import Counter
from research.nra2_bridge.stage3ae_train256_future_eval128_prereg import (
    TRAIN,EVAL,REVERSED_STYLES,STYLES,build_rows,validate,commitment,old_rows
)

class Stage3AEPreregTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=build_rows()

    def test_independent_topology_and_prompt_sha(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual(Counter(r.split for r in self.rows),{TRAIN:256,EVAL:128})
        train=[r for r in self.rows if r.split==TRAIN]
        evaluation=[r for r in self.rows if r.split==EVAL]
        prior=old_rows()
        t={r.topology_key for r in train}
        e={r.topology_key for r in evaluation}
        p={tuple(sorted(r.edges)) for r in prior}
        self.assertEqual(len(t),64)
        self.assertEqual(len(e),32)
        self.assertFalse(t&e)
        self.assertFalse((t|e)&p)
        self.assertFalse({r.prompt_sha256 for r in self.rows}&{r.prompt_sha256 for r in prior})

    def test_both_splits_balanced_and_unique_wordings(self):
        self.assertEqual(len({s for v in STYLES.values() for s in v}),8)
        for split,count in ((TRAIN,256),(EVAL,128)):
            x=[r for r in self.rows if r.split==split]
            self.assertEqual(sum(r.reachable for r in x),count//2)
            for reverse in (True,False):
                y=[r for r in x if (r.query_style in REVERSED_STYLES)==reverse]
                self.assertEqual(len(y),count//2)
                self.assertEqual(sum(r.reachable for r in y),count//4)

    def test_deterministic_precommit_without_model_scoring(self):
        d=commitment(self.rows)
        self.assertEqual(d,commitment(build_rows()))
        self.assertTrue(d["no_weights_or_training_or_scores"])
        self.assertTrue(d["new_eval_excluded_from_optimizer_and_checkpoint_selection"])
        self.assertTrue(d["previous_S3AA_is_already_scored_and_NOT_future_blind"])
        self.assertEqual(d["gate"],0.)

if __name__=="__main__":
    unittest.main()
