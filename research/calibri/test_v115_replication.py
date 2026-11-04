"""Preregistered split checks and behavior-first aggregate tests."""
import unittest
from v115.protocol import SPLITS,SEEDS,partition
from v115.pipeline import summarize

class ReplicationGuards(unittest.TestCase):
    def test_split_integrity(self):
        self.assertEqual(SPLITS,("v115_a","v115_b"))
        self.assertEqual(SEEDS,(115,116,117))
        for salt in SPLITS:
            fit,valid=partition(salt)
            self.assertEqual(len(fit),28)
            self.assertEqual(len(valid),8)
            self.assertFalse(set(fit)&set(valid))
    def test_wrong_split_rejected(self):
        with self.assertRaises(ValueError):
            partition("official_TEST")
    def test_no_teacher_mimicry_in_aggregation(self):
        modes=("dense","shared","shared_adapter")
        evidence=[]
        for salt in SPLITS:
            for seed in SEEDS:
                for mode in modes:
                    evidence.append({
                        "split":salt,"seed":seed,"mode":mode,
                        "heldout":{"exact_correct":8},
                        "checkpoint_bytes":1000 if mode=="dense" else 500,
                        "cpu_examples_per_second":120. if mode=="dense" else 110.,
                    })
        result=summarize(evidence)
        self.assertEqual(len(result["paired"]),12)
        self.assertEqual(result["models"]["shared"]["correct_answers"],48)
        self.assertTrue(all(row["passes_90pct_baseline_and_candidate_and_cpu"] for row in result["paired"]))

if __name__=="__main__":
    unittest.main()
