"""Must preserve per-question factual answer grouping before optimizing critic."""
import unittest
from v124.protocol import TRAIN
from v124.hard_examples import examples
from v125.metrics import independent_metric
from v125.protocol import (GATE_TRUE_ACCEPT_MIN,GATE_PLAUSIBLE_REJECT_MIN,
                           GATE_REAL_REJECT_MIN)
class PairGuards(unittest.TestCase):
    def test_training_group_has_one_true_and_three_wrong(self):
        chosen=TRAIN[:3]
        data={host:{q:"completely wrong response" for q,ref in chosen}
              for host in ("CB59_v119","Pythia14M")}
        rows=examples(chosen,data)
        for i in range(0,len(rows),4):
            self.assertEqual(sum(r[2] for r in rows[i:i+4]),1)
            self.assertEqual([r[3] for r in rows[i:i+4]],
             ["independently_correct_reference","plausible_false_reference",
              "actual_generated_CB59_v119","actual_generated_Pythia14M"])
    def test_quality_gates_declared_independently_of_results(self):
        self.assertEqual((GATE_TRUE_ACCEPT_MIN,GATE_PLAUSIBLE_REJECT_MIN,
                          GATE_REAL_REJECT_MIN),(.7,.7,.7))
if __name__=="__main__":unittest.main()
