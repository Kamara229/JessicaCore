"""S202 pure guards; no neural execution or QA/TEST."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calibri2.s158 import targeted_retention_gpu as s158
from calibri2.s202.four_lost_real_cpu_j7_trace import (
    TARGET_IDS, LOSSES, diagnostic_row, posthoc_capture,
)


class TestS202FourLostJ7Trace(unittest.TestCase):
    def test_exact_immutable_train_ids(self):
        self.assertEqual(len(TARGET_IDS), 4)
        self.assertEqual(set(LOSSES), {2,4,8})
        self.assertEqual(set(LOSSES[2]), set(TARGET_IDS[-2:]))
        self.assertEqual(set(LOSSES[4]), set(LOSSES[8]))
        self.assertEqual(len(set(TARGET_IDS)), 4)

    def test_posthoc_divergence_location_without_route_change(self):
        key = TARGET_IDS[0]
        old = {key: {"question": "original TRAIN prompt",
                     "expected": [11,22,2]}}
        result = diagnostic_row(key, old, {key:[11,22,2]}, {key:[40,22,2]})
        self.assertTrue(result["parent_correct_full_EOS"])
        self.assertFalse(result["candidate_correct_full_EOS"])
        self.assertFalse(result["candidate_first_token_correct"])
        self.assertEqual(result["first_divergent_position"], 0)

    def test_capture_does_not_write_without_actual_full_j7(self):
        with tempfile.TemporaryDirectory() as tmp:
            hook,state = posthoc_capture(2,Path(tmp))
            old = {k:{"question":k,"expected":[1,2,3]} for k in TARGET_IDS}
            parent = {k:[1,2,3] for k in TARGET_IDS}
            candidate = {**parent}
            for k in LOSSES[2]: candidate[k]=[5,2,3]
            with patch.object(s158, "score_full_and_pairs", return_value={
                "counterfactual_pairs_both_correct": 0}):
                hook(parent,candidate,old,{}, {}, {})
            self.assertEqual(state["matched"],0)
            self.assertEqual(list(Path(tmp).iterdir()),[])


if __name__ == "__main__":
    unittest.main()
