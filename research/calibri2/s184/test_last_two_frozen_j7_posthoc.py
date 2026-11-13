"""S184 strict offline SHA-only tests (synthetic data, no neural model)."""
import copy
import unittest
from calibri2.s184.last_two_frozen_j7_posthoc import analyze


def fixture():
    cases = []
    gold = []
    for i in range(456):
        sha = f"{i:064x}"
        target = [10, 20, 0]
        generated = ([11, 20, 0] if i < 9 else
                     [10, 21, 0] if i < 27 else
                     [10, 20, 22] if i == 27 else target[:])
        cases.append({
            "prompt_sha256": sha, "full_generated_ids": generated,
            "route": "S93", "terminated_in_eos": generated[-1] == 0,
            "gold_seen_at_inference": False, "guard_events": [],
            "conditional_step2": None,
        })
        gold.append({"question_sha256": sha, "target_ids": target})
    trace = {"stage": "S179_TRACE_ONLY", "number": 456,
             "labels_used": False, "cases": cases}
    labels = {
        "stage": "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT",
        "all_456_per_identity_results": gold,
    }
    # 26 gains; two remaining = one first-token and one missing EOS.
    gained = [f"{i:064x}" for i in range(8)] + [
        f"{i:064x}" for i in range(9, 27)]
    s182 = {
        "event": "S182_REAL_CPU_J7_456_128_SHADOW_PROTECTED",
        "source_s179_frozen_baseline_run": 43323981995,
        "retention_passed": True, "model_accepted": False,
        "old_original_train_before": 428, "old_original_train_after": 454,
        "new_train128_actual_full_EOS_correct": 128,
        "old_gains": 26, "old_losses": 0,
        "first_losses": 0, "non_shadow_prediction_changes": 0,
        "real_CPU_J7_decoder_invoked": True,
        "old_gain_sha_only": gained, "old_loss_sha_only": [],
    }
    s183 = {
        "event": "S183_REAL_CPU_S176_VS_S182_PAIRED_SAME_CPU_SHADOW_J7",
        "parent_S182_run": 43331434088,
        "independent_speed_gate_pass": True,
        "model_accepted": False,
        "correct_456": 454, "correct_new128": 128,
        "old_losses": 0, "matched_combined_speed_ratio": 2.49,
        "physical_bpp_including_benchmark_code": 0.681,
    }
    return trace, labels, s182, s183


class S184PosthocContracts(unittest.TestCase):
    def test_two_unresolved_sha_only_no_raw_labels(self):
        r = analyze(*fixture())
        self.assertEqual(r["remaining_incorrect"], 2)
        self.assertEqual(r["remaining_failure_families"],
                         {"EOS-or-length": 1, "first-token": 1})
        self.assertEqual({x["prompt_sha256"] for x in r["remaining_cases_SHA_only"]},
                         {f"{8:064x}", f"{27:064x}"})
        self.assertNotIn("target_ids", str(r))
        self.assertNotIn("full_generated_ids", str(r))
        self.assertTrue(r["no_model_execution_or_training"])

    def test_gains_must_be_originally_wrong(self):
        a = list(fixture())
        a[2]["old_gain_sha_only"][0] = f"{455:064x}"
        with self.assertRaisesRegex(AssertionError, "gains not misses"):
            analyze(*a)

    def test_speed_gate_fails_closed(self):
        a = list(fixture())
        a[3]["independent_speed_gate_pass"] = False
        with self.assertRaisesRegex(AssertionError, "S183 independent"):
            analyze(*a)

    def test_original_trace_leak_rejected(self):
        a = list(fixture())
        a[0]["cases"][0]["expected"] = [10, 20, 0]
        with self.assertRaisesRegex(AssertionError, "leakage"):
            analyze(*a)

    def test_requires_26_distinct_gains(self):
        a = list(fixture())
        a[2]["old_gain_sha_only"][0] = a[2]["old_gain_sha_only"][1]
        with self.assertRaisesRegex(AssertionError, "provenance"):
            analyze(*a)


if __name__ == "__main__":
    unittest.main()
