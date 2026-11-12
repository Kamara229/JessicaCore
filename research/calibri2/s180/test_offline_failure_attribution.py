"""S180 offline-only contract; synthetic rows, no pretrained model or Jessica."""
import copy
import unittest

from calibri2.s180.offline_failure_attribution import analyze


def fixture():
    gold = []
    cases = []
    for i in range(456):
        sha = f"{i:064x}"
        ids = [10, 20, 0]
        generated = [10, 21, 0] if i < 28 else ids[:]
        gold.append({"id": f"train_{i}", "question_sha256": sha,
                     "target_ids": ids})
        cases.append({
            "prompt_sha256": sha, "full_generated_ids": generated,
            "route": "S93", "terminated_in_eos": True,
            "guard_events": [], "conditional_step2": {
                "score_type": "actual_ar_decoder_conditional_step2_logits",
                "top5_token_ids_and_logits": [[21, 4.0], [20, 3.0], [0, -1.0]],
                "eos_rank": 3,
            },
            "gold_seen_at_inference": False,
        })
    return ({"stage": "S179_TRACE_ONLY", "number": 456,
             "labels_used": False, "cases": cases},
            {"stage": "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT",
             "all_456_per_identity_results": gold})


class S180OfflineContracts(unittest.TestCase):
    def test_posthoc_28_sha_only_and_second_token_attribution(self):
        trace, report = fixture()
        r = analyze(trace, report)
        self.assertEqual(r["full_eos_correct"], 428)
        self.assertEqual(r["incorrect_count"], 28)
        self.assertEqual(r["failure_flags_overlapping"]["second_wrong_after_first_right"], 28)
        self.assertEqual(r["observed_second_step_top5_coverage"]["gold_second_in_top5"], 28)
        self.assertFalse(r["gold_used_at_inference"])
        self.assertTrue(r["no_model_training_or_selection"])
        self.assertNotIn("target_ids", str(r))
        self.assertNotIn("full_generated_ids", str(r))

    def test_label_leakage_fails_closed(self):
        t, r = fixture()
        t["cases"][0]["expected"] = [10, 20, 0]
        with self.assertRaisesRegex(AssertionError, "leakage"):
            analyze(t, r)

    def test_mismatched_cohort_fails_closed(self):
        t, r = fixture()
        r["all_456_per_identity_results"][0]["question_sha256"] = "f" * 64
        with self.assertRaisesRegex(AssertionError, "cohort mismatch"):
            analyze(t, r)

    def test_s176_428_exact_fails_closed(self):
        t, r = fixture()
        t["cases"][28]["full_generated_ids"] = [10, 21, 0]
        with self.assertRaisesRegex(AssertionError, "source drift"):
            analyze(t, r)

    def test_missing_logits_never_forged(self):
        t, r = fixture()
        for i in range(28):
            t["cases"][i]["conditional_step2"] = None
        d = analyze(t, r)
        self.assertEqual(d["observed_second_step_top5_coverage"]["no_conditional_step2"], 28)

    def test_eos_integrity_fails_closed(self):
        t, r = fixture()
        t["cases"][0]["terminated_in_eos"] = False
        with self.assertRaisesRegex(AssertionError, "EOS trace integrity"):
            analyze(t, r)


if __name__ == "__main__":
    unittest.main()
