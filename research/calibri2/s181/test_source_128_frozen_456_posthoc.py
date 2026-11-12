"""Synthetic 456 source-join safety tests; no model or tokenization."""
import unittest
from pathlib import Path

from calibri2.s181.source_128_frozen_456_posthoc import (
    offline_posthoc, reconstruct_sources, sha)


def frozen_fixture():
    root = Path(__file__).resolve().parents[3]
    generated = reconstruct_sources(root)
    cases = []
    labels = []
    for i, (key, prompt, _gold) in enumerate(generated):
        h = sha(prompt)
        target = [10, 20, 0]
        is_wrong = i < 20
        cases.append({
            "prompt_sha256": h,
            "full_generated_ids": [10, 21, 0] if is_wrong else target,
            "gold_seen_at_inference": False,
        })
        labels.append({
            "id": key, "question_sha256": h, "target_ids": target,
        })
    for i in range(328):
        h = sha(f"NON_SOURCE_PLACEHOLDER_{i}")
        is_wrong = i < 8
        cases.append({
            "prompt_sha256": h,
            "full_generated_ids": [10, 21, 0] if is_wrong else [10, 20, 0],
            "gold_seen_at_inference": False,
        })
        labels.append({
            "id": f"OTHER/{i}", "question_sha256": h, "target_ids": [10, 20, 0],
        })
    return root, {
        "stage": "S179_TRACE_ONLY", "number": 456,
        "labels_used": False, "cases": cases,
    }, {
        "stage": "S87_456_TRAIN_SERIALIZED_COMPRESSED_CPU_AUDIT",
        "all_456_per_identity_results": labels,
    }


class S181OfflineChecks(unittest.TestCase):
    def test_source_prompt_sha_join_and_456_baseline(self):
        root, trace, gold = frozen_fixture()
        result = offline_posthoc(root, trace, gold)
        self.assertEqual(result["all_frozen_train_outputs_checked"], 456)
        self.assertEqual(result["frozen_full_eos_correct"], 428)
        self.assertEqual(result["shadow_source_regenerated_and_SHA_verified"], 128)
        self.assertEqual(result["source_metrics"]["source_S30"], 64)
        self.assertEqual(result["source_metrics"]["source_S54"], 64)
        self.assertEqual(result["source_metrics"]["lexical_source_could_recover_if_integrated"], 20)
        self.assertEqual(result["other_source_prompt_texts_not_generated"], 328)
        self.assertIsNone(result["genuine_integrated_gain"])
        self.assertIsNone(result["genuine_integrated_forgetting"])
        self.assertFalse(result["model_accepted"])

    def test_hash_drift_rejected(self):
        root, trace, gold = frozen_fixture()
        gold["all_456_per_identity_results"][0]["question_sha256"] = "f" * 64
        with self.assertRaisesRegex(AssertionError, "SHA drift"):
            offline_posthoc(root, trace, gold)

    def test_frozen_correctness_drift_rejected(self):
        root, trace, gold = frozen_fixture()
        trace["cases"][-1]["full_generated_ids"] = [10, 21, 0]
        with self.assertRaisesRegex(AssertionError, "428/456"):
            offline_posthoc(root, trace, gold)

if __name__ == "__main__":
    unittest.main()
