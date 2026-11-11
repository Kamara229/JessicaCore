"""Replay S143 gain classifier against audited real S142 artifact evidence.

Source: S142 run 43078895081, artifact 13885634834; the ZIP and
safetensors hashes were verified offline. This test reclassifies the
checked-in evidence only; it does not rerun CPU/FP16 inference or unseen QA.
"""
import json
from pathlib import Path
import unittest

from calibri2.s143.gain_reporting import classify_j7_development_gain


AUDIT_PATH = Path(__file__).with_name("S143_S142_ATTEMPT2_OFFLINE_AUDIT.json")


class S143RealS142ArtifactReporting(unittest.TestCase):
    def test_real_physical_j7_is_a_safe_historical_train_gain(self):
        audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(audit["source_workflow_run"], 43078895081)
        self.assertEqual(audit["source_artifact_id"], 13885634834)
        self.assertEqual(audit["final_fp16_bytes"], 1001348)
        self.assertEqual(
            audit["final_fp16_sha256"],
            "5a706d9012a61f24678929a2a6ee20f77c922e7e0f4bb516c061f32742cc56c7",
        )
        self.assertTrue(audit["safetensors_parse_verified"])
        self.assertTrue(audit["final_fp16_equals_guarded_j7_candidate"])
        self.assertTrue(audit["single_two_step_j7_committed"])
        self.assertTrue(audit["independent_456_128_replay_not_run"])
        self.assertTrue(audit["heldout_qa_not_run"])
        self.assertFalse(audit["model_accepted"])

        result = classify_j7_development_gain(
            committed=audit["single_two_step_j7_committed"],
            before_old_full=audit["previous_old456_full"],
            after_old_full=audit["final_old456_full"],
            before_new_full=audit["new_s122_full_before"],
            after_new_full=audit["new_s122_full_after"],
            before_new_pairs=0,
            after_new_pairs=0,
            lost_old_full=audit["lost_previously_correct_old_first_full"],
            lost_old_first=audit["lost_previously_correct_old_first_full"],
            lost_new_full=audit["lost_previously_correct_new_first_full"],
            lost_new_first=audit["lost_previously_correct_new_first_full"],
            fp16_reloaded=audit["safetensors_parse_verified"],
            immutable_source=True,
            physical_guard=audit["original_ci_report_attests_physical_456_128_cpu_replay"],
            j7_guard=audit["final_fp16_equals_guarded_j7_candidate"],
            physical_bpp=audit["physical_bpp"],
            cpu_ratio=audit["paired_cpu_ratio"],
        )
        self.assertEqual(result["historical_old456_full_delta"], 1)
        self.assertEqual(result["S122_new128_full_delta"], 0)
        self.assertEqual(result["S122_both_sides_pairs_delta"], 0)
        self.assertTrue(result["all_previously_correct_full_first_identities_preserved"])
        self.assertTrue(result["safe_development_TRAIN_improvement"])
        self.assertTrue(result["research_only_not_model_acceptance"])
        self.assertTrue(result["independent_unseen_QA_not_run"])


if __name__ == "__main__":
    unittest.main()
