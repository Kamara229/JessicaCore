"""S119q2d1 read-only original native full-vocab baseline comparison controls."""
from __future__ import annotations

import copy
import unittest
from collections import Counter
from pathlib import Path

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2d1_real_original_sameprobe_baseline import (
    summarize_original,load_q2d_proof,PROBE_COUNTS,STAGE,
)
from research.calibri_nra3.s119q2a_train_source_manifest import build_rows
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    source_inventory,
)


def fake_source_records():
    a,b=source_inventory(build_rows())
    return [
        {
            "kind":x["kind"],
            "source_split":x["source_split"],
            "target_position":0,
            "next_native_CE":12. if x["source_family"]=="q2a" else 10.,
            "correct_native_greedy_token":False,
            "source_id":x["id"],
        }
        for x in b
    ]


class S119Q2D1FrozenOriginalNativeControls(unittest.TestCase):
    def test_01_exact_30_known_source_probes_only(self):
        rows=fake_source_records()
        self.assertEqual(len(rows),30)
        self.assertEqual(Counter(r["source_split"] for r in rows),PROBE_COUNTS)
        self.assertEqual(Counter(r["kind"] for r in rows),
                         {k:5 for k in ("roles_word_order","negation",
                                        "transitive_comparison","event_time",
                                        "arithmetic","paraphrase")})
        r=summarize_original(rows)
        self.assertEqual(r["count_source_probes"],30)
        self.assertEqual(r["source_only_greedy_correct"],0)
        self.assertTrue(r["not_blind_independent_or_full_answer_generation"])

    def test_02_mean_is_actual_arithmetic_not_magic_baseline(self):
        rows=fake_source_records()
        out=summarize_original(rows)
        self.assertAlmostEqual(out["mean_source_only_native_CE"],
                               (18*12+12*10)/30)
        self.assertEqual(out["mean_source_only_by_domain"],{
            k:11.2 for k in (
                "roles_word_order","negation","transitive_comparison",
                "event_time","arithmetic","paraphrase"
            )
        })

    def test_03_fails_wrong_probe_counts(self):
        bad=fake_source_records()[:-1]
        with self.assertRaisesRegex(GateFailure,"(?i)only 18 CAL"):
            summarize_original(bad)
        bad=fake_source_records()
        bad[0]["source_split"]="TRAIN"
        with self.assertRaisesRegex(GateFailure,"(?i)only 18 CAL"):
            summarize_original(bad)

    def test_04_fails_missing_domain_even_with_30_rows(self):
        bad=fake_source_records()
        first=bad[0]["kind"]
        for x in bad:
            if x["kind"]==first:
                x["kind"]="arithmetic"
        with self.assertRaisesRegex(GateFailure,"six domains"):
            summarize_original(bad)

    def test_05_source_never_included_as_new_TRAIN(self):
        tr,pr=source_inventory(build_rows())
        self.assertEqual(len(tr),96)
        self.assertEqual(len(pr),30)
        self.assertFalse({x["prompt"] for x in tr}&{x["prompt"] for x in pr})
        self.assertTrue(all(x["source_split"]=="TRAIN" for x in tr))

    def test_06_parent_artifact_cannot_be_unverified_filename(self):
        with self.assertRaisesRegex(GateFailure,"fixed genuine"):
            load_q2d_proof(
                Path("wrong.json"),Path("wrongA.safetensors"),
                Path("wrongB.safetensors"),Path("s93"),Path("s142"))

    def test_07_stage_is_an_audit_not_train(self):
        self.assertIn("ORIGINAL_SOURCE_PROBE_BASELINE",STAGE)
        self.assertNotIn("TRAIN_96",STAGE)


if __name__=="__main__":
    unittest.main()
