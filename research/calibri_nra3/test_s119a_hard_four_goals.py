"""S119a CPU-only synthetic CONTRACT tests. No Calibri, GOLD or user data."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from research.calibri_nra3.s119a_hard_four_goals import (
    GateFailure, BPP_MAX, CPU_MIN, PROTECTED_J7, packed_budget_bytes,
    physical_bundle, exact_protected_j7, matched_cpu_speed,
    independent_relation_gain, candidate_acceptance,
)


def digest(x):
    return hashlib.sha256(x).hexdigest()


class S119HardGates(unittest.TestCase):
    def test_01_denominator_cannot_be_inflated(self):
        self.assertGreater(packed_budget_bytes(), 0)
        with self.assertRaises(GateFailure):
            packed_budget_bytes(43_261_760)

    def test_02_real_files_sha_counted_and_metadata_charged(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            files = {"core.cb61": b"real packed core bytes",
                     "gt.fp16": b"green tunnel bytes",
                     "nra3.fp16": b"neural child bytes",
                     "runtime_manifest.json": b"include me in budget"}
            for name, raw in files.items():
                (p / name).write_bytes(raw)
            inv = {name: digest(raw) for name, raw in files.items()}
            res = physical_bundle(p, inv, prospective_extra_bytes=1000)
            self.assertEqual(res["physical_bytes"], sum(map(len, files.values())))
            self.assertLess(res["projected_all_in_bpp"], BPP_MAX)
            with self.assertRaises(GateFailure):
                physical_bundle(p, inv, prospective_extra_bytes=packed_budget_bytes())

    def test_03_sha_mismatch_and_path_escape_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "model.bin").write_bytes(b"model")
            with self.assertRaises(GateFailure):
                physical_bundle(p, {"model.bin": "0"*64})
            with self.assertRaises(GateFailure):
                physical_bundle(p, {"../model.bin": digest(b"model")})

    def test_04_exact_protected_616_j7(self):
        baseline = {f"case-{j}": ("a"*64, (j + 3, 2))
                    for j in range(PROTECTED_J7)}
        candidate = baseline.copy()
        self.assertEqual(exact_protected_j7(baseline, candidate)["losses"], 0)
        candidate["case-345"] = ("a"*64, (12, 2))
        with self.assertRaises(GateFailure):
            exact_protected_j7(baseline, candidate)

    def test_05_any_missing_protected_J7_is_hard_fail(self):
        b = {f"case-{j}": ("a"*64, (j, 2))
             for j in range(PROTECTED_J7)}
        n = dict(b)
        n.pop("case-8")
        with self.assertRaises(GateFailure):
            exact_protected_j7(b, n)

    @staticmethod
    def paired(candidate_s):
        return [
            {"prompt_sha256": f"{i:064x}", "environment_sha256": "e"*64,
             "baseline_s": .100, "candidate_s": candidate_s,
             "baseline_token_ids": [i, 2], "candidate_token_ids": [i, 2]}
            for i in range(8) for _ in range(3)
        ]

    def test_06_cpu_tolerate_10_to_15_pct_not_beyond(self):
        ok = matched_cpu_speed(self.paired(.115))
        self.assertGreaterEqual(ok["paired_full_decode_speed_ratio"], CPU_MIN)
        self.assertFalse(ok["target_met"])
        with self.assertRaises(GateFailure):
            matched_cpu_speed(self.paired(.125))

    def test_07_cpu_environment_and_workload_must_match(self):
        items = self.paired(.108)
        items[0]["candidate_token_ids"] = [9, 2]
        with self.assertRaises(GateFailure):
            matched_cpu_speed(items)
        items = self.paired(.108)
        items[1]["environment_sha256"] = "f"*64
        with self.assertRaises(GateFailure):
            matched_cpu_speed(items)

    def test_08_unseen_sentence_paired_quality_positive(self):
        rows = [(2, 0, 1) for _ in range(128)]
        res = independent_relation_gain(rows)
        self.assertGreater(res["paired_sentence_bootstrap_95pct_pp"][0], 0)

    def test_09_no_independent_gain_rejected(self):
        rows = [(2, 1, 1) for _ in range(128)]
        with self.assertRaises(GateFailure):
            independent_relation_gain(rows)

    def test_10_no_synthetic_test_can_accept_model(self):
        with self.assertRaises(GateFailure):
            candidate_acceptance()


if __name__ == "__main__":
    unittest.main()
