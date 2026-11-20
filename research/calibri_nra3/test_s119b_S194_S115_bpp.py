"""S119b synthetic-only source/budget contracts; authentic archive tested by CI separately."""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119b_S194_parent_S115_child_physical_budget import (
    S194_REFERENCE_DENOM, FULL_NRA3_N, S115_EXPECTED_SHA256,
    parse_safetensors, historical_projection, audit,
)


def synthetic_safetensors():
    sizes = [10, 12, 11, 13, 17, 20, 53751-8-83, 8]
    assert sum(sizes)==FULL_NRA3_N
    keys=[f"model.nra3_neural_weights.tensor_{i}" for i in range(7)] + [
        "model.nra3_neural_weights.structural_trust_gate"]
    entries={}
    current=0
    for k,n in zip(keys,sizes):
        entries[k]={"dtype":"F16","shape":[n],
                    "data_offsets":[current,current+2*n]}
        current+=2*n
    hdr=json.dumps(entries).encode("utf8")
    return struct.pack("<Q",len(hdr))+hdr+b"\0"*current


class S119BProof(unittest.TestCase):
    def test_real_fp16_geometry_contract(self):
        r=parse_safetensors(synthetic_safetensors())
        self.assertEqual(r["neural_fp16_parameters"], 53751)
        self.assertEqual(r["additional_trust_weights"], 8)

    def test_corruption_truncation_rejected(self):
        raw=synthetic_safetensors()
        with self.assertRaises(GateFailure):
            parse_safetensors(raw[:-2])

    def test_no_denominator_inflation_from_43M(self):
        r=historical_projection(parent_bpp=0.65,physical_child_bytes=110000)
        self.assertEqual(r["s194_reference_denominator"], 43261760)
        self.assertGreater(r["projected_conservative_bpp"],r["projected_legacy_bpp"])

    def test_conservative_overspend_never_accepted(self):
        r=historical_projection(parent_bpp=0.65,physical_child_bytes=110000)
        self.assertFalse(r["candidate_passes_conservative_rate_projection"])
        self.assertFalse(r["candidate_accepted"])

    def test_hard_bpp_parent_rejected(self):
        with self.assertRaises(GateFailure):
            historical_projection(parent_bpp=0.700001,physical_child_bytes=111)
        with self.assertRaises(GateFailure):
            historical_projection(parent_bpp=.65,physical_child_bytes=-5)

    def test_physical_unknown_files_are_not_completeness_proof(self):
        r=historical_projection(parent_bpp=.55,physical_child_bytes=110000)
        self.assertTrue(r["candidate_passes_conservative_rate_projection"])
        self.assertFalse(r["S119b_physically_inventory_complete"])
        self.assertFalse(r["S119b_candidate_J7_full616_replayed"])
        self.assertFalse(r["S119b_real_CPU_speed_measured"])
        self.assertFalse(r["S119b_independent_quality_gain_scored"])

    def test_no_fabricated_s194_or_s115_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            s194=root/"S194_ACTUAL_REAL_CPU_J7_584_PLUS32_GATE.json"
            child=root/"NRA3_S115_SYNTAGRUS_NC_NEW8_BOUNDED_TRUST_53751_IN_CALIBRI_FP16.safetensors"
            meta=root/"NRA3_S115_SYNTAGRUS_NC_NEW8_TRAIN_ONLY_NO_DEV_REPORT.json"
            s194.write_text('{"stage":"not a genuine J7"}')
            child.write_bytes(synthetic_safetensors())
            meta.write_text("{}")
            with self.assertRaises(GateFailure):
                audit(s194,child,meta)


if __name__ == "__main__":
    unittest.main()
