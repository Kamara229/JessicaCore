"""S3BY: validate immutable TRAIN-only loader boundary, including negative cases."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import unittest
from research.nra2_bridge.stage3by_fit_only_loader_gate import FitOnlyGate, S3BX_COMMITMENT_SHA
from research.nra2_bridge.stage3bx_blocked_family_topology_protocol import (
    split_data, protocol, PRE_S3BR_PARENT_SHA, REJECTED_S3BT_SHA,
)


class S3BYFitOnlyGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.groups=split_data()

    def setUp(self):
        self.gate=FitOnlyGate()

    def test_pinned_fold_sha_and_counts(self):
        self.assertEqual(protocol(self.groups)["commitment_sha256"],S3BX_COMMITMENT_SHA)
        self.assertEqual(len(self.gate.allowed),192)
        self.assertEqual(len(self.gate.denied),320)
        self.assertFalse(self.gate.allowed & self.gate.denied)

    def test_requires_physically_verified_parent(self):
        with self.assertRaises(AssertionError):
            self.gate.minibatch(self.groups["fit"][:4])
        with self.assertRaises(AssertionError):
            self.gate.verify_parent(Path("/__nonexistent__/NRA2_S3BN_FP16.safetensors"))
        self.assertEqual(len(PRE_S3BR_PARENT_SHA),64)
        self.assertNotEqual(PRE_S3BR_PARENT_SHA,REJECTED_S3BT_SHA)

    def test_corrupt_physical_parent_fails_closed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            fake = Path(directory) / "S3BN_FP16.safetensors"
            fake.write_bytes(b"not the sealed S3BN physical checkpoint")
            with self.assertRaises(AssertionError):
                self.gate.verify_parent(fake)
            self.assertFalse(self.gate.parent_verified)
            with self.assertRaises(AssertionError):
                self.gate.minibatch(self.groups["fit"][:1])

    def test_exact_fit_only_rows(self):
        # Parent attestation is separately tested against the real artifact in CI.
        self.gate.parent_verified=True
        self.assertEqual(len(self.gate.minibatch(self.groups["fit"][:4])),4)
        for name in ("grammar_guard","topology_guard","double_guard"):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.gate.minibatch([self.groups[name][0]])
        with self.assertRaises(AssertionError):
            self.gate.minibatch([self.groups["fit"][0]]*2)
        with self.assertRaises(AssertionError):
            self.gate.minibatch([])

    def test_reject_tampered_label_graph_order_and_split(self):
        self.gate.parent_verified=True
        good=self.groups["fit"][0]
        changes=(dict(src=(good.src+1)%6),dict(reachable=not good.reachable),
                 dict(split="EVAL128"),dict(edges=()),
                 dict(question=good.question+" TRAIN"),dict(query_style=999))
        for kws in changes:
            with self.subTest(changed=kws),self.assertRaises(AssertionError):
                self.gate.minibatch([replace(good,**kws)])

    def test_never_accept_unknown_or_malformed_rows(self):
        self.gate.parent_verified=True
        for bad in (object(),None,"a string",{"prompt_sha256":self.groups["fit"][0].prompt_sha256}):
            with self.assertRaises(AssertionError):
                self.gate.minibatch([bad])


if __name__=="__main__":
    unittest.main()
