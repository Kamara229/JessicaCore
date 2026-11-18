"""S46 structural integrity tests only; NOT independent language evaluation."""
from __future__ import annotations
import unittest
import torch
from research.nra3_s20.s46_train_fitted_quote_boundary_24fold import (
    QuoteBoundaryReadout, counts, micro, STAGE, S45_RUN,
    CALIBRATION_STEPS, CALIBRATION_LR,
)
from research.nra3_s20.s43_new128_multi_event_trainonly_source import (
    train_partitions, build_train128,
)
from research.nra3_s20.s45_separately_frozen_new64_scope_curriculum import (
    train64, validate_source,
)


class TrainOnlyBoundaryContract(unittest.TestCase):
    def test_single_trainable_scalar_and_raw_only_forward(self):
        m = QuoteBoundaryReadout()
        self.assertEqual(sum(p.numel() for p in m.parameters()), 1)
        self.assertFalse(m.activation_allowed)
        self.assertFalse(m.candidate_accepted)
        self.assertEqual(m.output_gate, 0)
        raw = torch.tensor([[0., 1., 0., 1.]])
        frozen = torch.tensor([[.9, -.9, 2., -2.]])
        out = m(frozen, raw)
        self.assertEqual(tuple(out.shape), tuple(raw.shape))
        self.assertTrue(torch.all(out[raw == 1] > frozen[raw == 1]))
        self.assertTrue(torch.all(out[raw == 0] < frozen[raw == 0]))
        loss = torch.nn.functional.softplus(-out[raw == 1]).mean() + (
            torch.nn.functional.softplus(out[raw == 0]).mean())
        loss.backward()
        self.assertIsNotNone(m.raw_strength.grad)
        self.assertTrue(torch.isfinite(m.raw_strength.grad).item())
        self.assertEqual(CALIBRATION_STEPS, 120)
        self.assertEqual(CALIBRATION_LR, .15)

    def test_no_gold_or_training_labels_as_model_forward(self):
        m = QuoteBoundaryReadout()
        with self.assertRaisesRegex(ValueError, "boundary dimension"):
            m(torch.zeros(2, 4), torch.zeros(2, 3))
        with self.assertRaisesRegex(ValueError, "must be binary"):
            m(torch.zeros(1, 2), torch.tensor([[.5, 0.]]))

    def test_24_old_group_splits_extra64_never_text_overlap(self):
        rows = build_train128()
        extra = train64()
        self.assertEqual(validate_source()["TRAIN_only_rows"], 64)
        self.assertEqual(len(rows), 128)
        self.assertEqual(len(extra), 64)
        seen = 0
        for axis, group, training, held in train_partitions():
            source = [rows[i] for i in training] + list(extra)
            hold = [rows[i] for i in held]
            self.assertFalse({r.text for r in source} & {r.text for r in hold})
            self.assertEqual(len(source), 176 if axis == "LEXICAL" else 184)
            self.assertEqual(len(hold), 16 if axis == "LEXICAL" else 8)
            seen += 1
        self.assertEqual(seen, 24)

    def test_offline_metric_accounting(self):
        valid = torch.tensor([[True, True, True, False]])
        ref = torch.tensor([[True, False, True, False]])
        pred = torch.tensor([[True, True, False, True]])
        self.assertEqual(counts(pred, ref, valid), {
            "TP": 1, "FP": 1, "FN": 1})
        records = [
            {"axis": "LEXICAL", "q": {"TP": 2, "FP": 0, "FN": 1}},
            {"axis": "LEXICAL", "q": {"TP": 1, "FP": 1, "FN": 0}},
        ]
        self.assertEqual(micro(records, "q", "LEXICAL")["F1"], .75)
        self.assertEqual(S45_RUN, 44252197919)
        self.assertIn("S46", STAGE)


if __name__ == "__main__":
    unittest.main()
