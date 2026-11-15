"""S15c preflight source-only controls; no heldout model scoring in tests."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import torch

from research.nra3_s15.new_ru_eval24_prereg import (
    new_russian_eval24,EVAL24_CANONICAL_SHA,canonical_new_eval24,
)
from research.nra3_s15.one_time_frozen_s14_eval24 import (
    S15B_TRUE43M_HIDDEN_FP16_SHA,S14_PAID_FP16_SHA,
    S15_EVAL24_SHA,S8B_CB59_SHA,S11_ROUTER_SHA,
    word_gold,physical_sources
)
from research.nra3_s13.paid_relation_feedback import PaidRelationFeedback


class FirstAndOnlyNovelHeldout24Controls(unittest.TestCase):
    def test_sources_frozen_after_SHA_prereg_before_any_metric(self):
        self.assertEqual(EVAL24_CANONICAL_SHA,S15_EVAL24_SHA)
        self.assertEqual(hashlib.sha256(canonical_new_eval24(new_russian_eval24())).hexdigest(),
                         S15_EVAL24_SHA)
        self.assertEqual(len(new_russian_eval24()),24)
        for digest in (S15B_TRUE43M_HIDDEN_FP16_SHA,S14_PAID_FP16_SHA,
                       S8B_CB59_SHA,S11_ROUTER_SHA):
            self.assertEqual(len(digest),64)
        self.assertEqual(S15B_TRUE43M_HIDDEN_FP16_SHA,
           "fc4b438929d90e56ce8edc8743836fdb8f609bfaea228944d71723453227b9a5")
        self.assertEqual(S14_PAID_FP16_SHA,
           "5b96e2174e33da5232fcc2a4a397b6fab4370b6879adb684d75fea8a503f1279")

    def test_billing_and_s14_UPOS_physical_guard_refuses_wrong_bytes(self):
        physical=PaidRelationFeedback().physical_bytes()
        self.assertEqual(len(physical),2581)
        with self.assertRaises(ValueError):
            physical_sources(b"x"*653798,b"y"*123,physical)
        with self.assertRaises(ValueError):
            physical_sources(b"",b"",physical)

    def test_precommitted_past_only_syntax_gold_mask_does_not_look_future(self):
        rows=new_russian_eval24()
        observed_future=observed_past=0
        for row in rows:
            positions=list(range(10,10+len(row.words)))
            gu,gr,candidates=word_gold(
                row,positions,[0]*len(row.words))
            self.assertEqual(len(gu),len(row.words))
            self.assertEqual(len(gr),len(row.words))
            for i,w in enumerate(row.words):
                if w.head>i+1 or w.head==0:
                    observed_future+=1
                    self.assertNotIn(i,[k for k,j in candidates])
                if 0<w.head<=i:
                    observed_past+=1
                    self.assertIn((i,w.head-1),candidates)
            self.assertEqual(len(candidates),len(set(candidates)))
        self.assertGreater(observed_past,20)
        self.assertGreater(observed_future,15)

    def test_new_RU24_gold_roles_and_event_are_not_active_generation(self):
        cases=new_russian_eval24()
        self.assertTrue(any(r.event_truth=="NEGATED" for r in cases))
        self.assertTrue(any(r.event_truth=="FUTURE_UNREALIZED" for r in cases))
        self.assertTrue(any(r.antecedents for r in cases))
        self.assertFalse(PaidRelationFeedback().candidate_accepted)
        self.assertEqual(PaidRelationFeedback().output_gate,0)


if __name__=="__main__":
    unittest.main()
