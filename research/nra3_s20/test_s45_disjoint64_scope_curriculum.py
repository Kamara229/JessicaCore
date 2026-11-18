"""S45 NEW64 CURRICULUM leakage and exact original S44 optimizer contracts.

Local unit tests are ONLY structural and no Calibri43M/GT inference.
Real 24-fold CPU S44/S45 physical refits run in CI separately.
"""
from __future__ import annotations
import inspect
import unittest
import torch

from research.nra3_s20.s45_separately_frozen_new64_scope_curriculum import (
    train64,validate_source,source_digest,LEXICAL_FAMILIES,STRUCTURES,
)
from research.nra3_s20.s43_new128_multi_event_trainonly_source import (
    build_train128,train_partitions,validate_source as validate128,
)
from research.nra3_s20.s43_train128_joint_predicate_source_24fold import (
    batch_raw_plus_train_labels,
)
from research.nra3_s20.s44_train128_quote_calibrated_source_contrast_24fold import (
    fit_s44,S44_TRAIN_STEPS,S44_TRAIN_LR,
)
from research.nra3_s20.s45_compare_s44_plus_new64_train_curriculum_24fold import (
    micro_summary,
)

class S45AdditionalTrainOnlyCurriculum(unittest.TestCase):
    def test_exact_new64_disjoint_from_train128_and_old_sources(self):
        data=validate_source()
        self.assertEqual(data["TRAIN_only_rows"],64)
        self.assertEqual(data["lexical_families"],8)
        self.assertEqual(data["syntactic_patterns"],8)
        self.assertEqual(data["positive_directed_training_links"],80)
        self.assertEqual(data["negative_event_without_rejected_edges"],8)
        self.assertEqual(data["source_sha256"],source_digest())
        self.assertTrue(data["disjoint_S43_TRAIN128_S39_TRAIN80_S22_TRAIN256"])
        self.assertEqual(len(LEXICAL_FAMILIES),8)
        self.assertEqual(len(STRUCTURES),8)
        self.assertEqual(validate128()["count"],128)
        self.assertFalse({r.text for r in train64()} &
                         {r.text for r in build_train128()})

    def test_s45_additional_train_only_never_overlap_any_held_128(self):
        original=build_train128()
        extra=train64()
        for axis,group,tr,held in train_partitions():
            oldtrain=[original[i] for i in tr]
            heldrows=[original[i] for i in held]
            self.assertEqual(len(oldtrain),112 if axis=="LEXICAL" else 120)
            self.assertEqual(len(heldrows),16 if axis=="LEXICAL" else 8)
            self.assertEqual(len(oldtrain)+len(extra),
                             176 if axis=="LEXICAL" else 184)
            self.assertFalse({r.text for r in oldtrain+list(extra)}
                             &{r.text for r in heldrows})
        self.assertEqual(len(train_partitions()),24)

    def test_multievent_and_quotation_train_label_alignment_for_new64(self):
        new=train64()
        x,q,p,s,v,legal,event,quote,rejected=batch_raw_plus_train_labels(new)
        self.assertEqual(x.shape[0],64)
        self.assertEqual(q.shape[:2],v.shape)
        self.assertEqual(rejected.shape[:2],legal.shape[:2])
        self.assertEqual(int(rejected.sum()),80)
        self.assertTrue(bool(quote.sum()>0))
        self.assertTrue(bool(event.sum()>64))
        self.assertEqual(int(((rejected>.5)&~legal).sum()),0)
        self.assertTrue(torch.isfinite(x).all())
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.isfinite(s).all())

    def test_prior_S44_fit_seed_loss_and_no_S38_inference_reuse(self):
        script=inspect.getsource(fit_s44)
        self.assertIn("allow_S45_new64=False",script)
        self.assertIn("S43_SEED+fold",script)
        self.assertIn("S44_TRAIN_LR",script)
        self.assertNotIn("S38_",script)
        self.assertNotIn("gold[",inspect.getsource(
            __import__("research.nra3_s20.s43_joint_learned_predicate_source_adapter",
                       fromlist=["JointLearnedPredicateSourceS43"]).JointLearnedPredicateSourceS43.forward))
        import research.nra3_s20.s45_compare_s44_plus_new64_train_curriculum_24fold as runner
        text=inspect.getsource(runner)
        self.assertNotIn("from research.nra3_s20.s38_",text)
        self.assertIn("fit_s44(base_train,fold)",text)
        self.assertIn("fit_s44(",text)
        self.assertIn("allow_S45_new64=True",text)
        self.assertIn("score_after_frozen_training",text)
        self.assertEqual(S44_TRAIN_STEPS,120)
        self.assertEqual(S44_TRAIN_LR,.012)

if __name__=="__main__":
    unittest.main()
