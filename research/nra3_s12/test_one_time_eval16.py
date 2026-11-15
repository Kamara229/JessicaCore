"""S12c predeclared ONE heldout evaluation: source and no-gold contracts.

No original 43M, GT, S11 weights or protected EVAL are opened in unit
tests. The real source is locked independently by S12b before scoring CI.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import torch

from research.nra3_s12.heldout_ru_eval16 import (
    EVAL16_FULL_CANONICAL_SHA,build_eval16,canonical_eval16,SPLIT,
)
from research.nra3_s12.one_time_eval16 import (
    S12_PRE_REGISTER_FULL_SHA,
    S11_ORIGINAL_39_FP16_SHA,
    S8b_ORIGINAL_FROZEN_43_CB59_SHA,
    weak_eval_labels_posthoc,
)
from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord,CausalAlignment,
)
from research.nra3_s11.weak_ru72_train_labels import (
    weak_train_router_labels,
)
from research.nra3_s11.paid_router import (
    PaidThreeWayRouter,PHYSICAL_BYTES,SOURCE_PARAMETERS,
)


class FrozenOneTimeSixteenContracts(unittest.TestCase):
    def test_exact_gold_source_and_model_digest_frozen_before_EVAL(self):
        self.assertEqual(EVAL16_FULL_CANONICAL_SHA,S12_PRE_REGISTER_FULL_SHA)
        self.assertEqual(hashlib.sha256(canonical_eval16(build_eval16())).hexdigest(),
                         EVAL16_FULL_CANONICAL_SHA)
        self.assertEqual(len(build_eval16()),16)
        self.assertEqual(SOURCE_PARAMETERS,39)
        self.assertEqual(PHYSICAL_BYTES,123)
        self.assertEqual(len(S11_ORIGINAL_39_FP16_SHA),64)
        self.assertEqual(len(S8b_ORIGINAL_FROZEN_43_CB59_SHA),64)
        self.assertEqual(PaidThreeWayRouter().physical_bytes(),
                         PaidThreeWayRouter().physical_bytes())

    def test_all_heldout_gold_revealed_only_in_posthoc_eval(self):
        rows=build_eval16()
        histogram=torch.zeros(3,dtype=torch.long)
        for row in rows:
            n=len(row.words)
            words=tuple(AlignedWord(i+1,(i+1,),i+1)
                        for i in range(n))
            aligned=CausalAlignment(row.sha256(),n+2,tuple(range(n+2)),
                                    words,n+1,True)
            y=weak_eval_labels_posthoc(row,aligned)
            self.assertEqual(len(y),len(row.words))
            histogram+=torch.bincount(y,minlength=3)
            with self.assertRaises(ValueError):
                weak_eval_labels_posthoc(
                    row,replace(aligned,independent_eval_labels_loaded=False))
            with self.assertRaises(ValueError):
                weak_train_router_labels(row,aligned)
            modified=replace(row,split="TRAIN")
            with self.assertRaises(ValueError):
                weak_eval_labels_posthoc(modified,aligned)
            old_role=row.roles[0]
            new_role="NONE" if old_role!="NONE" else "AGENT"
            edited=replace(row,roles=(new_role,)+row.roles[1:])
            self.assertEqual(row.sha256(),edited.sha256())
            with self.assertRaises(ValueError):
                weak_eval_labels_posthoc(edited,aligned)
        self.assertTrue(bool((histogram>0).all()))
        self.assertGreater(int(histogram.sum()),70)


if __name__=="__main__":
    unittest.main()
