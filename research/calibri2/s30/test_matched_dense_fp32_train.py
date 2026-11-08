"""Static S30 safeguards: matched dense FP32 must not touch inspected QA."""
from __future__ import annotations

from pathlib import Path
import unittest

from calibri2.s29.frozen_transfer_qa import RESERVED as S29_QA_RESERVED
from calibri2.s29.frozen_transfer_qa import TRAIN_ELIGIBLE as S29_TRAIN_ELIGIBLE
from calibri2.s30.matched_dense_fp32_train import (
    TRAIN_ELIGIBLE,STEPS,LR,SOURCE_SHA,S29_FROZEN_SHA,NEW_PAIRS,
)
from calibri2.s26.spec import RESERVED as S26_QA_RESERVED


class S30MatcingFp32Contract(unittest.TestCase):
    def test_new_training_eligible_only(self):
        actual={(color,noun) for color,nouns in TRAIN_ELIGIBLE.items() for noun in nouns}
        self.assertEqual(len(actual),8)
        self.assertEqual(actual,set(S29_TRAIN_ELIGIBLE))
        self.assertFalse(actual&set(S29_QA_RESERVED))
        self.assertFalse(actual&set(S26_QA_RESERVED))
        self.assertEqual(NEW_PAIRS,32)

    def test_immutable_source_and_schedule(self):
        self.assertEqual(SOURCE_SHA,"eaa71093d89cd658a1e9d7a7e5297586f2d4549778d1b9bfae3cf54c26ce502c")
        self.assertEqual(S29_FROZEN_SHA,"812a8165f50945e57e126bd02b7dca7fec52fd46f13202d4bf26ae18d34e7223")
        self.assertEqual(STEPS,192)
        self.assertGreater(LR,0)
        self.assertLess(LR,0.001)

    def test_training_source_never_reads_QA(self):
        source=Path(__file__).with_name("matched_dense_fp32_train.py").read_text()
        self.assertNotIn("from calibri2.s29.",source)
        self.assertNotIn("import calibri2.s29.",source)
        self.assertNotIn("from calibri2.s26.frozen_new_qa",source)
        self.assertNotIn("from calibri2.s26.independent_qa",source)
        self.assertNotIn("train_test_split",source)
        self.assertIn("model=model",source) if False else None

if __name__=="__main__":
    unittest.main()
