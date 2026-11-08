"""S31 source-only TRAIN skill preservation guards."""
from pathlib import Path
import unittest

from calibri2.s31.guarded_dense_fp32_train import (
    START_SHA,TRAIN_STEPS,LR,BLOCK_SIZE,OLD_GUARD_WEIGHT,
    SCHEDULE,S29_SHA,full_correct_ids,
)
from calibri2.s30.matched_dense_fp32_train import SOURCE_SHA,S29_FROZEN_SHA

class S31(unittest.TestCase):
    def test_sealed_source_same_fp32_as_s30(self):
        self.assertEqual(START_SHA,SOURCE_SHA)
        self.assertEqual(S29_SHA,S29_FROZEN_SHA)
        self.assertEqual(TRAIN_STEPS,192)
        self.assertEqual(LR,0.00015)
        self.assertEqual(BLOCK_SIZE,24)
        self.assertEqual(OLD_GUARD_WEIGHT,.5)
        self.assertEqual(SCHEDULE,("new","new","old","new"))

    def test_old_answer_identity_not_just_accuracy_count(self):
        before={"answers":[
          {"id":"a","kind":"historical_TRAIN","strict_correct_full_EOS":True},
          {"id":"b","kind":"historical_TRAIN","strict_correct_full_EOS":True},
          {"id":"c","kind":"historical_TRAIN","strict_correct_full_EOS":False},
        ]}
        after={"answers":[
          {"id":"a","kind":"historical_TRAIN","strict_correct_full_EOS":True},
          {"id":"b","kind":"historical_TRAIN","strict_correct_full_EOS":False},
          {"id":"c","kind":"historical_TRAIN","strict_correct_full_EOS":True},
        ]}
        before_ids=full_correct_ids(before,"historical_TRAIN")
        after_ids=full_correct_ids(after,"historical_TRAIN")
        self.assertEqual(len(before_ids),len(after_ids))
        self.assertFalse(before_ids.issubset(after_ids))

    def test_no_qa_access_training(self):
        source=Path(__file__).with_name("guarded_dense_fp32_train.py").read_text()
        self.assertNotIn("from calibri2.s29.",source)
        self.assertNotIn("from calibri2.s26.frozen_new_qa",source)
        self.assertNotIn("from calibri2.s26.independent_qa",source)
        self.assertIn("optim.load_state_dict(state_before)",source)
        self.assertIn("model.load_state_dict(weights_before)",source)
        self.assertIn("old_baseline.issubset(remaining)",source)

if __name__=="__main__":
    unittest.main()
