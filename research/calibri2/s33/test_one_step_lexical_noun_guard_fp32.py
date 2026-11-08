"""S33 contract regression: TRAIN-only noun contrast and exact old answer gate."""
import unittest
from pathlib import Path

from calibri2.s33.one_step_lexical_noun_guard_fp32 import (
    SOURCE_SHA,S29_QA_FROZEN_SHA,TOTAL_STEPS,LR_BACKTRACK,
    OLD_GRADIENT_BLEND,NOUN_CE_WEIGHT,NOUN_FOIL_MARGIN,
    NOUN_OVER_EOS_MARGIN,SCHEDULE,SEED,
)
from calibri2.s30.matched_dense_fp32_train import (
    SOURCE_SHA as S30_SEED,S29_FROZEN_SHA
)
from calibri2.s32.microguard_pcgrad_fp32 import guarded_old_ids

class S33Contract(unittest.TestCase):
    def test_exact_FP32_source_and_fixed_single_training_program(self):
        self.assertEqual(SOURCE_SHA,S30_SEED)
        self.assertEqual(S29_QA_FROZEN_SHA,S29_FROZEN_SHA)
        self.assertEqual(TOTAL_STEPS,128)
        self.assertEqual(LR_BACKTRACK,(.00005,.00002,.000008))
        self.assertEqual(OLD_GRADIENT_BLEND,.2)
        self.assertEqual(NOUN_CE_WEIGHT,12.)
        self.assertEqual(NOUN_FOIL_MARGIN,3.)
        self.assertEqual(NOUN_OVER_EOS_MARGIN,2.)
        self.assertEqual(SCHEDULE,("new","new","old","new"))
        self.assertEqual(SEED,3030)

    def test_exact_answer_ID_preservation_not_accuracy_only(self):
        before=[{"id":"old_a","full_correct":True},
                {"id":"old_b","full_correct":True},
                {"id":"old_c","full_correct":False}]
        after=[{"id":"old_a","full_correct":True},
               {"id":"old_b","full_correct":False},
               {"id":"old_c","full_correct":True}]
        old=guarded_old_ids(before)
        new=guarded_old_ids(after)
        self.assertEqual(len(old),len(new))
        self.assertFalse(old.issubset(new))

    def test_no_qa_or_saved_reports_used_for_training_selection(self):
        source=Path(__file__).with_name("one_step_lexical_noun_guard_fp32.py").read_text()
        for forbidden in (
            "from calibri2.s29.", "import calibri2.s29.",
            "from calibri2.s26.frozen_new_qa",
            "from calibri2.s26.independent_qa",
            "S27_POSTHOC", "S28_READONLY"
        ):
            self.assertNotIn(forbidden,source)
        self.assertIn("old_train_probe(model,tok,old)",source)
        self.assertIn("protected.issubset(set(retained))",source)
        self.assertIn("model.load_state_dict(pre_weights)",source)
        self.assertIn("opt.load_state_dict(pre_optim)",source)
        self.assertIn("NOUN_FOIL_MARGIN",source)
        self.assertIn("NOUN_OVER_EOS_MARGIN",source)

if __name__=="__main__":
    unittest.main()
