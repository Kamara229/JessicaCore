"""Static S29 precommit seals: source-only, NO models/QA inference."""
from __future__ import annotations
import json
import unittest

from calibri2.s26.frozen_new_qa import cases as s26_cases
from calibri2.s26.spec import ALL_NOUNS as S26_NOUNS
from calibri2.s29.frozen_transfer_qa import (
    COLORS,NEW_NOUNS,RESERVED,TRAIN_ELIGIBLE,
    OPEN_16,novel_96,controls_48,cases,manifest
)

class FrozenS29Contract(unittest.TestCase):
    def test_precommit_deterministic_and_counts(self):
        m=manifest()
        self.assertEqual(m["frozen_sha256"],manifest()["frozen_sha256"])
        self.assertEqual([len(novel_96()),len(controls_48()),len(OPEN_16)],[96,48,16])
        self.assertEqual(len(cases()),144)
        self.assertEqual(len({r["id"] for r in cases()}),144)
        self.assertEqual(len({r["question"] for r in cases()}),144)
        self.assertEqual(set(r["split"] for r in cases()),{"QA"})
        self.assertEqual(sum(r["gold"]=="Yes" for r in controls_48()),12)
        self.assertEqual(sum(r["gold"]=="No" for r in controls_48()),12)

    def test_no_reuse_s26_qa(self):
        current=cases()
        old=s26_cases()
        self.assertFalse({r["question"] for r in current}&
                         {r["question"] for r in old})
        self.assertFalse({r["id"] for r in current}&{r["id"] for r in old})
        self.assertFalse(set(NEW_NOUNS)&set(S26_NOUNS))
        self.assertFalse(set(RESERVED)&set(TRAIN_ELIGIBLE))
        self.assertEqual(len(RESERVED),8)
        self.assertEqual(len(TRAIN_ELIGIBLE),8)

    def test_target_and_foil_not_reserved(self):
        all_new_targets=set()
        for r in novel_96():
            key=(r["novel_color"],r["novel_noun"])
            self.assertIn(key,RESERVED)
            self.assertEqual(r["gold"]," ".join(key))
            self.assertIn(r["gold"],r["question"])
            all_new_targets.add(key)
            # Withheld noun must not appear with another color in QA
            # prompt; otherwise the question could teach an unintended
            # new combination of the heldout noun within the same case.
            for color in COLORS:
                if color!=key[0]:
                    self.assertNotIn(f"{color} {key[1]}",r["question"])
        self.assertEqual(all_new_targets,set(RESERVED))

    def test_no_weights_training_or_bpp_claim(self):
        m=manifest()
        for field in (
          "S26_QA_already_inspected_and_not_reused_for_training",
          "S29_evaluation_new_and_frozen_before_any_subsequent_training",
          "no_checkpoints_or_hyperparameters_selected_using_these_answers",
          "require_entire_model_generated_answer_plus_EOS",
          "do_not_import_this_module_into_training"
        ):
            self.assertTrue(m[field])
        self.assertNotIn("accuracy",m)
        self.assertNotIn("model_result",m)

if __name__=="__main__":
    unittest.main()
