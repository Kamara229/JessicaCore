"""S12 one-time Russian heldout source: annotation, leakage and freeze tests."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

from research.nra3_s12.heldout_ru_eval16 import (
  EVAL16_FULL_CANONICAL_SHA,SPLIT,
  build_eval16,canonical_eval16,sealed_eval16,validate_eval16,
  PREVIOUS_S11_FROZEN_RUN,PREVIOUS_S11_FROZEN_HEAD_SHA,
  PREVIOUS_S11_PHYSICAL_ROUTER_SHA,
)
from research.nra3_s4.russian_ud_train_only import build_train
from research.nra3_s9.new_ru_train72 import build_s9_train72


class OneTimeHeldoutSixteen(unittest.TestCase):
    def test_new_16_standalone_cases_no_train_or_old_evals(self):
        r=build_eval16()
        self.assertEqual(len(r),16)
        self.assertTrue(all(row.split==SPLIT for row in r))
        self.assertEqual(len({row.construction for row in r}),16)
        train={x.sha256() for x in build_train()}|{
            x.sha256() for x in build_s9_train72()}
        self.assertFalse(train&{x.sha256() for x in r})
        self.assertTrue(any("не" in [w.text for w in x.words] for x in r))
        self.assertTrue(any("nsubj:pass" in [w.deprel for w in x.words] for x in r))
        self.assertTrue(any(x.roles[0]=="RECIPIENT" for x in r))
        self.assertTrue(any(x.words[0].deprel=="obj" for x in r))
        for x in r:
            self.assertEqual(x.words[0].start,0)
            self.assertEqual(x.words[-1].end,len(x.text))
            self.assertEqual(x.text[-1],".")
            self.assertEqual(sum(w.head==0 for w in x.words),1)
            self.assertEqual(len(x.words),len(x.roles))
            for w in x.words:
                self.assertEqual(x.text[w.start:w.end],w.text)
                seen={w.word_id}
                head=w.head
                while head:
                    self.assertNotIn(head,seen)
                    seen.add(head)
                    head=x.words[head-1].head

    def test_full_canonical_immutable_source_sha_and_model_precedes_evaluation(self):
        a=sealed_eval16()
        self.assertEqual(a["rows"],16)
        self.assertEqual(a["overlap_s4_s9_train_count"],0)
        self.assertEqual(a["prior_S11_locked_pre_EVAL_run"],PREVIOUS_S11_FROZEN_RUN)
        self.assertEqual(a["prior_S11_locked_pre_EVAL_head_sha"],
                         PREVIOUS_S11_FROZEN_HEAD_SHA)
        self.assertEqual(a["prior_S11_router_fp16_physical_sha256"],
                         PREVIOUS_S11_PHYSICAL_ROUTER_SHA)
        self.assertFalse(a["evaluation_started"])
        self.assertFalse(a["quality_or_generalization_gain_proven"])
        self.assertEqual(a["optimizer_steps_on_eval"],0)
        self.assertFalse(a["candidate_accepted"])
        self.assertEqual(a["output_gate"],0)
        observed=hashlib.sha256(canonical_eval16(build_eval16())).hexdigest()
        self.assertEqual(a["complete_annotation_sha256"],observed)
        if EVAL16_FULL_CANONICAL_SHA:
            self.assertEqual(observed,EVAL16_FULL_CANONICAL_SHA)
        self.assertGreater(len(canonical_eval16(build_eval16())),12_000)

    def test_fail_closed_on_duplicates_and_train_laundering(self):
        a=build_eval16()
        with self.assertRaises(ValueError):
            validate_eval16(a[:-1])
        with self.assertRaises(ValueError):
            validate_eval16(a[:1]*16)
        with self.assertRaises(ValueError):
            validate_eval16((replace(a[0],split="TRAIN"),)+a[1:])
        with self.assertRaises(ValueError):
            validate_eval16((replace(a[0],construction=a[1].construction),)+a[1:])
        one=a[0]
        bad_words=list(one.words)
        bad_words[0]=replace(bad_words[0],head=bad_words[0].word_id)
        with self.assertRaises(ValueError):
            validate_eval16((replace(one,words=tuple(bad_words)),)+a[1:])
        neg=a[8]
        self.assertIn("не",[w.text for w in neg.words])
        with self.assertRaises(ValueError):
            validate_eval16(a[:8]+(replace(neg,construction="event_occurred"),)+a[9:])


if __name__=="__main__":
    unittest.main()
