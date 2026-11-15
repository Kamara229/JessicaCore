"""S15 newly-authored UD/semantic/coref EVAL24 provenance/source-only guards."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import unittest

from research.nra3_s15.new_ru_eval24_prereg import (
    EVAL24_CANONICAL_SHA,SPLIT,S14_FROZEN_CI,S14_FROZEN_HEAD_SHA,
    S14_SELECTED_FP16_SHA,new_russian_eval24,canonical_new_eval24,
    sealed_new_eval24,validate_new_eval24,
)
from research.nra3_s4.russian_ud_train_only import build_train
from research.nra3_s9.new_ru_train72 import build_s9_train72


class NewRussianPostModelLockedOneTimeTest(unittest.TestCase):
    def test_24_new_full_Ud_roles_anaphora_event_modalities(self):
        rows=new_russian_eval24()
        self.assertEqual(len(rows),24)
        self.assertEqual(len({r.text for r in rows}),24)
        self.assertEqual(len({r.construction for r in rows}),24)
        self.assertTrue(all(r.split==SPLIT for r in rows))
        self.assertTrue(all(r.words[-1].text=="." for r in rows))
        train={r.sha256() for r in build_train()}|{
            r.sha256() for r in build_s9_train72()}
        self.assertFalse(train & {r.sha256() for r in rows})
        self.assertGreaterEqual(sum(r.event_truth=="NEGATED" for r in rows),2)
        self.assertGreaterEqual(sum(r.event_truth=="FUTURE_UNREALIZED" for r in rows),2)
        self.assertGreaterEqual(sum(bool(r.antecedents) for r in rows),4)
        self.assertGreaterEqual(sum(len(r.antecedents) for r in rows),6)
        self.assertGreaterEqual(sum(w.upos=="PRON" for r in rows for w in r.words),5)
        roles={s for row in rows for s in row.roles}
        self.assertTrue({"AGENT","PATIENT","SOURCE","RECIPIENT","DESTINATION",
                         "EXPERIENCER","INSTRUMENT","LOCATION"}<=roles)
        pos={w.upos for row in rows for w in row.words}
        self.assertTrue({"ADJ","ADP","ADV","AUX","CCONJ","NOUN","NUM",
                         "PART","PRON","PROPN","PUNCT","SCONJ","VERB"}<=pos)
        for r in rows:
            self.assertEqual(len(r.words),len(r.roles))
            self.assertEqual(sum(w.head==0 for w in r.words),1)
            for w in r.words:
                self.assertEqual(r.text[w.start:w.end],w.text)
                used={w.word_id}
                target=w.head
                while target:
                    self.assertNotIn(target,used)
                    used.add(target)
                    target=r.words[target-1].head

    def test_source_only_manifest_pin_post_existing_successful_S14_model(self):
        x=sealed_new_eval24()
        self.assertEqual(x["number_of_new_EVAL24_cases"],24)
        self.assertEqual(x["S14_model_frozen_run_BEFORE_S15_source"],43740415883)
        self.assertEqual(x["S14_model_frozen_head_BEFORE_S15_source"],
                         "ab95483ef5dc985b8a32941163fb248bc9640ef3")
        self.assertEqual(x["S14_physical_fp16_selected_sha_BEFORE_S15_source"],
                         "5b96e2174e33da5232fcc2a4a397b6fab4370b6879adb684d75fea8a503f1279")
        self.assertEqual(x["S14_model_frozen_run_BEFORE_S15_source"],S14_FROZEN_CI)
        self.assertEqual(x["S14_model_frozen_head_BEFORE_S15_source"],
                         S14_FROZEN_HEAD_SHA)
        self.assertEqual(x["S14_physical_fp16_selected_sha_BEFORE_S15_source"],
                         S14_SELECTED_FP16_SHA)
        c=canonical_new_eval24(new_russian_eval24())
        self.assertGreater(len(c),15000)
        self.assertEqual(hashlib.sha256(c).hexdigest(),
                         x["complete_UD_semantic_role_coreference_event_gold_sha256"])
        if EVAL24_CANONICAL_SHA:
            self.assertEqual(x["complete_UD_semantic_role_coreference_event_gold_sha256"],
                             EVAL24_CANONICAL_SHA)
        for field in ("S15_new_EVAL24_scored",
                      "S15_new_EVAL24_is_small_hand_authored_not_broad_natural_language"):
            self.assertIn(field,x)
        self.assertFalse(x["S15_new_EVAL24_scored"])
        self.assertEqual(x["S15_new_EVAL24_model_prefills"],0)
        self.assertEqual(x["S15_eval24_optimizer_steps"],0)
        self.assertFalse(x["candidate_accepted"])
        self.assertFalse(x["activation_allowed"])
        self.assertEqual(x["output_gate"],0)

    def test_preregister_fail_closed_split_source_cycles_antecedent_and_negation(self):
        original=new_russian_eval24()
        with self.assertRaises(ValueError):validate_new_eval24(original[:-1])
        with self.assertRaises(ValueError):validate_new_eval24(original[:1]*24)
        with self.assertRaises(ValueError):validate_new_eval24(
            (replace(original[0],split="TRAIN"),)+original[1:])
        with self.assertRaises(ValueError):validate_new_eval24(
            (replace(original[0],construction=original[1].construction),)+original[1:])
        row=original[0]
        broken=list(row.words)
        broken[0]=replace(broken[0],head=1)
        with self.assertRaises(ValueError):validate_new_eval24(
            (replace(row,words=tuple(broken)),)+original[1:])
        with self.assertRaises(ValueError):validate_new_eval24(
            original[:15]+(replace(original[15],antecedents=((1,5),)),)+original[16:])
        with self.assertRaises(ValueError):validate_new_eval24(
            original[:3]+(replace(original[3],event_truth="ASSERTED"),)+original[4:])
        # Editing gold roles without changing sentence text is still a
        # different full canonical SHA, so only frozen SHA is valid later.
        modified=replace(row,roles=("NONE",)+row.roles[1:])
        self.assertEqual(row.sha256(),modified.sha256())
        mutated=(modified,)+original[1:]
        self.assertNotEqual(
            hashlib.sha256(canonical_new_eval24(mutated)).hexdigest(),
            hashlib.sha256(canonical_new_eval24(original)).hexdigest())


if __name__=="__main__":
    unittest.main()
