"""S17 independent NEW EVAL source quality checks, no model calls."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import unittest

from research.nra3_s9.new_ru_train72 import build_s9_train72
from research.nra3_s4.russian_ud_train_only import build_train as build_s4_train
from research.nra3_s17.new_ru24_directed_eval_prereg import (
    SPLIT,S16B_FROZEN_SOURCE_RUN,S16B_FROZEN_SOURCE_SHA,
    S16B_FROZEN_FULL_HEAD_SHA,S17_EVAL24_COMPLETE_SHA,
    build_new_eval24,canonical_eval24,preregister_new_eval24,
    validate_new_eval24,gold_links
)

class NewIndependentRussian24SourceOnly(unittest.TestCase):
    def test_exact_new_unseen_24_texts_and_UD_lexical_contexts(self):
        rows=build_new_eval24()
        self.assertEqual(len(rows),24)
        self.assertEqual(len(set(r.text for r in rows)),24)
        self.assertEqual(len({r.construction for r in rows}),24)
        self.assertEqual({r.split for r in rows},{SPLIT})
        previous={x.sha256() for x in build_s9_train72()} | {
                  x.sha256() for x in build_s4_train()}
        self.assertFalse(previous & {r.text_sha256() for r in rows})
        links=[x for r in rows for x in gold_links(r)]
        self.assertGreaterEqual(len(links),65)
        self.assertGreaterEqual(sum(z[2]=="SOURCE" for z in links),2)
        self.assertGreaterEqual(sum(z[2]=="AGENT" for z in links),15)
        self.assertGreaterEqual(sum(z[2]=="PATIENT" for z in links),20)
        self.assertGreaterEqual(sum(z[2]=="RECIPIENT" for z in links),20)
        self.assertGreaterEqual(sum(r.event_modality=="NEGATED" for r in rows),2)
        self.assertGreaterEqual(sum(r.event_modality=="FUTURE_UNREALIZED" for r in rows),2)
        self.assertGreaterEqual(sum(sum(w.upos=="VERB" for w in r.words)>1
                                    for r in rows),3)
        self.assertTrue(any("nsubj:pass" in [w.deprel for w in r.words] for r in rows))
        self.assertTrue(any(any(w.upos=="PRON" for w in r.words) for r in rows))

    def test_label_head_graph_and_causal_directions_pass_when_valid(self):
        for row in build_new_eval24():
            self.assertEqual(row.text[row.words[-1].start:row.words[-1].end],".")
            self.assertEqual(sum(z.head==0 for z in row.words),1)
            self.assertTrue(all(row.text[z.start:z.end]==z.text for z in row.words))
            for predicate,argument,role in gold_links(row):
                self.assertEqual(row.words[predicate-1].upos,"VERB")
                self.assertEqual(row.words[argument-1].head,predicate)
                self.assertNotEqual(predicate,argument)
                self.assertIn(role,("AGENT","PATIENT","RECIPIENT","SOURCE"))
                self.assertLess(min(predicate,argument),max(predicate,argument))
            for z in row.words:
                seen={z.index};head=z.head
                while head:
                    self.assertNotIn(head,seen)
                    seen.add(head);head=row.words[head-1].head

    def test_SHA_is_complete_gold_not_just_sentence_text(self):
        rows=build_new_eval24()
        source=canonical_eval24(rows)
        fullsha=hashlib.sha256(source).hexdigest()
        m=preregister_new_eval24()
        self.assertEqual(m["S17_new_Russian_EVAL_complete_UD_and_directed_link_SHA256"],fullsha)
        self.assertEqual(m["S17_new_Russian_EVAL_sentence_count"],24)
        self.assertGreater(len(source),15000)
        if S17_EVAL24_COMPLETE_SHA:self.assertEqual(fullsha,S17_EVAL24_COMPLETE_SHA)
        changed=replace(rows[0].words[0],role="RECIPIENT")
        row=replace(rows[0],words=(changed,)+rows[0].words[1:])
        alt=(row,)+rows[1:]
        self.assertNotEqual(fullsha,hashlib.sha256(canonical_eval24(alt)).hexdigest())
        self.assertEqual(rows[0].text_sha256(),alt[0].text_sha256())
        self.assertEqual(m["S17_model_forward_calls"],0)
        self.assertFalse(m["S17_model_evaluation_scored"])
        self.assertEqual(m["S17_optimizer_steps"],0)
        self.assertTrue(m["S17_old_consumed_S15c_EVAL24_never_loaded"])
        self.assertFalse(m["candidate_accepted"])
        self.assertFalse(m["activation_allowed"])
        self.assertEqual(m["output_gate"],0)

    def test_first_model_source_frozen_before_new_heldout_exists(self):
        r=preregister_new_eval24()
        self.assertEqual(r["S16b_frozen_physical_source_run_BEFORE_EVAL"],43758892125)
        self.assertEqual(r["S16b_frozen_physical_SHA_BEFORE_EVAL"],
          "3fe4bbce718e53e4e812e94367ce25834f873cbed646479c59b1e3949d2ecf38")
        self.assertEqual(r["S16b_frozen_head_SHA_BEFORE_EVAL"],
          "ea063bf1fa0e856111c54f3b045220ca7aa5319e")
        self.assertEqual(S16B_FROZEN_SOURCE_RUN,43758892125)
        self.assertEqual(S16B_FROZEN_SOURCE_SHA,
          r["S16b_frozen_physical_SHA_BEFORE_EVAL"])
        self.assertEqual(S16B_FROZEN_FULL_HEAD_SHA,
          r["S16b_frozen_head_SHA_BEFORE_EVAL"])
        self.assertTrue(r["S17_old_consumed_S15c_EVAL24_never_loaded"])
        self.assertTrue(r["S17_old_consumed_S12c_EVAL16_never_loaded"])
        self.assertTrue(r["S17_old_five_EVAL128_and_reserved_sixth_S3CI_never_loaded"])
        self.assertTrue(r["S17_event_truth_and_pronoun_coreference_NOT_scored_by_valence_head"])

    def test_reject_semantic_gold_or_split_mutation_or_cycle(self):
        rows=build_new_eval24()
        with self.assertRaises(ValueError):
            validate_new_eval24(rows[:-1])
        with self.assertRaises(ValueError):
            validate_new_eval24(rows[:1]*24)
        with self.assertRaises(ValueError):
            validate_new_eval24((replace(rows[0],split="TRAIN"),)+rows[1:])
        bad=list(rows[0].words)
        bad[0]=replace(bad[0],head=1)
        with self.assertRaises(ValueError):
            validate_new_eval24((replace(rows[0],words=tuple(bad)),)+rows[1:])
        with self.assertRaises(ValueError):
            validate_new_eval24((replace(rows[10],event_modality="ASSERTED"),)+rows[:10]+rows[11:])


if __name__=="__main__":
    unittest.main()
