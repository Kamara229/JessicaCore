"""S9: new train72 deterministic UD, no earlier TRAIN/EVAL reuse.

These tests validate ANNOTATION CONTRACT, not linguistic proficiency.
"""
from __future__ import annotations

import dataclasses
import hashlib
import unittest

from research.nra3_s9.new_ru_train72 import (
    S9_TRAIN_SPLIT,S9_TRAIN72_CANONICAL_SHA256,TEMPLATES,
    build_s9_train72,canonical_train72,seal_train72,
    validate_s9_train72,
)
from research.nra3_s4.russian_ud_train_only import build_train as old_train


class NewRussian72(unittest.TestCase):
    def test_72_new_train_only_rows_six_contrastive_forms(self):
        rows=build_s9_train72()
        self.assertEqual(len(rows),72)
        self.assertEqual(len({x.sha256() for x in rows}),72)
        self.assertEqual(len({x.lexical_family for x in rows}),12)
        self.assertEqual(set(x.construction for x in rows),set(TEMPLATES))
        self.assertTrue(all(x.split==S9_TRAIN_SPLIT for x in rows))
        self.assertFalse({x.sha256() for x in rows}&
                         {x.sha256() for x in old_train()})
        for k in range(12):
            family=rows[6*k:6*k+6]
            self.assertEqual(tuple(x.construction for x in family),TEMPLATES)
            self.assertEqual(len({x.lexical_family for x in family}),1)
            self.assertEqual({x.roles.count("PATIENT") for x in family},{1})

    def test_semantic_subject_not_always_agent_and_voice_changes(self):
        rows=build_s9_train72()
        for k in range(12):
            a,b,c,d,e,f=rows[6*k:6*k+6]
            self.assertEqual(a.words[0].deprel,"nsubj")
            self.assertEqual(a.roles[0],"AGENT")
            self.assertEqual(b.words[0].deprel,"obj")
            self.assertEqual(b.roles[0],"PATIENT")
            self.assertEqual(c.words[0].deprel,"nsubj")
            self.assertEqual(c.roles[0],"RECIPIENT")
            self.assertEqual(c.roles[4],"SOURCE")
            self.assertNotIn("AGENT",c.roles)
            self.assertEqual(d.words[0].deprel,"nsubj:pass")
            self.assertEqual(d.roles[0],"PATIENT")
            self.assertEqual(d.roles[3],"AGENT")
            self.assertIn("Voice=Pass",d.words[2].feats)
            self.assertEqual(e.words[0].deprel,"amod")
            self.assertEqual(e.words[0].head,2)
            self.assertEqual(f.words[0].deprel,"advmod")
            self.assertEqual(f.words[0].head,3)
            # Same semantic roles under inversion for same family.
            for role in ("AGENT","RECIPIENT","PATIENT"):
                self.assertEqual(a.roles.count(role),b.roles.count(role))
                self.assertEqual(a.roles.count(role),d.roles.count(role))

    def test_real_UD_offsets_are_complete_and_heads_noncyclic(self):
        rows=build_s9_train72()
        for row in rows:
            n=len(row.words)
            self.assertEqual(row.text[-1],".")
            self.assertEqual(row.words[-1].text,".")
            self.assertEqual(row.words[-1].end,len(row.text))
            self.assertEqual(row.words[0].start,0)
            self.assertEqual(sum(w.head==0 for w in row.words),1)
            for w in row.words:
                self.assertEqual(row.text[w.start:w.end],w.text)
                visited={w.word_id}
                cur=w.head
                while cur!=0:
                    self.assertNotIn(cur,visited,"S9 UD syntax contains a dependency cycle")
                    visited.add(cur)
                    cur=row.words[cur-1].head
                self.assertLessEqual(len(visited),n)
        self.assertGreater(len(canonical_train72(rows)),10_000)

    def test_immutable_source_and_fail_closed_overwrites(self):
        rows=build_s9_train72()
        source=canonical_train72(rows)
        report=seal_train72()
        self.assertEqual(report["source_rows"],72)
        self.assertEqual(report["additional_independent_eval_rows_used"],0)
        self.assertEqual(report["old_five_EVAL128_untouched"],True)
        self.assertEqual(report["reserved_sixth_S3CI_EVAL128_untouched"],True)
        self.assertEqual(report["candidate_accepted"],False)
        self.assertEqual(report["output_gate"],0)
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         report["canonical_train72_sha256"])
        if S9_TRAIN72_CANONICAL_SHA256:
            self.assertEqual(report["canonical_train72_sha256"],
                             S9_TRAIN72_CANONICAL_SHA256)
        with self.assertRaises(ValueError):
            validate_s9_train72(rows[:-1])
        with self.assertRaises(ValueError):
            validate_s9_train72((rows[0],)*72)
        bad=dataclasses.replace(rows[2],split="EVAL")
        with self.assertRaises(ValueError):
            validate_s9_train72(rows[:2]+(bad,)+rows[3:])
        words=list(rows[0].words)
        words[0]=dataclasses.replace(words[0],head=words[0].word_id)
        bad=dataclasses.replace(rows[0],words=tuple(words))
        with self.assertRaises(ValueError):
            validate_s9_train72((bad,)+rows[1:])


if __name__=="__main__":
    unittest.main()
