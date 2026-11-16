"""NRA3 S18 source-only tests: NO original43M, no heldout EVAL, no training."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import unittest

from research.nra3_s9.new_ru_train72 import build_s9_train72
from research.nra3_s4.russian_ud_train_only import build_train as s4_train
from research.nra3_s18.ru96_multiverb_train_source import (
    build_s18_new_train96,canonical_s18_new96_train,
    seal_s18_new96_train,validate_s18_train96,
    direct_predicate_argument_gold,CONSTRUCTIONS,FAMILIES,
    S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,S18_SPLIT
)


class New96RealRussianMultiVerbTrainGold(unittest.TestCase):
    def test_all_96_original_TRAIN_and_12_independent_verbs_8_structures(self):
        rows=build_s18_new_train96()
        self.assertEqual(len(rows),96)
        self.assertEqual({r.split for r in rows},{S18_SPLIT})
        self.assertEqual(len({r.text_sha256() for r in rows}),96)
        self.assertEqual(len({f.verb_lemma for f in FAMILIES}),12)
        self.assertEqual({r.construction for r in rows},set(CONSTRUCTIONS))
        self.assertEqual({r.lexical_family for r in rows},set(range(12)))
        for i in range(12):
            fam=[z for z in rows if z.lexical_family==i]
            self.assertEqual(len(fam),8)
            self.assertEqual({z.construction for z in fam},set(CONSTRUCTIONS))
            lemmas={w.lemma for z in fam for w in z.words if w.upos=="VERB"}
            self.assertIn(FAMILIES[i].verb_lemma,lemmas)

    def test_all_300_directed_links_case_patient_passive_source_and_multiverb(self):
        rows=build_s18_new_train96()
        edges=[x for r in rows for x in direct_predicate_argument_gold(r)]
        self.assertEqual(len(edges),300)
        self.assertEqual({name:sum(x[2]==name for x in edges)
                          for name in ("AGENT","PATIENT","RECIPIENT","SOURCE")},
                         {"AGENT":96,"PATIENT":96,"RECIPIENT":96,"SOURCE":12})
        for r in rows:
            self.assertEqual([z.index for z in r.words],list(range(1,len(r.words)+1)))
            self.assertEqual(sum(z.head==0 for z in r.words),1)
            self.assertTrue(all(r.text[z.start:z.end]==z.form for z in r.words))
            self.assertTrue(all(r.words[p-1].upos=="VERB" and
                                r.words[a-1].head==p for p,a,_ in
                                direct_predicate_argument_gold(r)))
            if r.construction=="passive_agent_patient_inversion":
                self.assertEqual(r.words[0].semantic_role,"PATIENT")
                self.assertEqual(r.words[3].semantic_role,"AGENT")
            if r.construction=="recipient_subject_with_source":
                self.assertEqual(r.words[0].semantic_role,"RECIPIENT")
                self.assertEqual(r.words[4].semantic_role,"SOURCE")
                self.assertEqual(sum(z.semantic_role=="AGENT" for z in r.words),0)
            if r.construction=="two_finite_predicates_distinct_agents":
                self.assertEqual(sum(z.upos=="VERB" for z in r.words),2)
                self.assertEqual(sum(z.semantic_role=="AGENT" for z in r.words),2)
                self.assertEqual(len(direct_predicate_argument_gold(r)),4)

    def test_no_old_S9_S4_train_text_duplicate_and_no_old_EVAL_import(self):
        rows=build_s18_new_train96()
        old={r.sha256() for r in build_s9_train72()}|{
             r.sha256() for r in s4_train()}
        self.assertFalse(old&{z.text_sha256() for z in rows})
        import ast,inspect
        import research.nra3_s18.ru96_multiverb_train_source as source
        tree=ast.parse(inspect.getsource(source))
        imports=[alias.name for n in ast.walk(tree)
                 if isinstance(n,ast.Import) for alias in n.names]
        imports+=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any("nra3_s17" in x or "nra3_s15" in x or
            "nra3_s12" in x for x in imports))

    def test_all_96_full_source_semantic_UD_SHA_is_complete_and_sealed(self):
        rows=build_s18_new_train96()
        orig=canonical_s18_new96_train(rows)
        manifest=seal_s18_new96_train()
        sha=hashlib.sha256(orig).hexdigest()
        self.assertEqual(manifest["S18_complete_original_96_new_TRAIN_full_annotation_SHA256"],sha)
        if S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256:
            self.assertEqual(S18_CANONICAL_96_FULL_UD_AND_ROLES_SHA256,sha)
        self.assertEqual(manifest["S18_original_source_rows"],96)
        self.assertEqual(manifest["S18_new_directed_verb_to_argument_links"],300)
        self.assertEqual(manifest["S18_new_distinct_verb_lemmas"],12)
        self.assertEqual(manifest["S18_new_syntactic_constructions"],8)
        self.assertEqual(manifest["S18_new_optimizer_steps"],0)
        self.assertFalse(manifest["S18_original_43M_GT_NRA_C_or_2309FP16_model_loaded"])
        self.assertTrue(manifest["S18_old_S17c_24_independent_EVAL_now_consumed_not_read"])
        self.assertFalse(manifest["candidate_accepted"])
        self.assertFalse(manifest["activation_allowed"])
        self.assertEqual(manifest["output_gate"],0)
        origrow=rows[0]
        changed=replace(origrow.words[0],lemma="новая_лемма_только_для_SHA_проверки")
        altered=(replace(origrow,words=(changed,)+origrow.words[1:]),)+rows[1:]
        self.assertEqual(origrow.text_sha256(),altered[0].text_sha256())
        self.assertNotEqual(sha,hashlib.sha256(
            canonical_s18_new96_train(altered)).hexdigest())

    def test_reject_wrong_split_gold_head_cycle_and_missing_verb_families(self):
        rows=build_s18_new_train96()
        with self.assertRaises(ValueError):validate_s18_train96(rows[:-1])
        with self.assertRaises(ValueError):
            validate_s18_train96((replace(rows[0],split="EVAL"),)+rows[1:])
        wrong=list(rows[0].words);wrong[0]=replace(wrong[0],head=wrong[0].index)
        with self.assertRaises(ValueError):
            validate_s18_train96((replace(rows[0],words=tuple(wrong)),)+rows[1:])
        replaced=list(rows)
        replaced[0]=replace(replaced[0],text=replaced[1].text)
        with self.assertRaises(ValueError):
            validate_s18_train96(tuple(replaced))


if __name__=="__main__":
    unittest.main()
