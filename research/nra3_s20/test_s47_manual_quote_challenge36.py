"""Structural self-tests only, do NOT tune on S47 adversarial gold."""
from __future__ import annotations
import unittest

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s47_frozen_manual_quote_challenge36 import (
    STAGE, challenge36, unmark, sealed_source_digest,
)
from research.nra3_s20.s47_first_sealed_quote_challenge36_cpu import (
    manual_token_gold, stats, SOURCE_COMMIT, SOURCE_BLOB,
)
from research.nra3_s20.s43_new128_multi_event_trainonly_source import build_train128
from research.nra3_s20.s45_separately_frozen_new64_scope_curriculum import train64


class S47FrozenManualAuditStructure(unittest.TestCase):
    def test_sealed36_all_have_valid_independent_character_labels(self):
        rows=challenge36()
        self.assertEqual(len(rows),36)
        self.assertEqual(len({r.case_id for r in rows}),36)
        self.assertEqual(len(sealed_source_digest()),64)
        self.assertIn("SEPARATELY_PRECOMMITTED",STAGE)
        self.assertEqual(SOURCE_COMMIT,"3f997d2af82ac75cd2aa0310a9af6aa208dbbb7f")
        self.assertEqual(SOURCE_BLOB,"e46c88b14acd9c95b9dced410a69b36e3b5dff64")
        for r in rows:
            tokens=raw_surface_grammar(r.text)
            self.assertGreaterEqual(len(tokens),3,r.case_id)
            self.assertLessEqual(len(tokens),40,r.case_id)
            self.assertNotIn("⟦",r.text)
            self.assertNotIn("⟧",r.text)
            gold=manual_token_gold(tokens,r.gold_character_spans)
            self.assertEqual(len(gold),len(tokens))
            self.assertTrue(all(0<=start<stop<=len(r.text)
                                for start,stop in r.gold_character_spans))
        self.assertFalse({r.text for r in rows} &
                         {r.text for r in [*build_train128(),*train64()]})

    def test_unmarked_manual_spans_do_not_derive_from_quote_heuristic(self):
        plain,spans=unmark('Раз «⟦текст ⟧“⟦внутри⟧”⟦ далее⟧» завершено.')
        self.assertEqual(plain,'Раз «текст “внутри” далее» завершено.')
        self.assertEqual(len(spans),3)
        with self.assertRaisesRegex(ValueError,"unclosed"):
            unmark("Обрыв ⟦текста")
        with self.assertRaisesRegex(ValueError,"unmatched"):
            unmark("Ошибка ⟧ конца")

    def test_malformed_quote_policy_sealed_without_gold(self):
        cases={r.case_id:r for r in challenge36()}
        for key in ("Q23","Q24","Q25","Q26","Q28","Q29","Q30","Q31","Q33"):
            self.assertEqual(cases[key].gold_character_spans,(),key)
        self.assertGreater(len(cases["Q07"].gold_character_spans),1)
        self.assertGreater(len(cases["Q22"].gold_character_spans),1)
        self.assertGreater(len(cases["Q27"].gold_character_spans),0)
        self.assertGreater(len(cases["Q36"].gold_character_spans),0)

    def test_metric_support_math(self):
        import torch
        mask=torch.tensor([True,False,True,False])
        gold=torch.tensor([True,True,False,False])
        self.assertEqual(stats(mask,gold),{"TP":1,"FP":1,"FN":1})


if __name__=="__main__":
    unittest.main()
