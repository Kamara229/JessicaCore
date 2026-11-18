"""S48 structural-only unit tests; protected S47/S38 not imported or scored.

No S48 HELD26 labels are used in these tests. The held source is
imported only after fitting during the physical CPU evaluator stage.
"""
from __future__ import annotations
import unittest
import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s48_matched_unicode_quote_graph import matched_quote_graph_from_raw
from research.nra3_s20.s48_frozen_manual_train36_source import (
    train36,digest,remove_gold_markers,
)
from research.nra3_s20.s48_train36_structural_quote_HELD26_cpu import (
    StructuralQuoteReadout,token_gold_from_manual,stats,
    TRAIN_SOURCE_COMMIT,TRAIN_SOURCE_BLOB,HELD_SOURCE_COMMIT,HELD_SOURCE_BLOB,
)


class S48QuoteGraphTrainOnlyTests(unittest.TestCase):
    def tokens(self,text):
        tok=raw_surface_grammar(text)
        r=matched_quote_graph_from_raw(text)
        return tok,r

    def test_balanced_pairs_and_nested_quotes(self):
        text="Учитель сказал «пример „внутреннее слово“ готов» сегодня."
        tok,res=self.tokens(text)
        claimed=[t.form for t,x in zip(tok,res.content) if x]
        self.assertEqual(claimed,["пример","внутреннее","слово","готов"])
        self.assertEqual(len(res.valid_outer_pairs),1)
        self.assertEqual(res.dangling_openers,0)
        self.assertEqual(res.ignored_apostrophes,0)

    def test_unclosed_outer_span_is_not_quote(self):
        for text in (
            "Текст содержит «забытый фрагмент до конца.",
            'Журнал содержит "незавершённую запись сегодня.',
            "Пометка осталась «фраза “внутренняя без завершения».",
            "Остаток записи “одна строка без окончания.",
        ):
            _,res=self.tokens(text)
            self.assertFalse(bool(res.content.any()),text)
            self.assertEqual(len(res.valid_outer_pairs),0)

    def test_completed_quote_before_orphan_remains(self):
        tok,res=self.tokens("Одна «готова», а вторая «пока незакрыта.")
        claimed=[t.form for t,x in zip(tok,res.content) if x]
        self.assertEqual(claimed,["готова"])
        self.assertEqual(res.dangling_openers,1)

    def test_single_curly_german_ascii_and_apostrophes(self):
        text="Слово O'Brien рядом с „важная заметка“ и ‘малая часть’."
        tok,res=self.tokens(text)
        claimed=[t.form for t,x in zip(tok,res.content) if x]
        self.assertEqual(claimed,["важная","заметка","малая","часть"])
        self.assertEqual(res.ignored_apostrophes,1)

    def test_escaped_quote_marks_are_literal(self):
        text='Файл содержит \\"ошибка\\" и «готово».'
        tok,res=self.tokens(text)
        claimed=[t.form for t,x in zip(tok,res.content) if x]
        self.assertEqual(claimed,["готово"])
        self.assertEqual(res.ignored_escaped_quote_marks,2)

    def test_gold_marker_parser_and_original_S48_train36(self):
        t,s=remove_gold_markers("Пишем «⟦готово⟧» сейчас.")
        self.assertEqual(t,"Пишем «готово» сейчас.")
        self.assertEqual(len(s),1)
        gold=token_gold_from_manual(raw_surface_grammar(t),s)
        self.assertEqual(int(gold.sum()),1)
        rows=train36()
        self.assertEqual(len(rows),36)
        self.assertEqual(len(set(r.text for r in rows)),36)
        self.assertEqual(len(digest()),64)
        self.assertEqual(TRAIN_SOURCE_COMMIT,"95592557e95965b149ee9b7780ceba164f93a7ff")
        self.assertEqual(TRAIN_SOURCE_BLOB,"ed6a76001aa1b593ddf2e7877af9ac9cbb59c416")
        self.assertEqual(HELD_SOURCE_COMMIT,"cbf5641e9ab4a36e7dddfe4b830cab0833ab7fe5")
        self.assertEqual(HELD_SOURCE_BLOB,"98812ce767720857817f3ecceeb36cb5d5408cff")
        for row in rows:
            tok=raw_surface_grammar(row.text)
            self.assertTrue(3<=len(tok)<=40,row.row_id)
            gold=token_gold_from_manual(tok,row.gold_char_spans)
            self.assertEqual(len(gold),len(tok))
            self.assertNotIn("⟦",row.text)

    def test_four_trainable_parameters_and_gradient(self):
        r=StructuralQuoteReadout()
        self.assertEqual(sum(p.numel() for p in r.parameters()),4)
        self.assertEqual(r.output_gate,0)
        self.assertFalse(r.candidate_accepted)
        self.assertFalse(r.activation_allowed)
        logits=torch.tensor([.5,-.5,.2])
        old=torch.tensor([1.,0.,0.])
        structural=torch.tensor([1.,0.,1.])
        prediction=r(logits,old,structural)
        loss=torch.nn.functional.softplus(-prediction[0])+torch.nn.functional.softplus(prediction[1])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in r.parameters()))
        self.assertEqual(stats(torch.tensor([True,False]),
                               torch.tensor([True,True])),{"TP":1,"FP":0,"FN":1})


if __name__=="__main__":
    unittest.main()
