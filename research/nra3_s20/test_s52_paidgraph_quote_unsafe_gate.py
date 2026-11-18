"""S52 structural CI unit tests; synthetic dummy features, NO actual 43M claim.

Real genuine Calibri43M+GT and paid S26 SHA-pinned CPU operation is tested
in a separate S52 GitHub workflow using physically archived S51 and S26.
No protected S38, S47, S48 HELD26 or independent EVAL gold is loaded.
"""
from __future__ import annotations
import unittest
import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s19.paid_grammar_parser import CompactWordGrammar
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s20.s26_predicted_event_inversion_rescue import EventGroundedInversionGraph
from research.nra3_s20.s48_matched_unicode_quote_graph import matched_quote_graph_from_raw
from research.nra3_s20.s52_paidgraph_native_quote_bpe_abstain import (
    S52QuoteAwareFrozenPaidGraphBridge,prediction_totals,tensor_sha,
)


def dummy_native339(texts):
    words=[raw_surface_grammar(t) for t in texts]
    max_w=max(len(seq) for seq in words)
    x=torch.zeros((len(texts),max_w,339),dtype=torch.float32)
    valid=torch.zeros((len(texts),max_w),dtype=torch.bool)
    for row,(text,tokens) in enumerate(zip(texts,words)):
        raw_quote=matched_quote_graph_from_raw(text).content
        valid[row,:len(tokens)]=True
        for j,t in enumerate(tokens):
            x[row,j,:320]=.01*(row+1)+torch.arange(320)*.00001+(j*.0001)
            x[row,j,320:338]=torch.tensor(t.features,dtype=torch.float32)
            x[row,j,338]=raw_quote[j]
    return x,valid,words


def untrained_shadow_for_geometry_tests():
    torch.manual_seed(521118)
    parent=PaidContextualEventScope(CompactWordGrammar())
    model=EventGroundedInversionGraph(parent)
    return model.eval().requires_grad_(False)


class S52OriginalPaidGraphSafetyContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=untrained_shadow_for_geometry_tests()

    def test_original_five_heads_bitwise_unchanged_quote_and_unsafe_mask(self):
        texts=("Нина сказала «важное объявление» и ушла.",
               "Олег не отправил заказ сегодня.")
        x,valid,tokens=dummy_native339(texts)
        unsafe=torch.zeros_like(valid)
        unsafe[0,0]=True
        digest=tensor_sha(x,valid,unsafe)
        with torch.inference_mode():
            before=self.model(x[...,:338],valid)
        observer=S52QuoteAwareFrozenPaidGraphBridge(self.model)
        after=observer.observe(x,valid,unsafe)
        for name in ("event_status","role","argument_polarity",
                     "coreference","event_relation"):
            self.assertTrue(torch.equal(getattr(before,name),
                                        getattr(after.original_paid_graph_logits,name)))
        self.assertEqual(digest,tensor_sha(x,valid,unsafe))
        self.assertTrue(bool((after.eligible_word_mask&unsafe).sum()) is False)
        self.assertFalse(bool((after.eligible_word_mask&after.quote_word_mask).any()))
        self.assertFalse(bool((after.eligible_word_mask&after.punctuation_mask).any()))
        self.assertEqual(int((after.eligible_directed_pair_mask.diagonal(dim1=1,dim2=2)).sum()),0)
        self.assertEqual(observer.output_gate,0)
        self.assertFalse(observer.candidate_accepted)
        self.assertFalse(observer.activation_allowed)

    def test_quote_content_not_treated_as_direct_assertion(self):
        x,valid,tokens=dummy_native339(("Борис сказал «сегодня не выдавать паспорт» и вышел.",))
        observer=S52QuoteAwareFrozenPaidGraphBridge(self.model)
        out=observer.observe(x,valid,torch.zeros_like(valid))
        quoted=[t.form for j,t in enumerate(tokens[0]) if out.quote_word_mask[0,j]]
        self.assertIn("выдавать",quoted)
        self.assertNotIn("сказал",quoted)
        for j,t in enumerate(tokens[0]):
            if out.quote_word_mask[0,j]:
                self.assertFalse(bool(out.eligible_word_mask[0,j]))
                self.assertFalse(bool(out.eligible_directed_pair_mask[0,j].any()))
                self.assertFalse(bool(out.eligible_directed_pair_mask[0,:,j].any()))

    def test_prediction_totals_only_counts_not_accuracy(self):
        x,valid,_=dummy_native339(("Игорь посмотрел «план» и пошёл.",))
        observed=S52QuoteAwareFrozenPaidGraphBridge(self.model).observe(
            x,valid,torch.zeros_like(valid))
        metrics=prediction_totals(observed)
        self.assertEqual(set(metrics),{
            "event","role","argument_REJECTED","coreference","event_relation"})
        for row in metrics.values():
            self.assertGreaterEqual(row["original_frozen_S26_argmax_nonNONE_unvalidated"],
                                    row["structurally_eligible_unvalidated"])
            self.assertEqual(row["abstained_due_to_quote_BPE_or_punctuation"],
                row["original_frozen_S26_argmax_nonNONE_unvalidated"]
                -row["structurally_eligible_unvalidated"])

    def test_nonzero_gate_disallowed(self):
        x,valid,_=dummy_native339(("Мастер сказал «готово» вчера.",))
        observer=S52QuoteAwareFrozenPaidGraphBridge(self.model)
        observer.output_gate=1
        with self.assertRaises(ValueError):
            observer.observe(x,valid,torch.zeros_like(valid))
        observer.output_gate=0
        observer.candidate_accepted=True
        with self.assertRaises(ValueError):
            observer.observe(x,valid,torch.zeros_like(valid))

    def test_invalid_hidden_unsafe_padding_and_quote_nonbinary_rejected(self):
        x,valid,_=dummy_native339(("Мастер сказал «готово» вчера.",))
        observer=S52QuoteAwareFrozenPaidGraphBridge(self.model)
        unsafe=torch.zeros_like(valid)
        bad=x.clone()
        bad[0,0,338]=.25
        with self.assertRaises(ValueError):
            observer.observe(bad,valid,unsafe)
        with self.assertRaises(ValueError):
            observer.observe(x[...,:338],valid,unsafe)
        with self.assertRaises(ValueError):
            observer.observe(x,valid,unsafe.float())

    def test_train_not_allowed_and_masked_model_never_has_gradient(self):
        x,valid,_=dummy_native339(("Письмо содержит «важное сообщение» сегодня.",))
        observer=S52QuoteAwareFrozenPaidGraphBridge(self.model)
        out=observer.observe(x,valid,torch.zeros_like(valid))
        self.assertFalse(out.original_paid_graph_logits.event_status.requires_grad)
        self.assertTrue(all(not p.requires_grad for p in observer.model.parameters()))
        observer.training_allowed=True
        with self.assertRaises(ValueError):
            observer.observe(x,valid,torch.zeros_like(valid))


if __name__=="__main__":
    unittest.main()
