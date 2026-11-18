"""S49 original 43M GT zero-gate read-port contracts on synthetic ABI ONLY.

No S38/S47/EVAL gold. Physical 43M S22c native data is checked separately
in the GitHub Actions runner by S49 CPU evaluator. Unit tests cannot
substitute for original 43M+GT capture and do not claim it.
"""
from __future__ import annotations
import unittest
import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s49_real43m_gt_quote_sidecar_zero_gate import (
    Frozen43MGreenTunnelQuotePort,
    frozen_native_manifest_digest,
)


def sample_original_native_word338(texts):
    words=[raw_surface_grammar(t) for t in texts]
    w=max(len(row) for row in words)
    x=torch.zeros(len(texts),w,338,dtype=torch.float32)
    valid=torch.zeros(len(texts),w,dtype=torch.bool)
    for i,seq in enumerate(words):
        n=len(seq)
        valid[i,:n]=True
        for j,tok in enumerate(seq):
            x[i,j,:320]=torch.arange(320,dtype=torch.float32)*.0001+(i+j+1)*.001
            x[i,j,320:]=torch.tensor(tok.features,dtype=torch.float32)
    return x,valid


class S49ZeroGateNativeReadTests(unittest.TestCase):
    def test_original_338_tensor_same_object_data_ptr_and_sha(self):
        texts=("Дежурный сказал «готово» и вышел.",
               "Инспектор проверил “текст” вчера.")
        native,mask=sample_original_native_word338(texts)
        digest=frozen_native_manifest_digest(native,mask)
        port=Frozen43MGreenTunnelQuotePort()
        self.assertEqual(port.original_layer_zero_based,7)
        self.assertEqual(port.output_gate,0)
        self.assertFalse(port.candidate_accepted)
        self.assertFalse(port.activation_allowed)
        self.assertFalse(port.training_allowed)
        out=port.forward_original_unchanged(native,mask,texts)
        self.assertIs(out,native)
        self.assertEqual(out.data_ptr(),native.data_ptr())
        augmented,aux=port.shadow_augmented_features_only(native,mask,texts)
        self.assertEqual(tuple(augmented.shape),(*native.shape[:2],339))
        self.assertTrue(torch.equal(augmented[...,:338],native))
        self.assertEqual(tuple(aux.per_row_quote_tokens),(1,1))
        self.assertEqual(tuple(aux.per_row_matched_pairs),(1,1))
        self.assertTrue(torch.equal(aux.detached_original_hidden320,native[...,:320]))
        self.assertEqual(frozen_native_manifest_digest(native,mask),digest)
        self.assertTrue(torch.equal(augmented[~mask],torch.zeros_like(augmented[~mask])))

    def test_apostrophes_and_unclosed_spans_not_quote(self):
        texts=("Список содержит O'Brien и “согласовано”.",
               "Архив хранит «незавершённую заметку без конца.")
        x,v=sample_original_native_word338(texts)
        _,aux=Frozen43MGreenTunnelQuotePort().shadow_augmented_features_only(x,v,texts)
        self.assertEqual(tuple(aux.per_row_quote_tokens),(1,0))
        self.assertEqual(tuple(aux.per_row_matched_pairs),(1,0))
        self.assertEqual(tuple(aux.per_row_unmatched_openers),(0,1))

    def test_activation_denied_and_original_source_immutable(self):
        texts=("Токен рядом с «цитатой».",)
        x,v=sample_original_native_word338(texts)
        p=Frozen43MGreenTunnelQuotePort()
        p.output_gate=1
        with self.assertRaisesRegex(ValueError,"activated or trained"):
            p.forward_original_unchanged(x,v,texts)
        p.output_gate=0
        p.candidate_accepted=True
        with self.assertRaises(ValueError):
            p.observe(x,v,texts)
        p.candidate_accepted=False
        p.training_allowed=True
        with self.assertRaises(ValueError):
            p.observe(x,v,texts)

    def test_wrong_hidden_dimensions_and_goldlike_raw18_fail_closed(self):
        texts=("Автор сказал «совет» прямо.",)
        x,v=sample_original_native_word338(texts)
        p=Frozen43MGreenTunnelQuotePort()
        with self.assertRaises(ValueError):
            p.observe(x[...,:337].clone(),v,texts)
        corrupted=x.clone()
        corrupted[0,0,320:]=.25
        with self.assertRaises(ValueError):
            p.observe(corrupted,v,texts)
        altered=x.clone()
        altered[0,0,320]=1-altered[0,0,320]
        with self.assertRaisesRegex(ValueError,"lex alignment"):
            p.observe(altered,v,texts)
        with self.assertRaises(ValueError):
            p.observe(x,~v,texts)

    def test_unsupported_long_text_fails_closed_without_core_change(self):
        texts=(" ".join(["слово"]*42)+".",)
        x,v=sample_original_native_word338(texts)
        p=Frozen43MGreenTunnelQuotePort()
        before=frozen_native_manifest_digest(x,v)
        native,aux=p.shadow_augmented_features_only(x,v,texts)
        self.assertEqual(aux.unsupported_row_ids,(0,))
        self.assertEqual(tuple(aux.per_row_quote_tokens),(0,))
        self.assertFalse(bool(aux.supported_mask.any()))
        self.assertTrue(torch.equal(native[...,:338],x))
        self.assertEqual(before,frozen_native_manifest_digest(x,v))


if __name__=="__main__":
    unittest.main()
