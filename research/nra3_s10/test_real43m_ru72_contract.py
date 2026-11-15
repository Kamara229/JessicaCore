"""S10 offline contracts only; actual Calibri 43M measured in separate CPU CI.

Tests capture read-only layer8 hooks on a toy 12-layer structure; toy
representations must NEVER be published as genuine 43M artifacts.
"""
from __future__ import annotations

import hashlib
import unittest
import torch

from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort
from research.nra3_s5.test_frozen_layer8_read_port import ToyDecoder
from research.nra3_s9.new_ru_train72 import (
    S9_TRAIN72_CANONICAL_SHA256,build_s9_train72,canonical_train72,seal_train72,
)
from research.nra3_s9.locked_real_token_alignment import align_new_train72
from research.nra3_s4.test_russian_ud_train import CharFastTokenizer


class S10TrueCPUReadPortContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(310010)

    def test_exact_full_72_SOURCE_SHA_before_freezing_43M(self):
        data=build_s9_train72()
        self.assertEqual(len(data),72)
        self.assertEqual(
            hashlib.sha256(canonical_train72(data)).hexdigest(),
            S9_TRAIN72_CANONICAL_SHA256,
        )
        self.assertEqual(
            S9_TRAIN72_CANONICAL_SHA256,
            "b09d60a77a17d30c2dbf846e40ab97860def8635e2bf31a5b0ab8981832b744d",
        )
        manifest=seal_train72()
        self.assertEqual(manifest["additional_independent_eval_rows_used"],0)
        self.assertTrue(manifest["old_five_EVAL128_untouched"])
        self.assertTrue(manifest["reserved_sixth_S3CI_EVAL128_untouched"])
        self.assertFalse(manifest["real_frozen_43M_cpu_layer8_hidden320_captured"])
        self.assertFalse(manifest["candidate_accepted"])

    def test_readonly_hook_can_collect_72_distinct_prompts_no_gold(self):
        decoder=ToyDecoder()
        collector=FrozenLayer8ReadPort()
        tok=CharFastTokenizer()
        for row in build_s9_train72():
            a=align_new_train72(tok,row)
            n=a.tokenizer_token_count
            x=torch.randn(1,n,320)
            ids=list(a.prefill_token_ids)
            before=decoder(x)
            with collector.capture(decoder,row.text,ids):
                after=decoder(x)
            self.assertTrue(torch.equal(before,after))
        self.assertEqual(collector.observed_calls,72)
        rows=build_s9_train72()
        frozen=collector.frozen_batch(tuple(r.text for r in rows))
        self.assertEqual(tuple(frozen["token_hidden320"].shape[:1]),(72,))
        self.assertEqual(frozen["token_hidden320"].shape[-1],320)
        self.assertTrue(torch.all(frozen["token_lengths"]>len(rows[0].words)))
        self.assertEqual(frozen["token_mask"].dtype,torch.bool)
        self.assertEqual(collector.output_gate,0)
        self.assertFalse(collector.candidate_accepted)
        with self.assertRaises(AssertionError):
            collector.frozen_batch(tuple(r.text for r in rows[:-1]))

    def test_hook_rejects_invalid_source_count_and_unchanged_original_J7(self):
        # S10 read port operates only *after* separately confirmed S194 J7.
        # This test is structural and explicitly NOT another real J7 run.
        collector=FrozenLayer8ReadPort()
        decoder=ToyDecoder()
        source=build_s9_train72()[0]
        with self.assertRaises(RuntimeError):
            with collector.capture(decoder,source.text,[1,2,3]):
                pass
        self.assertEqual(collector.observed_calls,0)
        self.assertFalse(collector._active)


if __name__=="__main__":
    unittest.main()
