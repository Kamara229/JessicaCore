import tempfile
import unittest
import hashlib
import json
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import LlamaConfig, LlamaForCausalLM

from direct_codebook_v59 import CB59, encode
from direct_runtime_v59 import NativeKernel, bind_into_llama, compile_kernel
from v94.generation import generate_one
from v108.generation import assistant_turn
from v108.pipeline import locked_sources
from binary_geometry_v64 import MODEL_SHA
from v109.feedback import recode_rows


class Tokens:
    eos_token_id = 2

    def encode(self, prompt, **kwargs):
        return [4, 5]

    def decode(self, ids, **kwargs):
        return " ".join(str(x) for x in ids if x != self.eos_token_id)


class RuntimeTest(unittest.TestCase):
    def test_stop_and_fixed_benchmark(self):
        tok = Tokens()

        def run(sequence, stop):
            steps = iter(sequence)

            def model(**kwargs):
                logits = torch.zeros(1, 1, 12)
                logits[0, 0, next(steps)] = 1
                return SimpleNamespace(logits=logits, past_key_values=())

            return generate_one(model, tok, "x", stop_at_eos=stop)

        self.assertEqual(run([2], True)["ids"], [2])
        self.assertEqual(run([8, 2], True)["full_model_token_steps"], 2)
        self.assertEqual(run([8] * 32, True)["ids"], [8] * 32)
        fixed = run([8, 2] + [7] * 30, False)
        self.assertEqual(fixed["full_model_token_steps"], 32)
        self.assertEqual(assistant_turn(tok, fixed["ids"]),
                         {"text": "8", "content_tokens": 1, "ended_by_eos": True,
                          "tokens_after_eos": 30})

    def test_real_meta_binding_matches_dense_initialization(self):
        torch.manual_seed(109)
        cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                          num_attention_heads=2, num_key_value_heads=2, num_hidden_layers=1,
                          tie_word_embeddings=True, pad_token_id=0)
        cfg._attn_implementation = "eager"
        dense = LlamaForCausalLM(cfg).eval()
        with torch.no_grad():
            dense.model.norm.weight.fill_(1.375)
        archive = CB59(encode(dense.state_dict())[0])
        with init_empty_weights(include_buffers=False):
            empty = LlamaForCausalLM(cfg).eval()
        self.assertTrue(all(p.is_meta for p in empty.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            lib = compile_kernel(Path(__file__).with_name("direct_kernel_v59.c"), Path(folder) / "test.so")
            kernel = NativeKernel(lib)
            for model in (dense, empty):
                bind_into_llama(model, archive, kernel)
                self.assertIs(model.lm_head.handle, model.model.embed_tokens.handle)
                self.assertFalse(any(p.is_meta or p.ndim >= 2 for p in model.parameters()))
            ids = torch.tensor([[3, 8, 2]])
            with torch.inference_mode():
                torch.testing.assert_close(dense(ids).logits, empty(ids).logits, atol=0, rtol=0)

    def test_compensating_across_groups(self):
        # Both groups round down separately, but together overshoot the desired sum error.
        book = np.array([[0.], [1.]], dtype=np.float32)
        target = np.array([[.45, .45]], dtype=np.float32)
        basis = np.ones((2, 1), dtype=np.float32) / np.sqrt(np.float32(2))
        initial = np.zeros((1, 2), dtype=np.uint8)
        codes, before, after = recode_rows(target, book, initial, basis)
        self.assertEqual(int(codes.sum()), 1)
        self.assertLess(after, before)
        self.assertFalse(initial.any())

    def test_replacing_both_v102_bytes_and_their_report_cannot_change_the_reference(self):
        digest = lambda data: hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            base, other, report, lock = [Path(tmp) / n for n in ("base", "other", "report", "lock")]
            base.write_bytes(b"old")
            other.write_bytes(b"new")
            lock.write_text(json.dumps({"contract_version": "CALIBRI_BASELINE_LOCK_2026-11-04",
                "original_model_revision": MODEL_SHA,
                "frozen_reference_sources": {"CB59": {"sha256": digest(b"old"), "complete_archive_bytes": 3}}}))
            report.write_text(json.dumps({"new_separate_source_sha256": digest(b"new"),
                                          "original_immutable_reference_sha256": digest(b"old")}))
            with patch("v108.pipeline.CB59") as parser:
                with self.assertRaisesRegex(AssertionError, "lock"):
                    locked_sources(base, other, report, lock)
                parser.assert_not_called()


if __name__ == "__main__":
    unittest.main()
