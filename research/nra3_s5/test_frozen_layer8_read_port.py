"""S5 CPU real-layer8 read-port structural tests. No fake Calibri J7 claims."""
import unittest

import torch
from torch import nn

from research.nra3_s4.russian_ud_train_only import (
    TRAIN12_CANONICAL_SHA256,build_train,commitment,
)
from research.nra3_s5.frozen_layer8_read_port import FrozenLayer8ReadPort


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(nn.Identity() for _ in range(12))

    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)
        return x


class S5FrozenReadPortTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(501501)

    def test_complete_S4_SHA_prereg_locked_for_real_CPU(self):
        source=commitment(build_train())
        self.assertEqual(source["canonical_training_rows_sha256"],
                         TRAIN12_CANONICAL_SHA256)
        self.assertEqual(TRAIN12_CANONICAL_SHA256,
                         "6ee1f270d413f4cfc79727bd48386ee29881520fcf6834309869733fd00f7a61")
        self.assertEqual(source["independent_EVAL_count"],0)
        self.assertFalse(source["QAT_training_performed"])

    def test_read_port_runs_after_existing_GT_and_leaves_output_object_exact(self):
        decoder=ToyDecoder()
        read=FrozenLayer8ReadPort()
        gt_scale=torch.tensor(0.015)
        # Emulate already installed original GT/S103 residual on layer8.
        other=decoder.model.layers[7].register_forward_hook(
            lambda _module,_args,y: y+gt_scale)
        try:
            a=torch.randn(1,6,320)
            baseline=decoder(a)
            self.assertTrue(torch.equal(baseline,a+gt_scale))
            q="Анна отправила Павлу письмо."
            with read.capture(decoder,q,[1,2,3,4,5,6]):
                new=decoder(a)
                self.assertTrue(torch.equal(new,baseline))
            saved=read.frozen_batch((q,))
            self.assertEqual(tuple(saved["token_hidden320"].shape),(1,6,320))
            self.assertTrue(torch.equal(
                saved["token_hidden320"][0],baseline[0].half()))
            self.assertEqual(saved["token_lengths"].tolist(),[6])
            self.assertEqual(saved["input_ids"][0].tolist(),[1,2,3,4,5,6])
            self.assertEqual(read.observed_calls,1)
            self.assertEqual(read.output_gate,0)
            self.assertFalse(read.candidate_accepted)
            with self.assertRaises(ValueError):
                with read.capture(decoder,q,[1,2,3,4,5,6]):
                    decoder(a)
            self.assertTrue(torch.equal(decoder(a),baseline))
        finally:
            other.remove()

    def test_reject_duplicate_calls_invalid_hidden_and_active_gate(self):
        decoder=ToyDecoder()
        read=FrozenLayer8ReadPort()
        q="Павел получил письмо от Анны."
        with self.assertRaises(RuntimeError):
            with read.capture(decoder,q,[1,2,3,4]):
                decoder(torch.randn(1,4,320))
                decoder(torch.randn(1,4,320))
        self.assertEqual(read.observed_calls,0)
        with self.assertRaises(ValueError):
            with read.capture(decoder,q,[1,2,3,4]):
                decoder(torch.randn(1,5,320))
        self.assertEqual(read.observed_calls,0)
        with read.capture(decoder,q,[1,2,3,4]):
            decoder(torch.randn(1,4,320))
        read.output_gate=0.2
        with self.assertRaises(ValueError):
            with read.capture(decoder,"Мария получила письмо.",[1,2,3]):
                decoder(torch.randn(1,3,320))

    def test_fail_closed_wrong_decoder_or_padding(self):
        read=FrozenLayer8ReadPort()
        with self.assertRaises(ValueError):
            with read.capture(nn.Identity(),"text",[1,2]):
                pass
        with self.assertRaises(ValueError):
            with read.capture(ToyDecoder(),"text",[0]):
                pass
        with self.assertRaises(AssertionError):
            read.frozen_batch(("text",))

if __name__=="__main__":
    unittest.main()
