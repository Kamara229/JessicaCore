"""CPU-only tests. Zero real S101 weights, zero QA. Physical codec proofs."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from safetensors.torch import load_file,save_file
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import (
    export_compact_q2,load_compact_q2,physical_raw_bytes,PREFIX,
    _encode_matrix_q2,_decode_matrix_q2,
)
from calibri2.s126.test_full_layer_state_tunnel import TwelveLayerModel


class TestCompactGT(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1264)
        self.gt=FullLayerStateTunnel(width=320,depth=12,rank=4)

    def test_roundtrip_original_12_layer_zero_exact(self):
        original=TwelveLayerModel().eval()
        x=torch.randn(1,3,320)
        expected=original(x)
        with TemporaryDirectory() as folder:
            packed=Path(folder)/"compact.safetensors"
            original_payload=export_compact_q2(self.gt)
            save_file(original_payload,str(packed))
            reloaded=load_file(str(packed))
            rank4=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
            load_compact_q2(rank4,reloaded)
            rebuilt=export_compact_q2(rank4)
            self.assertEqual(set(reloaded),set(rebuilt))
            self.assertTrue(all(torch.equal(reloaded[k],rebuilt[k])
                                for k in reloaded))
            self.assertLessEqual(physical_raw_bytes(reloaded),700)
            self.assertLess(packed.stat().st_size,1500)
            rank4.install(original)
            with rank4.scope():
                output=original(x)
            self.assertTrue(torch.equal(expected,output))
            self.assertEqual(rank4.last_forward_coverage,tuple(range(12)))
            rank4.uninstall()

    def test_gradients_after_quantized_reload(self):
        packed=export_compact_q2(self.gt)
        load_compact_q2(self.gt,packed)
        self.gt.gates.data.fill_(0.1)
        model=TwelveLayerModel()
        self.gt.install(model)
        with self.gt.scope():
            x=torch.randn(1,2,320)
            result=model(x).square().mean()
        result.backward()
        self.assertTrue(bool(torch.isfinite(self.gt.encoder.grad).all()))
        self.assertTrue(bool(torch.isfinite(self.gt.decoder.grad).all()))
        self.assertTrue(bool(torch.isfinite(self.gt.gates.grad).all()))
        self.assertEqual(self.gt.new_parameter_count(),2584)
        self.gt.uninstall()

    def test_invalid_physical_rejected(self):
        payload=export_compact_q2(self.gt)
        invalid=dict(payload)
        invalid.pop(PREFIX+"encoder_codes")
        with self.assertRaises(ValueError):
            load_compact_q2(self.gt,invalid)
        invalid=dict(payload)
        invalid[PREFIX+"decoder_codes"]=torch.zeros((1,),dtype=torch.uint8)
        with self.assertRaises(ValueError):
            load_compact_q2(self.gt,invalid)
        invalid=dict(payload)
        invalid[PREFIX+"encoder_scale"]=torch.zeros(1,dtype=torch.float16)
        with self.assertRaises(ValueError):
            load_compact_q2(self.gt,invalid)

    def test_bit_order_reversible_and_quant_error_bounded(self):
        x=torch.randn(4,320)*0.02
        code,scale=_encode_matrix_q2(x)
        decoded=_decode_matrix_q2(code,scale,(4,320))
        self.assertEqual(code.numel(),320)
        self.assertLessEqual(float((decoded-x).abs().max()),
                             float(scale[0])+1e-4)
        self.assertLess(float((decoded-x).square().mean()),0.0004)


if __name__=="__main__":
    unittest.main()
