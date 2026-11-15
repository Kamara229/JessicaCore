"""CPU-only S1 tests: genuine CB59 bytes, frozen GT hook order, no EVAL."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from research.nra3_s0.neural_area import NRA3Config, NRA3NeuralArea
from research.nra3_s1.physical_cb59 import (
    encode_area, reconstruct_state, save_and_validate, budget_for_paid_components,
)
from research.nra3_s1.green_tunnel_compatibility import (
    NRA3WithExistingGreenTunnel, module_fingerprint,
)
from calibri2.s87.internal_green_tunnel import IntegratedGreenTunnel
from direct_codebook_v59 import pack5, unpack5
import numpy as np
from calibri2.s103.shared_all_layer_green_bridge import SharedInterLayerGreenBridge


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(12)])

    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)
        return x


class NRA3PackedAndGTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(309117)

    @staticmethod
    def small():
        return NRA3NeuralArea(NRA3Config(
            hidden=32,heads=4,grammar_blocks=1,semantic_blocks=1,
            relation_width=24,relation_steps=2)).eval()

    def test_original_CB59_bitstream_and_new_unaligned_group_counts(self):
        for count in (1, 4, 7, 8, 9, 12, 17, 33, 64):
            ids=(np.arange(count,dtype=np.uint8)*7)%32
            actual=pack5(ids)
            self.assertEqual(len(actual),(count*5+7)//8)
            self.assertTrue(np.array_equal(unpack5(actual,count),ids))
        # Historical eight-code CB59 byte packing is unchanged.
        source=np.arange(8,dtype=np.uint8)
        self.assertEqual(len(pack5(source)),5)
        self.assertTrue(np.array_equal(unpack5(pack5(source),8),source))

    def test_exact_full_CB59_physical_roundtrip_and_corruption(self):
        area=self.small()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"NRA3_S1_REAL_COMPLETE.cb59"
            report=save_and_validate(area,p)
            self.assertGreater(report["complete_new_area_bytes"],0)
            self.assertGreater(report["matrix_tensor_count"],10)
            self.assertEqual(report["unique_new_area_parameters"],
                             area.parameter_count())
            self.assertEqual(report["output_gate"],0)
            self.assertFalse(report["candidate_accepted"])
            self.assertFalse(report["native_packed_CPU_kernel_verified"])
            self.assertEqual(report["actual_new_area_bpp"],
                             8*p.stat().st_size/area.parameter_count())
            restored=reconstruct_state(p.read_bytes(),area)
            self.assertEqual(set(restored),set(area.state_dict()))
            area2=self.small()
            area2.load_state_dict(restored,strict=True)
            x=torch.randn(1,3,320)
            with torch.inference_mode():
                original=area(x)
                packed=area2(x)
            self.assertEqual(tuple(packed.relations.shape),(1,3,24))
            self.assertTrue(torch.isfinite(packed.relations).all())
            # The quantization quality may be poor; do NOT claim parity
            # merely because the physically decoded model runs.
            self.assertEqual(original.relations.shape,packed.relations.shape)
            with self.assertRaises(ValueError):
                save_and_validate(area,p)
            corrupted=bytearray(p.read_bytes())
            corrupted[len(corrupted)//2]^=1
            with self.assertRaises(ValueError):
                reconstruct_state(bytes(corrupted),area)
            b=budget_for_paid_components(
                parent_physical_bytes=b"parent"*2000,
                parent_unique_parameters=100000,
                nra_physical_bytes=p.read_bytes(),
                existing_gt_physical_bytes=b"GT-SIDE-REAL-CONTENT",
                parent_includes_gt=False,
                package_manifest_bytes=b"manifest")
            self.assertEqual(b["whole_package_bytes"],
                             12000+p.stat().st_size+len(b"GT-SIDE-REAL-CONTENT")+8)
            self.assertTrue(b["green_tunnel_accounted_exactly_once"])
            with self.assertRaises(ValueError):
                budget_for_paid_components(
                    parent_physical_bytes=b"parent"*2000,
                    parent_unique_parameters=100000,
                    nra_physical_bytes=p.read_bytes(),
                    existing_gt_physical_bytes=b"GT-SIDE-REAL-CONTENT",
                    parent_includes_gt=True)

    def test_midlayer_observation_after_real_GT_and_all_layer_bridge(self):
        model=ToyDecoder()
        tunnel=IntegratedGreenTunnel(width=320,rank=8,layers=(3,9))
        with torch.no_grad():
            tunnel.blocks["3"].up.weight.fill_(0.02)
            tunnel.blocks["9"].up.weight.fill_(0.01)
        tunnel.install(model)
        bridge=SharedInterLayerGreenBridge(tunnel).install(model)
        with torch.no_grad():
            bridge.layer_gains[bridge.additional_layers.index(7)]=0.23
        gt_sha=module_fingerprint(tunnel)
        bridge_sha=module_fingerprint(bridge)
        port=NRA3WithExistingGreenTunnel(self.small(),green_tunnel=tunnel,
                                          shared_bridge=bridge).install(model)
        x=torch.randn(1,4,320)
        try:
            with torch.inference_mode():
                for route in (None,"S87","S91","S93"):
                    with bridge.profile(route):
                        baseline=model(x)
                        with port.port.activated():
                            after=model(x)
                        self.assertTrue(torch.equal(after,baseline),
                                        f"GT route {route}")
                        self.assertEqual(port.port.last_shape,(1,4,320))
                        self.assertEqual(port.port.last_features["relation"],(1,4,24))
            self.assertEqual(gt_sha,module_fingerprint(tunnel))
            self.assertEqual(bridge_sha,module_fingerprint(bridge))
            self.assertTrue(port.integrity()["existing_green_tunnel_unchanged"])
            with self.assertRaises(RuntimeError):
                port.port.set_output_gate(0.01)
            with torch.no_grad():
                bridge.layer_gains[0]+=1
            with self.assertRaises(RuntimeError):
                port.integrity()
            with torch.no_grad():
                bridge.layer_gains[0]-=1
        finally:
            port.uninstall()
            bridge.uninstall()
            tunnel.uninstall()

    def test_fail_closed_missing_green_tunnel(self):
        model=ToyDecoder()
        with self.assertRaises(ValueError):
            NRA3WithExistingGreenTunnel(self.small(),
                green_tunnel=IntegratedGreenTunnel()).install(model)


if __name__=="__main__":
    unittest.main()
