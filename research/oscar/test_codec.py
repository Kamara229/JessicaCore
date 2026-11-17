"""Native OSCAR unit tests, no network, original model or sealed QA needed."""
from __future__ import annotations
import unittest

import numpy as np
import zlib
import torch
from torch import nn

from direct_codebook_v59 import CB59,unpack5
from oscar.codec import OscarConverter
from oscar.pack import bundle,cb61_pack,cb61_unpack


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(64,16)
        self.lm_head=nn.Linear(16,64,bias=False)
        self.lm_head.weight=self.embed.weight
        self.layer=nn.Linear(16,32,bias=False)
        self.norm=nn.LayerNorm(16)


def decoded_mirror(archive:CB59,source:TinyModel):
    clone=TinyModel()
    with torch.no_grad():
        for name,param in clone.state_dict().items():
            if name in archive.handles:
                h=archive.handles[name]
                num=h.shape[0]*h.shape[1]//8
                idx=unpack5(h.packed.tobytes(),num)
                arr=(h.centroids[idx].reshape(h.shape)*
                     h.row_scales[:,None])
            else:
                arr=archive.raw[name]
            param.copy_(torch.tensor(np.asarray(arr).copy(),dtype=torch.float32))
    assert clone.embed.weight is clone.lm_head.weight
    return clone


class OscarCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        cls.model=TinyModel().eval()
        cls.fresh=OscarConverter(mode="cold").convert(
            cls.model.state_dict(),layer_count=1,derived_from_calibri=False)
        cls.prior=CB59(cls.fresh.raw_cb59)

    def test_cold_complete_paid_checkpoint_and_alias(self):
        c=self.fresh
        self.assertEqual(c.mode,"cold")
        self.assertEqual(c.matrices,2)
        self.assertGreater(c.code_groups,0)
        self.assertEqual(c.source_unique_params,sum(x.numel() for x in self.model.parameters()))
        self.assertIs(self.prior.handles["embed.weight"],self.prior.handles["lm_head.weight"])
        self.assertTrue(c.summary()["all_5bit_vector_codes_assigned_from_FP32"])

    def test_inherited_deterministic_FP32_mirror_exact_physical(self):
        model=decoded_mirror(self.prior,self.model)
        result=OscarConverter(mode="inherited",prior=self.prior,
                              tolerance=2e-4).convert(
                                  model.state_dict(),layer_count=1,
                                  derived_from_calibri=True)
        self.assertEqual(self.fresh.raw_cb59,result.raw_cb59)
        self.assertEqual(result.sha256,self.fresh.sha256)
        self.assertEqual(result.code_groups,self.fresh.code_groups)
        for row in result.stats.values():
            self.assertLessEqual(row["weight_max_abs"],2e-4)

    def test_inherited_actually_reads_changed_FP32_not_old_indices(self):
        model=decoded_mirror(self.prior,self.model)
        with torch.no_grad():
            model.layer.weight[0,:8].add_(20.)
            model.norm.bias[0].add_(0.15)
        cand=OscarConverter(mode="inherited",prior=self.prior).convert(
            model.state_dict(),layer_count=1,derived_from_calibri=False)
        self.assertNotEqual(cand.raw_cb59,self.fresh.raw_cb59)
        self.assertEqual(cand.physical_bytes,self.fresh.physical_bytes)
        after=CB59(cand.raw_cb59)
        self.assertTrue(torch.allclose(
            torch.from_numpy(after.raw["norm.bias"].copy()),
            model.norm.bias.detach(),atol=1e-6))
        self.assertTrue(np.array_equal(
            self.prior.handles["layer.weight"].centroids,
            after.handles["layer.weight"].centroids))
        self.assertTrue(np.array_equal(
            self.prior.handles["layer.weight"].row_scales,
            after.handles["layer.weight"].row_scales))

    def test_tied_alias_identity_fail_closed(self):
        model=decoded_mirror(self.prior,self.model)
        d=dict(model.state_dict())
        d["lm_head.weight"]=d["lm_head.weight"].clone()
        with self.assertRaisesRegex(ValueError,"tied"):
            OscarConverter(mode="inherited",prior=self.prior).convert(d)

    def test_inherited_requires_real_prior(self):
        with self.assertRaises(ValueError):
            OscarConverter(mode="inherited")
        with self.assertRaises(ValueError):
            OscarConverter(mode="cold",prior=self.prior)
        with self.assertRaises(ValueError):
            OscarConverter(mode="any")

    def test_CB59_CB61_and_corrupt_hash(self):
        c=cb61_pack(self.fresh.raw_cb59)
        self.assertEqual(cb61_unpack(c),self.fresh.raw_cb59)
        broken=bytearray(c)
        broken[-3]^=0x40
        with self.assertRaises((ValueError,zlib.error)):
            cb61_unpack(bytes(broken))
        broken=bytearray(self.fresh.raw_cb59)
        broken[-2]^=0x20
        with self.assertRaises(ValueError):
            CB59(bytes(broken))

    def test_executable_FP32_safetensors_inherited_roundtrip(self):
        import tempfile
        import json
        from pathlib import Path
        from safetensors.torch import save_file
        from oscar.__main__ import main
        model=decoded_mirror(self.prior,self.model)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            checkpoint=root/"dense_FP32.safetensors"
            prior_file=root/"actual_trained_prior.cb59"
            # The original canonical tied weight is persisted ONCE; Oscar
            # must reconstruct the alias from the authenticated prior.
            tensors={name:value.detach().cpu().contiguous()
                     for name,value in model.state_dict().items()
                     if name!="lm_head.weight"}
            save_file(tensors,str(checkpoint))
            prior_file.write_bytes(self.prior.blob)
            out=root/"oscar_result"
            main(["--fp32",str(checkpoint),
                  "--prior",str(prior_file),
                  "--mode","inherited",
                  "--depth","1",
                  "--output",str(out),
                  "--derived-from-calibri"])
            from oscar.pack import cb61_unpack
            self.assertEqual(
                cb61_unpack((out/"Oscar_full_converted_CB61.s61").read_bytes()),
                self.prior.blob)
            report=json.loads(
                (out/"Oscar_convert_FP32_actual_physical_report.json").read_text())
            self.assertEqual(report["conversion"]["full_original_2D_matrices"],2)
            self.assertFalse(report["model_quality_accepted"])

    def test_full_bundle_measures_literal_disk_bytes(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/"oscar_candidate.zip"
            c=cb61_pack(self.fresh.raw_cb59)
            s={"neurons.safetensors":b"test-tensor-bytes",
               "gt.bin":b"full12-dummy-bits"}
            report=bundle(f,c,s,
                     source_params=self.fresh.source_unique_params,
                     nominal_params=self.fresh.source_unique_params+10000,
                     metadata={"test_only":True})
            self.assertEqual(report.bytes,f.stat().st_size)
            self.assertTrue(report.nominal_bpp<report.conservative_core_bpp)
            self.assertEqual(report.auxiliary_bytes["gt.bin"],17)
            self.assertTrue(report.as_dict()["all_components_shipped_once"])


if __name__=="__main__":
    unittest.main()
