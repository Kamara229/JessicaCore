"""Oscar independent FP32 input, full source/CRC and warm inversion safety."""
from __future__ import annotations
import unittest

import numpy as np
import torch
from torch import nn

from direct_codebook_v59 import CB59, encode, unpack5
from oscar.encoder import (
    MODE_ADAPT, MODE_COLD, MODE_LOCKED, encode_fp32, to_cb61,
)


class ToyCheckpoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(64,16)
        self.dense=nn.Linear(16,16,bias=True)
        self.head=nn.Linear(16,64,bias=False)
        self.head.weight=self.embed.weight


def decoded_matrix(h):
    m,n=h.shape
    code=unpack5(h.packed.tobytes(),m*(n//8))
    return h.centroids[code].reshape(m,n).astype(np.float32)*h.row_scales[:,None]


def reify_original_into_FP32(reference):
    toy=ToyCheckpoint()
    state=toy.state_dict()
    with torch.no_grad():
        for name,t in state.items():
            if name in reference.handles:
                payload=decoded_matrix(reference.handles[name])
            else:
                payload=reference.raw[name]
            t.copy_(torch.from_numpy(np.array(payload,copy=True)).float())
    if toy.embed.weight.data_ptr()!=toy.head.weight.data_ptr():
        raise AssertionError("Oscar synthetic trained FP32 mirror broke tied source")
    return toy.state_dict()


class OscarCodecContract(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7041)
        self.toy=ToyCheckpoint().eval()
        self.warm_old=encode_fp32(self.toy.state_dict(),mode=MODE_COLD)
        self.original=self.warm_old.authenticated_decoder()

    def test_cold_from_arbitrary_FP32_requires_no_prior(self):
        ans=encode_fp32(self.toy.state_dict(),mode=MODE_COLD)
        decoder=ans.authenticated_decoder()
        self.assertEqual(ans.report["prior_used"],False)
        self.assertEqual(ans.report["matrix_count"],2)
        self.assertEqual(decoder.meta["original_unique_parameter_count"],
                         sum(p.numel() for p in self.toy.parameters()))
        self.assertIs(decoder.handles["head.weight"],
                      decoder.handles["embed.weight"])
        self.assertEqual(ans.blob,self.warm_old.blob)

    def test_warm_inverse_exact_derived_FP32_not_old_indices(self):
        independent_input=reify_original_into_FP32(self.original)
        encoded=encode_fp32(independent_input,mode=MODE_LOCKED,
                            prior_blob=self.warm_old.blob)
        new=encoded.authenticated_decoder()
        self.assertEqual(encoded.report["all_old_packed_prior_indices_excluded"],True)
        self.assertEqual(encoded.report["complete_2D_matrices"],2)
        for canonical in self.original.meta["trained_unique_keys"]:
            if canonical not in self.original.handles:
                continue
            old=decoded_matrix(self.original.handles[canonical])
            fresh=decoded_matrix(new.handles[canonical])
            self.assertTrue(np.allclose(old,fresh,rtol=0,atol=2e-4),canonical)
        self.assertEqual(encoded.report["new_original_8weight_groups_encoded"],
                         64*(16//8)+16*(16//8))

    def test_new_FP32_not_hidden_prior_1d_and_trainable_codebooks(self):
        torch.manual_seed(9042)
        novel=ToyCheckpoint()
        assert not torch.allclose(novel.embed.weight,self.toy.embed.weight)
        with torch.no_grad():
            novel.dense.bias.add_(100.0)
        for mode in (MODE_LOCKED,MODE_ADAPT):
            fresh=encode_fp32(novel.state_dict(),mode=mode,
                              prior_blob=self.warm_old.blob,
                              adapt_samples=128,adapt_steps=3,prior_anchor=.15)
            out=fresh.authenticated_decoder()
            self.assertNotEqual(fresh.blob,self.warm_old.blob)
            self.assertTrue(np.array_equal(out.raw["dense.bias"],
                novel.dense.bias.detach().numpy()))
            self.assertIs(out.handles["head.weight"],
                          out.handles["embed.weight"])
            self.assertTrue(np.isfinite(fresh.report["FP32_RMSE"]))
            self.assertEqual(fresh.report["complete_2D_matrices"],2)
            self.assertEqual(fresh.report["all_old_packed_prior_indices_excluded"],True)

    def test_reject_incompatible_FP32_prior_and_unsupported_modes(self):
        with self.assertRaises(ValueError):
            encode_fp32(self.toy.state_dict(),mode="magic")
        with self.assertRaises(ValueError):
            encode_fp32(self.toy.state_dict(),mode=MODE_LOCKED)
        class Wrong(nn.Module):
            def __init__(self):
                super().__init__()
                self.a=nn.Linear(24,16)
        with self.assertRaises(ValueError):
            encode_fp32(Wrong().state_dict(),mode=MODE_ADAPT,
                        prior_blob=self.warm_old.blob)

    def test_corruption_and_generic_CB61_envelope(self):
        code=self.warm_old.blob
        damaged=bytearray(code)
        damaged[-50]^=0x41
        with self.assertRaises(ValueError):
            CB59(bytes(damaged))
        wrapped=to_cb61(code)
        self.assertEqual(wrapped[:8],b"CB61ZL01")
        import zlib,struct,hashlib
        header=struct.Struct("<8sQI32s")
        magic,n,crc,sha=header.unpack_from(wrapped)
        body=zlib.decompress(wrapped[header.size:])
        self.assertEqual(magic,b"CB61ZL01")
        self.assertEqual(n,len(body))
        self.assertEqual(zlib.crc32(body),crc)
        self.assertEqual(hashlib.sha256(body).digest(),sha)
        self.assertEqual(body,code)

    def test_reject_invalid_fp32_values(self):
        bad={k:v.detach().clone() for k,v in self.toy.state_dict().items()}
        bad["dense.bias"][0]=float("nan")
        with self.assertRaises((ValueError,RuntimeError)):
            encode_fp32(bad,mode=MODE_COLD)


if __name__=="__main__":
    unittest.main()
