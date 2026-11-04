import hashlib
import unittest
import numpy as np
import torch
from torch import nn
from direct_codebook_v59 import encode,CB59,unpack5
from v138.repack import KEY,TIED,checked_source,encoded_all,patch_one_matrix
from v138.scoring import ARMS,summarize_cases


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(64,16)
        self.lm_head=nn.Linear(16,64,bias=False)
        self.lm_head.weight=self.model.embed_tokens.weight
        self.linear=nn.Linear(16,16)


class BitExactReencode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(138)
        cls.original=TinyModel()
        cls.weight=cls.original.model.embed_tokens.weight.detach().numpy().copy()
        cls.base,_,_=encode(cls.original.state_dict())

    def test_self_reencoding_reproduces_byte_identical_original(self):
        handle=checked_source(self.base,(64,16)).handles[KEY]
        total=self.weight.size//8
        original_codes=unpack5(handle.packed.tobytes(),total)
        # The unchanged original 5-bit codes and original FP16 centroid table
        # must rebuild CRC+SHA to the SAME complete binary file.
        from direct_codebook_v59 import pack5
        copy=patch_one_matrix(self.base,handle.centroids,pack5(original_codes),
                              expected_shape=(64,16))
        self.assertEqual(copy,self.base)

    def test_changed_center_has_exact_source_bytes_and_tied_alias(self):
        handle=checked_source(self.base,(64,16)).handles[KEY]
        centers=handle.centroids.copy()
        centers[0,0]+=np.float32(0.5)
        from direct_codebook_v59 import pack5
        codes=unpack5(handle.packed.tobytes(),self.weight.size//8)
        patched=patch_one_matrix(self.base,centers,pack5(codes),
                                 expected_shape=(64,16))
        self.assertNotEqual(hashlib.sha256(patched).digest(),hashlib.sha256(self.base).digest())
        self.assertEqual(len(patched),len(self.base))
        restored=checked_source(patched,(64,16))
        self.assertIs(restored.handles[KEY],restored.handles[TIED])
        self.assertEqual(restored.meta,CB59(self.base).meta)
        self.assertTrue(np.array_equal(restored.handles[KEY].packed,
                                       handle.packed))
        self.assertTrue(np.array_equal(restored.handles["linear.weight"].packed,
                                       CB59(self.base).handles["linear.weight"].packed))
        corrupted=bytearray(patched)
        corrupted[25]^=1
        with self.assertRaises(ValueError):
            CB59(bytes(corrupted))

    def test_real_nearest_code_pack_is_byte_aligned(self):
        base=checked_source(self.base,(64,16))
        handle=base.handles[KEY]
        packed=encoded_all(self.weight,handle.row_scales,handle.centroids,
                           chunk_groups=32,expected_shape=(64,16))
        self.assertEqual(len(packed),len(handle.packed))
        self.assertEqual(packed,handle.packed.tobytes())

    def test_malformed_shapes_and_wrong_arms_rejected(self):
        self.assertEqual(len(ARMS),3)
        base=checked_source(self.base,(64,16))
        with self.assertRaises(ValueError):
            encoded_all(self.weight[:2],base.handles[KEY].row_scales,
                        base.handles[KEY].centroids,expected_shape=(64,16))
        with self.assertRaises(ValueError):
            summarize_cases({"original_CB59":[]})


if __name__=="__main__":
    unittest.main()
