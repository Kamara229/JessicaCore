"""Toy original pretrained-source refit should preserve full ABI and nonhead bytes."""
from __future__ import annotations
import unittest
import numpy as np
import torch
from torch import nn
from direct_codebook_v59 import CB59,encode,HEADER
from v102.recode import recode_source,swap_whole_tied_head_blob

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(64,64)
        self.head=nn.Linear(64,64,bias=False)
        self.head.weight=self.embed.weight
        self.body=nn.Linear(64,64)
    def forward(self,ids):return self.head(self.body(self.embed(ids)))

class TestV102(unittest.TestCase):
    def test_archive_exact_size_alias_and_other_weights(self):
        torch.manual_seed(102)
        model=Tiny()
        source,_,_=encode(model.state_dict())
        old=CB59(source,expected_keys=model.state_dict().keys())
        new,summary=recode_source(source,model.head.weight)
        target=CB59(new,expected_keys=model.state_dict().keys())
        self.assertEqual(len(new),len(source))
        self.assertNotEqual(new,source)
        self.assertIs(target.handles["head.weight"] if "head.weight" in target.handles else
                      target.handles["lm_head.weight"],target.handles["embed.weight"])
        self.assertEqual(summary["new_complete_archive_bytes"],len(new))
        self.assertEqual(old.handles["body.weight"].packed.tobytes(),
                         target.handles["body.weight"].packed.tobytes())
        self.assertTrue(np.array_equal(old.raw["body.bias"],target.raw["body.bias"]))
        self.assertNotEqual(old.handles["embed.weight"].packed.tobytes(),
                            target.handles["embed.weight"].packed.tobytes())
    def test_corrupted_physical_head_segments_rejected(self):
        torch.manual_seed(103)
        model=Tiny()
        source,_,_=encode(model.state_dict())
        with self.assertRaises(AssertionError):
            swap_whole_tied_head_blob(source,{"packed":b"x","row_scales":b"x",
                                             "centroids":b"x"})

if __name__=="__main__":unittest.main()
