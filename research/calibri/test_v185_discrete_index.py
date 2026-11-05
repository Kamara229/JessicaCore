"""Real 5-bit mutable index roundtrip / projected TRAIN-gradient regression."""
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from direct_codebook_v59 import encode,CB59,unpack5
from v185 import protocol as p
from v185.codec import replace_discrete
from v185.optim import propose_codes,apply_codes


class MutableQuantizedCodeIndices(unittest.TestCase):
    def test_fixed_budget_and_immutable_public_probes(self):
        self.assertEqual(p.SOURCE_BYTES,11229303)
        self.assertEqual(p.MAX_BYTES,11770063)
        self.assertEqual(len(p.CASE_IDS),24)
        self.assertEqual(len(p.STEPS_OPERATORS),p.STEPS)
        self.assertEqual(p.MAX_INDEX_UPDATES_PER_STEP,512)
        self.assertEqual(len(set(p.PUBLIC_TRAIN_INDICES)),16)
        self.assertTrue(all(x.endswith(".weight") for x in p.STEPS_OPERATORS))

    def test_discrete_proposal_is_deterministic_and_only_5bit(self):
        torch.manual_seed(185)
        rows,cols=16,16
        centers=torch.arange(32*8,dtype=torch.float32).reshape(32,8)/100
        src=SimpleNamespace(shape=(rows,cols),base_scales=torch.ones(rows),
            centroids=torch.nn.Parameter(centers.clone()),
            codes=torch.zeros(rows*cols//8,dtype=torch.long))
        teacher=np.ones((rows,cols),dtype=np.float32)*.3
        gradient=np.ones((rows,cols),dtype=np.float32)*(-.05)
        first=propose_codes(src,gradient,teacher)
        second=propose_codes(src,gradient,teacher)
        self.assertEqual(first,second)
        self.assertLessEqual(len(first["indices"]),p.MAX_INDEX_UPDATES_PER_STEP)
        if first["indices"]:
            old=src.codes.clone()
            apply_codes(src,first,True)
            self.assertTrue(torch.any(old!=src.codes))
            self.assertTrue(torch.all(src.codes>=0))
            self.assertTrue(torch.all(src.codes<32))
            apply_codes(src,first,False)
            self.assertTrue(torch.equal(src.codes,old))
        self.assertEqual(propose_codes(src,np.zeros_like(gradient),teacher)
            ["proposed_groups"],0)

    def test_export_paid_indices_and_codebooks_preserves_original_aliases(self):
        torch.manual_seed(189)
        tied=torch.randn(16,8)
        state={"model.embed_tokens.weight":tied,
               "lm_head.weight":tied,
               "model.norm.weight":torch.ones(8)}
        physical,_,_=encode(state)
        original=CB59(physical,expected_keys=state.keys())
        key="model.embed_tokens.weight"
        h=original.handles[key]
        codes=torch.tensor(unpack5(h.packed.tobytes(),16),dtype=torch.long)
        source=SimpleNamespace(codes=codes.clone(),
            centroids=torch.nn.Parameter(torch.from_numpy(h.centroids.copy())))
        source.codes[0]=(int(source.codes[0])+1)%32
        with torch.no_grad():source.centroids[0,0]+=0.05
        model,seal=replace_discrete(original,{key:source},strict=False)
        reloaded=CB59(model)
        self.assertEqual(len(model),len(physical))
        self.assertNotEqual(original.hash(),reloaded.hash())
        self.assertEqual(seal["actual_changed_5bit_group_indices"],1)
        self.assertGreater(seal["actually_changed_FP16_codebook_cells"],0)
        self.assertIs(reloaded.handles["model.embed_tokens.weight"],
                      reloaded.handles["lm_head.weight"])
        np.testing.assert_array_equal(original.raw["model.norm.weight"],
                                      reloaded.raw["model.norm.weight"])
        with self.assertRaises(ValueError):
            replace_discrete(original,{},strict=False)

if __name__=="__main__":
    unittest.main()
