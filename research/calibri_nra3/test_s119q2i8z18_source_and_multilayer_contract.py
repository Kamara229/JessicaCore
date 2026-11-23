"""Z18 source-only preregistered new RU 28TRAIN/14DEV, 12-layer state hooks."""
import unittest
from collections import Counter
import torch
from torch import nn
from research.calibri_nra3.s119q2i8z18_fresh28train14dev_multilayer_source import (
    verify,rows,digest,KINDS,MULTI_LAYER_ZERO_BASED,CANDIDATE_TOP_K,
)
from research.calibri_nra3.s119q2i8z18_multilayer_h320_contract import (
    capture_original_selected_layers,fixed_multi_layer_h320,HIDDEN,
)

class TestZ18(unittest.TestCase):
    def test_01_source_28_train14_dev_and_historical_disjoint(self):
        x=verify()
        self.assertEqual((x["TRAIN_count"],x["DEV_count"]),(28,14))
        self.assertEqual(x["TRAIN_SHA256"],digest(rows("TRAIN")))
        self.assertEqual(x["DEV_SHA256"],digest(rows("DEV")))
        self.assertEqual(Counter(y["kind"] for y in rows("TRAIN")),
                         {k:4 for k in KINDS})
        self.assertEqual(Counter(y["kind"] for y in rows("DEV")),
                         {k:2 for k in KINDS})
        self.assertTrue(x["disjoint_old_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16"])
        self.assertEqual((MULTI_LAYER_ZERO_BASED,CANDIDATE_TOP_K),((2,6,10),64))
        for bad in ("J7","GOLD","EVAL128","TEST","TRAIN+DEV"):
            with self.assertRaises(ValueError):rows(bad)

    def test_02_capture_actual_three_layers_in_only_one_decoder_forward(self):
        class FakeLayer(nn.Module):
            def __init__(self,i):
                super().__init__()
                self.i=i
            def forward(self,x):
                return (x+self.i/100,)
        class Wrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers=nn.ModuleList([FakeLayer(i) for i in range(12)])
            def forward(self,x):
                for layer in self.layers:
                    x=layer(x)[0]
                return x
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.model=Wrapped()
        model=Toy().eval()
        history={k:set(model.model.layers[k]._forward_hooks)
                 for k in MULTI_LAYER_ZERO_BASED}
        with capture_original_selected_layers(model) as states:
            y=model.model(torch.randn((1,8,HIDDEN)))
        self.assertEqual(y.shape,(1,8,HIDDEN))
        self.assertEqual(set(states),set(MULTI_LAYER_ZERO_BASED))
        final=torch.randn((1,HIDDEN))
        mix=fixed_multi_layer_h320(states,final)
        self.assertEqual(mix.shape,(1,HIDDEN))
        self.assertFalse(mix.requires_grad)
        self.assertTrue(torch.isfinite(mix).all())
        self.assertTrue(all(set(model.model.layers[k]._forward_hooks)==history[k]
                            for k in MULTI_LAYER_ZERO_BASED))
        with self.assertRaises(ValueError):
            with capture_original_selected_layers(model):
                pass
        self.assertTrue(all(set(model.model.layers[k]._forward_hooks)==history[k]
                            for k in MULTI_LAYER_ZERO_BASED))

    def test_03_strict_no_missing_reused_or_wrong_layer_shapes(self):
        base={2:torch.ones((1,HIDDEN)),
              6:torch.ones((1,HIDDEN)),10:torch.ones((1,HIDDEN))}
        with self.assertRaises(ValueError):
            fixed_multi_layer_h320({2:base[2],6:base[6]},
                                   torch.ones((1,HIDDEN)))
        with self.assertRaises(ValueError):
            fixed_multi_layer_h320(base,torch.ones((1,321)))
        with self.assertRaises(ValueError):
            fixed_multi_layer_h320(base,torch.full((1,HIDDEN),float("nan")))

if __name__=="__main__":
    unittest.main()
