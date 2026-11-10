"""S111 research-only synthetic tests for prompt-conditioned dual donor gates."""
from __future__ import annotations

import unittest
import torch
from calibri2.s103.test_shared_all_layer_green_bridge import DummyModel,FakeTunnel
from calibri2.s111.prompt_conditioned_dual_tunnel import PromptConditionedDualTunnel


class TestPromptConditionedDualTunnel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(111)
        self.model=DummyModel()
        self.original=FakeTunnel().install(self.model)
        self.bridge=PromptConditionedDualTunnel(self.original).install(self.model)
        self.h=torch.tensor([[0.3,-0.4,0.7,0.8]],dtype=torch.float32)
        self.f1=torch.arange(1,513,dtype=torch.float32)/512.
        self.f2=self.f1.flip(0)

    def tearDown(self):
        self.bridge.uninstall()
        self.original.uninstall()

    def test_all_layers_connected_only_680_raw_paid_bytes(self):
        self.assertEqual(self.bridge.connected_layers,tuple(range(12)))
        self.assertEqual(len(self.bridge._handles),10)
        self.assertEqual(len(self.original._handles),2)
        self.assertEqual(tuple(self.bridge.coefficients.shape),(2,10,17))
        self.assertEqual(len(list(self.bridge.parameters())),1)
        self.assertEqual(list(self.bridge.state_dict()),["coefficients"])
        self.assertEqual(self.bridge.raw_weight_bytes(),680)

    def test_zero_gain_exact_original_inference_even_when_prompt_changes(self):
        with torch.inference_mode():
            for route,features in (("S93",self.f1),("S93",self.f2),
                                   ("S91",None),("S87",None),(None,None)):
                with self.bridge.prompt_scope(route,features):
                    actual=self.model(self.h)
                self.bridge.uninstall()
                expected=self.model(self.h)
                self.bridge.install(self.model)
                self.assertTrue(torch.equal(actual,expected),route)

    def test_prompt_specific_gates_change_downstream_states_without_route_drift(self):
        with torch.no_grad():
            self.bridge.coefficients[0,0,1]=0.9
            self.bridge.coefficients[1,0,2]=-0.6
        with torch.inference_mode():
            with self.bridge.prompt_scope("S93",self.f1):
                y1=self.model(self.h)
                in1=self.model.model.layers[1].last_input.clone()
            with self.bridge.prompt_scope("S93",self.f2):
                y2=self.model(self.h)
                in2=self.model.model.layers[1].last_input.clone()
            with self.bridge.prompt_scope("S87",None):
                y87=self.model(self.h)
            with self.bridge.prompt_scope("S91",None):
                y91=self.model(self.h)
        self.assertFalse(torch.equal(y1,y2))
        self.assertFalse(torch.equal(in1,in2))
        self.assertTrue(torch.equal(y87,y91))

    def test_gradients_reach_two_donors_and_context_features(self):
        with self.bridge.prompt_scope("S93",self.f1):
            objective=self.model(self.h).square().mean()
            objective.backward()
        g=self.bridge.coefficients.grad
        self.assertEqual(tuple(g.shape),(2,10,17))
        self.assertTrue(torch.isfinite(g).all())
        self.assertGreater(int(torch.count_nonzero(g[:, :, 0])),0)
        self.assertGreater(int(torch.count_nonzero(g[:, :, 1:])),0)
        self.assertTrue(torch.equal(self.bridge.coefficients,torch.zeros_like(g)))

    def test_exact_sidecar_quantization_validation_and_scope(self):
        with torch.no_grad():
            self.bridge.coefficients[0,1,4]=0.1234567
        physical=self.bridge.export_physical_gain()
        self.assertEqual(list(physical),[self.bridge.KEY])
        self.assertEqual(physical[self.bridge.KEY].dtype,torch.float16)
        with torch.no_grad():
            self.bridge.coefficients.zero_()
        self.bridge.load_physical_gain(physical)
        self.assertEqual(float(self.bridge.coefficients[0,1,4]),
                         float(physical[self.bridge.KEY][0,1,4]))
        with self.assertRaises(ValueError):
            self.bridge.load_physical_gain({"wrong":physical[self.bridge.KEY]})
        with self.assertRaises(ValueError):
            self.bridge.load_physical_gain({self.bridge.KEY:torch.zeros(2,10,16,dtype=torch.float16)})
        with self.assertRaises(ValueError):
            self.bridge.load_physical_gain({self.bridge.KEY:torch.full((2,10,17),float("nan"),dtype=torch.float16)})
        with self.assertRaises(ValueError):
            self.bridge.prepare_prompt(torch.randn(511))
        with self.assertRaises(ValueError):
            with self.bridge.prompt_scope("S93",None):
                pass
        try:
            with self.bridge.prompt_scope("S93",self.f2):
                raise RuntimeError("forced")
        except RuntimeError:
            pass
        self.assertIsNone(self.bridge._active_profile)
        self.assertIsNone(self.bridge._features)


if __name__=="__main__":
    unittest.main(verbosity=2)
