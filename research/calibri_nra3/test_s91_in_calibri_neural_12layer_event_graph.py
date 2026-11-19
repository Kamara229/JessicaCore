"""NRA3 S91 neural model-integration smoke tests, NOT true43M/J7 tests.

Toy 12-layer backbone proves a new NN child belongs to Calibri
state_dict, every old layer returns exact residual identity, and
all 12 hidden-state source streams genuinely affect graph outputs.
Real original Calibri43M CPU capture is the next independent stage,
after old protected616 J7 is physically replayed.
"""
from __future__ import annotations
import ast
import inspect
import unittest
from pathlib import Path

import torch
from torch import nn

from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,STAGE,N_LAYERS,N_ROLES,N_STATES,N_MICRO,
    MEMORY_SLOTS,
)
from research.nra3_s20.s82_predicate_centered_situation_fields import AREAS


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias=nn.Parameter(torch.randn(320)*0.001)

    def forward(self,x):
        return x+self.bias


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([ToyLayer() for _ in range(N_LAYERS)])

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x


class ToyCalibri(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=ToyBackbone()

    def forward(self,x):
        return self.model(x)


class TestNRA3NeuralInsideCalibri(unittest.TestCase):
    def test_eight_independent_NRA2_microareas_and_12layers(self):
        model=NRA3InCalibriNeuralWeights()
        b=model.budget()
        self.assertEqual(STAGE,
            "CALIBRI_NRA3_S91_NEURAL_12LAYER_RELATIONAL_EVENT_GRAPH_RESEARCH_ONLY")
        self.assertEqual(b["new_neural_parameters"],22311)
        self.assertEqual(b["nominal_FP16_weight_bytes"],44622)
        self.assertEqual(b["all_12_original_calibri_layers"],12)
        self.assertEqual(b["learned_layer_streams"],4)
        self.assertEqual(b["grammatical_event_argument_role_heads"],8)
        self.assertEqual(b["NRA2_independent_mini_areas_preserved"],8)
        self.assertEqual(b["trainable_in_model_memory_slots"],16)
        self.assertFalse(b["external_runtime_parser_required"])
        self.assertTrue(b["all_in_bpp_and_CPU_not_certified"])
        self.assertFalse(model.is_trained)
        self.assertEqual(model.output_gate,0)
        self.assertFalse(model.candidate_accepted)
        self.assertFalse(model.activation_allowed)
        self.assertEqual(len(AREAS),len(model.mini_areas))
        self.assertEqual(len({id(z[0].weight) for z in model.mini_areas.values()}),8)

    def test_original_hidden12_source_only_shapes_and_mask(self):
        torch.manual_seed(9101)
        model=NRA3InCalibriNeuralWeights()
        h=torch.randn(2,12,6,320)
        valid=torch.tensor([[True]*6,[True]*4+[False]*2])
        got=model(h,valid)
        self.assertEqual(got.event_mention_logits.shape,(2,6))
        self.assertEqual(got.role_arc_logits.shape,(2,8,6,6))
        self.assertEqual(got.event_state_logits.shape,(2,6,6))
        self.assertEqual(got.mini_area_logits.shape,(2,6,8))
        self.assertEqual(got.layer_router_weights.shape,(4,12))
        self.assertEqual(got.memory_slot_attention.shape,(2,6,16))
        self.assertTrue(torch.equal(got.native_valid,valid))
        self.assertTrue(torch.all(got.event_mention_logits[1,4:]==-1.e4))
        self.assertTrue(torch.all(got.role_arc_logits[1,:,4:,:]==-1.e4))
        self.assertTrue(torch.all(got.role_arc_logits[1,:,:,4:]==-1.e4))
        self.assertTrue(torch.all(got.event_state_logits[1,4:]==-1.e4))
        self.assertTrue(torch.all(got.mini_area_logits[1,4:]==-1.e4))
        self.assertTrue(torch.allclose(got.layer_router_weights.sum(-1),
                                       torch.ones(4)))
        self.assertTrue(torch.allclose(got.memory_slot_attention.sum(-1),
                                       torch.ones(2,6)))
        self.assertIs(model.preserve_original_output(h,got),h)

    def test_all_12_layers_are_accessible_not_only_eighth(self):
        torch.manual_seed(9102)
        model=NRA3InCalibriNeuralWeights().eval()
        h=torch.randn(1,12,4,320)
        valid=torch.ones(1,4,dtype=torch.bool)
        with torch.inference_mode():
            original=model(h,valid).event_state_logits
            left=h.clone();left[:,0]+=3.
            right=h.clone();right[:,11]-=2.
            changed_first=model(left,valid).event_state_logits
            changed_last=model(right,valid).event_state_logits
        self.assertFalse(torch.equal(original,changed_first))
        self.assertFalse(torch.equal(original,changed_last))

    def test_12layer_router_arc_memory_each_receives_neural_gradient(self):
        torch.manual_seed(9103)
        block=NRA3InCalibriNeuralWeights()
        hidden=torch.randn(2,12,5,320)
        graph=block(hidden,torch.ones(2,5,dtype=torch.bool))
        loss=(graph.event_mention_logits.square().mean()
              +graph.role_arc_logits.square().mean()
              +graph.event_state_logits.square().mean()
              +graph.mini_area_logits.square().mean())
        loss.backward()
        for name,p in block.named_parameters():
            self.assertIsNotNone(p.grad,name)
            self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertFalse(hidden.requires_grad)
        self.assertIsNone(hidden.grad)

    def test_new_NRA3_params_physically_REGISTER_INSIDE_model_weights(self):
        torch.manual_seed(9104)
        backbone=ToyCalibri().eval()
        before={k:v.detach().clone() for k,v in backbone.state_dict().items()}
        inp=torch.randn(1,5,320)
        with torch.inference_mode():
            baseline=backbone(inp).clone()
        graph=NRA3InCalibriNeuralWeights()
        graph.attach_to_RESEARCH_calibri(backbone)
        state=backbone.state_dict()
        self.assertTrue(any(k.startswith("model.nra3_neural_weights.")
                            for k in state))
        self.assertGreater(sum(p.numel() for p in graph.parameters()),20000)
        for k,v in before.items():
            self.assertTrue(torch.equal(v,state[k]),k)
        with torch.inference_mode():
            with graph.read_only_all12_source_scope(backbone):
                got=backbone(inp)
            self.assertTrue(torch.equal(baseline,got))
            self.assertIsNotNone(graph.latest_source_graph)
            self.assertEqual(graph.latest_source_hidden.shape,(1,12,5,320))
            self.assertEqual(graph.capture_count,1)
            self.assertEqual(graph.latest_source_graph.role_arc_logits.shape,
                             (1,8,5,5))
            self.assertTrue(torch.equal(baseline,backbone(inp)))
        self.assertTrue(all(len(z._forward_hooks)==0
                            for z in backbone.model.layers))
        for k,v in before.items():
            self.assertTrue(torch.equal(v,backbone.state_dict()[k]),k)
        with self.assertRaisesRegex(ValueError,"fresh 12-layer"):
            graph.attach_to_RESEARCH_calibri(backbone)

    def test_unverified_gate_and_unsupported_hidden_refuse_activation(self):
        graph=NRA3InCalibriNeuralWeights()
        h=torch.randn(1,12,3,320)
        valid=torch.ones(1,3,dtype=torch.bool)
        graph.output_gate=1
        with self.assertRaisesRegex(ValueError,"activate"):
            graph(h,valid)
        graph.output_gate=0
        with self.assertRaisesRegex(ValueError,"twelve layer"):
            graph(torch.randn(1,8,3,320),valid)
        with self.assertRaisesRegex(ValueError,"twelve layer"):
            graph(torch.randn(1,12,3,319),valid)

    def test_neural_slot_consolidation_is_trainable_and_snapshot_frozen(self):
        graph=NRA3InCalibriNeuralWeights()
        old=graph.event_memory_slots.detach().clone()
        self.assertEqual(float(graph.knowledge_consolidation_penalty(old)),0.)
        with torch.no_grad():
            graph.event_memory_slots.add_(0.01)
        penalty=graph.knowledge_consolidation_penalty(old)
        self.assertGreater(float(penalty),0.)
        penalty.backward()
        self.assertIsNotNone(graph.event_memory_slots.grad)
        self.assertTrue(torch.isfinite(graph.event_memory_slots.grad).all())

    def test_no_spent_test_or_external_parser_runtime_imports(self):
        p=Path("research/calibri_nra3/s91_neural_multilayer_event_graph.py")
        tree=ast.parse(p.read_text(encoding="utf8"))
        imports=[z.module or "" for z in tree.body if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s79" in v.lower() or
                             "s86" in v.lower() or
                             "s88" in v.lower() or
                             "gold" in v.lower() or
                             "stanza" in v.lower() or
                             "udpipe" in v.lower()
                             for v in imports))
        sig=inspect.signature(NRA3InCalibriNeuralWeights.forward)
        self.assertEqual(set(sig.parameters),
                         {"self","original_layer_hidden","source_native_token_valid"})


if __name__=="__main__":
    unittest.main()
