"""S119d synthetic-only contracts. Actual 12-layer CPU TRAIN-A is separate CI step."""
import unittest
import torch

from research.calibri_nra3.s119d_reuse_old37087_true12_inweight_eventgraph import (
    partition_full_state, two_hop_parameter_free_event_relation_graph,
    DROPPED,PREFIX,
)
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import (
    NRA3BoundedStructuralTrustInCalibriWeights,
)
from research.calibri_nra3.s119a_hard_four_goals import GateFailure


class SharedInWeightGraphContract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_s119d_partition_preserves_full_8_mini_heads_and_all12_router(self):
        model=NRA3BoundedStructuralTrustInCalibriWeights().eval()
        full={PREFIX+k:v.detach().half() for k,v in model.state_dict().items()}
        old,removed=partition_full_state(full)
        self.assertEqual(sum(v.numel() for v in old.values()),37087)
        self.assertEqual(sum(v.numel() for v in removed.values()),16664)
        self.assertEqual(set(removed),DROPPED)
        for key in ("role_query.weight","role_key.weight","role_bilinear",
                    "event_memory_slots","shared_hidden_projector.weight",
                    "layer_router_logits"):
            self.assertIn(key,old)
        self.assertEqual(sum(k.startswith("mini_areas.") for k in old),32)

    def test_reused_shared_graph_all_shapes_mask_and_grad_parameter_count(self):
        torch.manual_seed(119)
        shared=NRA3RelationalMorphologyInCalibriWeights().eval()
        h=torch.randn(1,12,7,320)
        valid=torch.tensor([[True,True,True,True,True,False,False]])
        x=shared(h,valid)
        n_before=sum(p.numel() for p in shared.parameters())
        with torch.no_grad():
            states=two_hop_parameter_free_event_relation_graph(x,valid)
        self.assertEqual(states.shape,(1,7,6))
        self.assertTrue(torch.isfinite(states).all())
        self.assertTrue(torch.equal(states[0,5:],torch.zeros_like(states[0,5:])))
        self.assertEqual(n_before,37087)
        self.assertEqual(n_before,sum(p.numel() for p in shared.parameters()))
        self.assertEqual(shared.output_gate,0)

    def test_old_8_mini_morphology_bitwise_identical_vs_full(self):
        model=NRA3BoundedStructuralTrustInCalibriWeights().eval()
        weights={k:v.detach().half() for k,v in model.state_dict().items()}
        full={PREFIX+k:v for k,v in weights.items()}
        model.load_state_dict({k:v.float() for k,v in weights.items()},strict=True)
        old,_=partition_full_state(full)
        shared=NRA3RelationalMorphologyInCalibriWeights().eval()
        shared.load_state_dict({k:v.float() for k,v in old.items()},strict=True)
        h=torch.randn(1,12,5,320)
        valid=torch.ones((1,5),dtype=torch.bool)
        with torch.no_grad():
            old_scores=shared(h,valid)
            full_scores=model(h,valid)
        self.assertTrue(torch.equal(old_scores.mini_area_logits,full_scores.mini_area_logits))
        self.assertTrue(torch.equal(old_scores.event_state_logits,full_scores.event_state_logits))
        self.assertTrue(torch.equal(old_scores.memory_slot_attention,full_scores.memory_slot_attention))
        for attr in ("upos_logits","case_logits","number_logits","voice_logits"):
            self.assertTrue(torch.equal(getattr(old_scores,attr),getattr(full_scores,attr)))

    def test_no_missing_pretrained_original_weight_allowed(self):
        model=NRA3BoundedStructuralTrustInCalibriWeights().eval()
        all_weights={PREFIX+k:v.detach().half() for k,v in model.state_dict().items()}
        del all_weights[PREFIX+"role_query.weight"]
        with self.assertRaises(GateFailure):
            partition_full_state(all_weights)

    def test_no_linguistic_gold_required_at_graph_forward(self):
        model=NRA3RelationalMorphologyInCalibriWeights().eval()
        h=torch.randn(1,12,3,320)
        valid=torch.ones(1,3,dtype=torch.bool)
        with torch.no_grad():
            graph=model(h,valid)
            result=two_hop_parameter_free_event_relation_graph(graph,valid)
        self.assertEqual(result.shape,(1,3,6))
        self.assertFalse(model.candidate_accepted)
        self.assertFalse(model.activation_allowed)


if __name__=="__main__":
    unittest.main()
