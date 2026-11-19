"""S72 original 43M/NRA2 situation micro-areas contracts, SYNTHETIC ONLY."""
import unittest

import torch

from research.nra3_s20.s72_nra2_situation_microareas import (
    AREAS,MAX_ACTIVE,NEG_INF,SituationMicroAreas,
)


class TestS72SituationMicroAreas(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7200)
        self.net=SituationMicroAreas()
        b,k,w=2,3,7
        self.f=torch.randn(b,k,36)
        self.words=torch.randn(b,w,320,dtype=torch.float16)
        self.indices=torch.tensor([[1,3,5],[0,2,4]],dtype=torch.long)
        self.safe=torch.ones(b,k,w,dtype=torch.bool)
        self.safe[0,0,4]=False
        self.allowed=torch.ones(b,k,dtype=torch.bool)
        self.allowed[1,2]=False
        self.old_joint=torch.randn(b,k,w+1)
        self.old_joint[0,0,4]=NEG_INF
        self.old_joint[1,2,:]=NEG_INF
        self.old_neg=torch.randn(b,k,w)
        self.old_fact=torch.randn(b,k,3)

    def call(self):
        return self.net(self.f,self.words,self.indices,self.safe,
                        self.allowed,self.old_joint,self.old_neg,self.old_fact)

    def test_untrained_zero_effect_on_valid_frozen_s65_logits(self):
        shadow=self.call()
        self.assertEqual(shadow.target_or_none.shape,(2,3,8))
        self.assertEqual(shadow.rejected_edge.shape,(2,3,7))
        self.assertEqual(shadow.event_factuality.shape,(2,3,3))
        self.assertTrue(torch.equal(shadow.target_or_none,self.old_joint))
        self.assertTrue(torch.equal(shadow.rejected_edge,self.old_neg))
        self.assertTrue(torch.equal(shadow.event_factuality,self.old_fact))
        self.assertEqual(shadow.output_gate,0)
        self.assertFalse(shadow.candidate_accepted)
        self.assertFalse(shadow.activation_allowed)
        self.assertFalse(shadow.online_causal_certified)

    def test_compositional_multi_label_route_max_four_not_single_choice(self):
        shadow=self.call()
        weights=shadow.active_area_weights
        self.assertEqual(weights.shape,(2,3,len(AREAS)))
        active=(weights>0).sum(-1)
        self.assertTrue(torch.equal(active[self.allowed],
                                    torch.full_like(active[self.allowed],MAX_ACTIVE)))
        self.assertEqual(int(active[1,2]),0)
        self.assertTrue(torch.allclose(
            weights[self.allowed].sum(-1),
            torch.ones(int(self.allowed.sum())),atol=1e-6))
        self.assertFalse(bool((weights[~self.allowed]!=0).any()))

    def test_independent_area_heads_can_train_without_parent_grad(self):
        shadow=self.call()
        loss=(shadow.target_or_none[0,0,0]
              +shadow.rejected_edge[0,0,0]
              +shadow.event_factuality[0,0].sum())
        loss.backward()
        self.assertTrue(any(p.grad is not None and
                            bool(torch.isfinite(p.grad).all())
                            for p in self.net.parameters()))
        self.assertIsNone(self.words.grad)
        self.assertIsNone(self.f.grad)
        self.assertIsNone(self.old_joint.grad)
        self.assertIsNone(self.old_fact.grad)

    def test_train_only_auxiliary_area_labels_support_overlap(self):
        shadow=self.call()
        labels=torch.zeros_like(shadow.router_logits)
        labels[0,0,[0,1,2]]=1  # quote + scope + passive co-occur
        labels[1,0,[3,6]]=1
        loss=self.net.old_train_only_area_supervision(
            shadow.router_logits,labels,self.allowed)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertGreater(float(loss),0)
        loss.backward()
        self.assertIsNotNone(self.net.router.weight.grad)
        invalid=labels.clone()
        invalid[0,0,0]=.5
        with self.assertRaisesRegex(ValueError,"binary situation"):
            self.net.old_train_only_area_supervision(
                shadow.router_logits,invalid,self.allowed)

    def test_no_gold_feature_or_unsafe_input_admitted(self):
        self.f.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,"all parents frozen"):
            self.call()
        self.f.requires_grad_(False)
        invalid=self.indices.clone()
        invalid[0,0]=80
        with self.assertRaisesRegex(ValueError,"event positions"):
            self.net(self.f,self.words,invalid,self.safe,self.allowed,
                     self.old_joint,self.old_neg,self.old_fact)

    def test_small_shadow_budget_and_eight_distinct_areas(self):
        self.assertEqual(len(AREAS),8)
        self.assertEqual(len(set(AREAS)),8)
        budget=self.net.parameter_budget()
        self.assertLess(budget["new_architecture_parameters"],15000)
        self.assertEqual(budget["new_FP16_weight_bytes_if_fully_serialized"],
                         budget["new_architecture_parameters"]*2)
        self.assertEqual(budget["max_simultaneously_active_areas"],4)
        self.assertFalse(budget["original43M_or_GT_weights_touched"])
        self.assertTrue(budget["physical_all_in_bpp_and_CPU_speed_NOT_certified"])

if __name__=="__main__":
    unittest.main()
