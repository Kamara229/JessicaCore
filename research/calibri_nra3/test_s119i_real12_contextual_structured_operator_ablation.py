"""S119i synthetic hard contracts; real S114 source-ablation separate CI step."""
import unittest
from dataclasses import replace
import torch
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119g_unified_planning_internal_action_neural_area import (
    UnifiedCalibriWithNPA1,
)
from research.calibri_nra3.s119i_real12_contextual_structured_operator_ablation import (
    ContextualStructuredInterface,
    UnifiedCalibriWithStructuredInterface,calculate_example,train_variant,
)

class S119IUnifyingCausalRelationalRepresentations(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119121)
        torch.set_num_threads(2)

    def test_01_full_prior_53751_old_neural_and_NPA_unchanged_structure(self):
        m=UnifiedCalibriWithStructuredInterface()
        self.assertEqual(sum(p.numel()
            for p in m.nra3_neural_weights.parameters()),53751)
        self.assertEqual(sum(p.numel()
            for p in m.core_growth.parameters()),30720)
        self.assertEqual(m.reasoning_role_gate.numel(),8)
        self.assertEqual(sum(p.numel()
            for p in m.planning_action_area.parameters()),32298)
        self.assertGreater(sum(p.numel()
            for p in m.structured_interface.parameters()),10000)
        self.assertFalse(m.candidate_accepted)
        self.assertEqual(m.output_gate,0)

    def test_02_early_layer_zero_residual_exact_original_unified_forward(self):
        m=UnifiedCalibriWithStructuredInterface().eval()
        h=torch.randn(1,12,6,320)
        v=torch.ones((1,6),dtype=torch.bool)
        with torch.inference_mode():
            old_g,old_r,old_p=UnifiedCalibriWithNPA1.forward(m,h,v)
            g,r,p,_=m(h,v,contextual=True)
        for key in ("role_arc_logits","mini_area_logits","event_state_logits",
                    "upos_logits","case_logits","voice_logits"):
            self.assertTrue(torch.equal(getattr(g,key),getattr(old_g,key)),key)
        self.assertTrue(torch.equal(r,old_r))
        self.assertTrue(torch.equal(p.next_event_logits,old_p.next_event_logits))

    def test_03_contextual_graph_responds_to_PREVIOUS_token_differences(self):
        m=UnifiedCalibriWithStructuredInterface().eval()
        h=torch.randn(1,12,7,320)
        v=torch.ones(1,7,dtype=torch.bool)
        with torch.no_grad():
            old=m.nra3_neural_weights(h,v)
            s=m.structured_interface
            s.fuse.weight.fill_(0.01)
            first=s(h[:,0],old,v,contextual=True)
            alt=h.clone()
            alt[:,:,0]+=25.
            old2=m.nra3_neural_weights(alt,v)
            second=s(alt[:,0],old2,v,contextual=True)
            local_1=s(h[:,0],old,v,contextual=False)
            # Full real-token source context can differ in the two paths.
        self.assertFalse(torch.equal(first[:,-1],second[:,-1]))
        self.assertTrue(torch.equal(first[:,:-1],h[:,0,:-1]))
        self.assertTrue(torch.equal(local_1[:,:-1],h[:,0,:-1]))

    def test_04_no_parameter_added_only_for_contextual_variant(self):
        m=UnifiedCalibriWithStructuredInterface().eval()
        h=torch.randn(1,12,4,320)
        v=torch.ones(1,4,dtype=torch.bool)
        n=sum(t.numel() for t in m.parameters())
        with torch.no_grad():
            m(h,v,contextual=True)
            m(h,v,contextual=False)
        self.assertEqual(sum(t.numel() for t in m.parameters()),n)

    def test_05_old_NRA3_tensors_never_receive_joint_update(self):
        m=UnifiedCalibriWithStructuredInterface()
        for p in m.nra3_neural_weights.parameters():p.requires_grad_(False)
        h=torch.randn(1,12,12,320)
        v=torch.ones(1,12,dtype=torch.bool)
        old={k:z.detach().clone()
             for k,z in m.nra3_neural_weights.state_dict().items()}
        loss,_=calculate_example(m,h,v,6,contextual=True)
        loss.backward()
        self.assertGreater(float(
            m.structured_interface.fuse.weight.grad.abs().sum()),0)
        self.assertGreater(float(m.core_growth.up.grad.abs().sum()),0)
        self.assertGreater(float(
            m.planning_action_area.pred_next_event.weight.grad.abs().sum()),0)
        for k,z in old.items():
            self.assertTrue(torch.equal(z,m.nra3_neural_weights.state_dict()[k]))

    def test_06_cannot_read_unseen_future_as_training_input(self):
        m=UnifiedCalibriWithStructuredInterface()
        h=torch.randn(1,12,6,320)
        v=torch.ones(1,6,dtype=torch.bool)
        with self.assertRaises(GateFailure):
            calculate_example(m,h,v,4,contextual=True)

    def test_07_cannot_activate_structured_unverified_candidate(self):
        m=UnifiedCalibriWithStructuredInterface()
        m.structured_interface.output_gate=1
        h=torch.randn(1,12,4,320)
        v=torch.ones(1,4,dtype=torch.bool)
        with self.assertRaises(GateFailure):
            m(h,v,contextual=True)

    def test_08_must_reject_noncontiguous_fake_mask(self):
        m=UnifiedCalibriWithStructuredInterface()
        h=torch.randn(1,12,5,320)
        v=torch.tensor([[True,False,True,True,True]])
        with self.assertRaises(GateFailure):
            m(h,v,contextual=True)

    def test_09_old_neural_role_arcs_have_caused_structured_last_output(self):
        m=UnifiedCalibriWithStructuredInterface()
        h=torch.randn(1,12,8,320)
        v=torch.ones(1,8,dtype=torch.bool)
        with torch.no_grad():
            g=m.nra3_neural_weights(h,v)
            s=m.structured_interface
            s.fuse.weight.fill_(0.01)
            before=s(h[:,0],g,v,contextual=True)
            flipped=replace(g,role_arc_logits=g.role_arc_logits.clone())
            flipped.role_arc_logits[:, :, -1, :] = torch.arange(8,dtype=torch.float32)
            after=s(h[:,0],flipped,v,contextual=True)
        self.assertFalse(torch.equal(before[:,-1],after[:,-1]))

if __name__=="__main__":
    unittest.main()
