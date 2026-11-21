"""Synthetic API contracts; real SHA-locked TRAIN-A source via workflow."""
import unittest
import torch
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119g_unified_planning_internal_action_neural_area import (
    ACTION_STEPS,LATENT_ACTIONS,NeuralPlanningActionArea,
    UnifiedCalibriWithNPA1,single_joint_loss_step,
)

class S119GUnifiedPlanningInternalActionArea(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119119)
        torch.set_num_threads(2)

    def test_01_new_area_actual_neural_weights_in_one_body(self):
        m=UnifiedCalibriWithNPA1()
        self.assertEqual(sum(p.numel()
            for p in m.nra3_neural_weights.parameters()),53751)
        self.assertEqual(sum(p.numel()
            for p in m.core_growth.parameters()),30720)
        self.assertEqual(m.reasoning_role_gate.numel(),8)
        self.assertGreater(sum(p.numel()
            for p in m.planning_action_area.parameters()),10000)
        self.assertIn("planning_action_area.action_selector.weight",m.state_dict())
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.candidate_accepted)

    def test_02_three_steps_internal_actual_neural_transitions(self):
        m=UnifiedCalibriWithNPA1().eval()
        h=torch.randn(1,12,6,320)
        v=torch.ones(1,6,dtype=torch.bool)
        with torch.no_grad():
            g,r,p=m(h,v)
        self.assertEqual(tuple(p.action_logits.shape),(1,3,4))
        self.assertEqual(tuple(p.state_trace.shape),(1,3,32))
        self.assertEqual(tuple(p.next_event_logits.shape),(1,6))
        self.assertTrue(torch.isfinite(p.next_event_logits).all())
        self.assertFalse(torch.equal(p.state_trace[:,0],p.state_trace[:,1]))
        self.assertEqual(len(LATENT_ACTIONS),4)
        self.assertEqual(ACTION_STEPS,3)

    def test_03_plan_prefix_must_not_read_future_padded_tokens(self):
        m=UnifiedCalibriWithNPA1().eval()
        h=torch.randn(1,12,6,320)
        valid=torch.ones(1,6,dtype=torch.bool)
        with torch.no_grad():
            _,_,only=m(h,valid)
            padded=torch.cat((h,torch.randn(1,12,3,320)*999),2)
            padded_valid=torch.cat((valid,torch.zeros(1,3,dtype=torch.bool)),1)
            _,_,other=m(padded,padded_valid)
        self.assertTrue(torch.allclose(
            only.next_event_logits,other.next_event_logits,atol=1e-5,rtol=1e-5),
            "NPA1 cannot observe invalid future tokens")

    def test_04_single_objective_receives_core_nra3_and_action_grad(self):
        m=UnifiedCalibriWithNPA1()
        for p in m.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        h=torch.randn(1,12,8,320)
        v=torch.ones(1,8,dtype=torch.bool)
        old={k:t.detach().clone()
             for k,t in m.nra3_neural_weights.state_dict().items()}
        opt=torch.optim.Adam(m.new_joint_trainables(),lr=.0005)
        trace=single_joint_loss_step(m,h,v,5,opt)
        for k,x in trace["grads_l1"].items():
            self.assertGreater(x,1.e-10,k)
        self.assertTrue(all(torch.equal(
            m.nra3_neural_weights.state_dict()[k],x)
            for k,x in old.items()))
        self.assertFalse(m.candidate_accepted)

    def test_05_noncausal_discontiguous_mask_rejected(self):
        m=UnifiedCalibriWithNPA1()
        h=torch.randn(1,12,5,320)
        v=torch.tensor([[True,False,True,True,True]])
        with self.assertRaises(GateFailure):
            m(h,v)

    def test_06_output_gate_must_stay_zero(self):
        m=UnifiedCalibriWithNPA1()
        m.planning_action_area.output_gate=1
        h=torch.randn(1,12,4,320)
        v=torch.ones(1,4,dtype=torch.bool)
        with self.assertRaises(GateFailure):
            m(h,v)

    def test_07_future_source_only_loss_must_have_next_token(self):
        m=UnifiedCalibriWithNPA1()
        for p in m.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        h=torch.randn(1,12,5,320)
        v=torch.ones(1,5,dtype=torch.bool)
        opt=torch.optim.Adam(m.new_joint_trainables(),lr=.0005)
        with self.assertRaises(GateFailure):
            single_joint_loss_step(m,h,v,5,opt)

    def test_08_model_is_not_a_external_tool_executor(self):
        m=NeuralPlanningActionArea()
        self.assertFalse(any(hasattr(m,x) for x in
            ("web","browser","open_url","execute_tool","call_external_tool")))
        self.assertTrue(hasattr(m,"internal_action_matrices"))
        self.assertTrue(hasattr(m,"state_update"))

if __name__=="__main__":
    unittest.main()
