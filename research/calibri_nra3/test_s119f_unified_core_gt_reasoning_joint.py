"""S119f synthetic-only hard contracts, genuine old12 source in separate CI step."""
import unittest
import torch
from research.calibri_nra3.s119f_unified_core_gt_reasoning_joint import (
    UnifiedCalibri, train_joint,
)
from research.calibri_nra3.s119a_hard_four_goals import GateFailure


class UnifiedCoreAndGraphContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119)
        torch.set_num_threads(2)

    def test_01_full_old_NRA3_child_not_reduced(self):
        m=UnifiedCalibri()
        self.assertEqual(sum(p.numel() for p in m.nra3_neural_weights.parameters()),53751)
        self.assertEqual(sum(p.numel() for p in m.core_growth.parameters()),30720)
        self.assertEqual(m.reasoning_role_gate.numel(),8)
        self.assertFalse(m.candidate_accepted)
        self.assertFalse(m.activation_allowed)

    def test_02_one_model_forward_initial_identity(self):
        m=UnifiedCalibri()
        h=torch.randn(1,12,4,320)
        v=torch.ones((1,4),dtype=torch.bool)
        with torch.inference_mode():
            old=m.nra3_neural_weights(h,v)
            g,p=m(h,v)
        for name in ("mini_area_logits","role_arc_logits","event_state_logits",
                     "memory_slot_attention","upos_logits","case_logits",
                     "number_logits","voice_logits"):
            self.assertTrue(torch.equal(getattr(old,name),getattr(g,name)),name)
        self.assertTrue(torch.equal(p,old.event_state_logits))

    def test_03_one_optimizer_both_core_and_NRA3R_receive_gradient(self):
        m=UnifiedCalibri()
        for p in m.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        h=torch.randn(1,12,5,320)
        v=torch.ones(1,5,dtype=torch.bool)
        series=train_joint(m,h,v)
        self.assertEqual(len(series),2)
        self.assertTrue(all(t["core_up_gradient_l1"]>0 and
                            t["graph_gate_gradient_l1"]>0 for t in series))
        self.assertFalse(m.candidate_accepted)

    def test_04_cannot_activate_unaccepted_trial(self):
        m=UnifiedCalibri()
        m.output_gate=1
        h=torch.randn(1,12,3,320)
        v=torch.ones(1,3,dtype=torch.bool)
        with self.assertRaises(GateFailure):
            m(h,v)

    def test_05_attach_only_to_research_12layer_not_original_root(self):
        m=UnifiedCalibri()
        class ResearchMock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model=torch.nn.Module()
                self.model.layers=torch.nn.ModuleList([torch.nn.Identity()
                                                       for _ in range(12)])
        mock=ResearchMock()
        m.attach_to_research_only(mock)
        self.assertIs(mock.model.nra3_calibri_joint,m)
        with self.assertRaises(GateFailure):
            m.attach_to_research_only(mock)

if __name__=="__main__":
    unittest.main()
