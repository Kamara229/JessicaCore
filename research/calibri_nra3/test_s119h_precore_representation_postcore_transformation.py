"""S119h synthetic-only fail closed tests; physical S119g source separately."""
import unittest
import torch
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119h_precore_representation_postcore_transformation import (
    NeuralRepresentationArea,NeuralTransformationArea,
    UnifiedCalibriWithRepresentationAndTransformation,
    Causal12LayerMock,model_logits,synthetic_joint_training,
)

class S119HNeuralMathRepresentationAndTransformation(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119)
        torch.set_num_threads(2)

    def test_01_two_areas_are_real_learned_in_one_calibri_model(self):
        m=UnifiedCalibriWithRepresentationAndTransformation()
        names=set(m.state_dict())
        self.assertIn("representation_area.project.weight",names)
        self.assertIn("transformation_area.decode.weight",names)
        self.assertIn("transformation_area.semantic_context.weight",names)
        self.assertEqual(sum(p.numel() for p in
                             m.nra3_neural_weights.parameters()),53751)
        self.assertGreater(sum(p.numel()
                              for p in m.representation_area.parameters()),10000)
        self.assertGreater(sum(p.numel()
                              for p in m.transformation_area.parameters()),10000)
        self.assertFalse(m.candidate_accepted)

    def test_02_exact_original_identity_before_any_training(self):
        m=UnifiedCalibriWithRepresentationAndTransformation()
        core=Causal12LayerMock()
        x=torch.randint(0,37,(2,6))
        with torch.inference_mode():
            before=core.unmodified_logits(x)
            after,_,_,_=model_logits(core,m,x)
        self.assertTrue(torch.equal(before,after))
        self.assertTrue(torch.equal(before.argmax(-1),after.argmax(-1)))

    def test_03_R_is_pre_core_T_after_core_before_vocab_logit(self):
        m=UnifiedCalibriWithRepresentationAndTransformation()
        core=Causal12LayerMock()
        events=[]
        handles=[
            m.representation_area.register_forward_hook(
                lambda a,b,c:events.append("representation")),
            core.layers[0].register_forward_hook(
                lambda a,b,c:events.append("core_first")),
            core.layers[-1].register_forward_hook(
                lambda a,b,c:events.append("core_last")),
            m.transformation_area.register_forward_hook(
                lambda a,b,c:events.append("transformation")),
            core.output_embeddings.register_forward_hook(
                lambda a,b,c:events.append("vocabulary"))]
        try:
            with torch.inference_mode():
                model_logits(core,m,torch.randint(0,37,(1,4)))
            self.assertEqual(events,["representation","core_first",
                                     "core_last","transformation","vocabulary"])
        finally:
            for h in handles:h.remove()

    def test_04_two_areas_dont_hardcode_math_or_delete_words(self):
        R=NeuralRepresentationArea()
        x=torch.randn(2,4,320)
        self.assertTrue(torch.equal(R(x),x))
        self.assertEqual(tuple(R.project.weight.shape),(64,320))
        self.assertEqual(R.router.out_features,4)

    def test_05_joint_math_mock_training_reaches_all_components(self):
        core=Causal12LayerMock()
        m=UnifiedCalibriWithRepresentationAndTransformation()
        for p in m.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        t=synthetic_joint_training(core,m)
        self.assertEqual(len(t),2)
        self.assertTrue(all(all(g>0 for g in x["grad_l1"].values())
                            for x in t))
        self.assertFalse(m.candidate_accepted)

    def test_06_R_or_T_activation_cannot_bypass_gate(self):
        x=torch.randn(1,3,320)
        R=NeuralRepresentationArea()
        R.output_gate=1
        with self.assertRaises(GateFailure):
            R(x)
        T=NeuralTransformationArea()
        T.output_gate=1
        with self.assertRaises(GateFailure):
            T(x,None,None,torch.ones((1,3),dtype=torch.bool))

    def test_07_R_cannot_access_next_token(self):
        r=NeuralRepresentationArea()
        with torch.no_grad():
            r.fuse.weight.fill_(0.01)
        a=torch.randn(1,4,320)
        b=torch.cat((a,torch.randn(1,2,320)*10000),dim=1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(r(a),r(b)[:,:4],
                                           atol=1e-5,rtol=1e-5))

if __name__=="__main__":
    unittest.main()
