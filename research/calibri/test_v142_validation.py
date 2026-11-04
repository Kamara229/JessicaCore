import unittest
import torch
from v142.scoring import compare_logits,aggregate,contrast
from v142.protocol import DOMAINS,ARMS,WINDOW,WINDOWS_PER_DOMAIN,pairs


class ValidScores(unittest.TestCase):
    def test_fixed_source_scope(self):
        self.assertEqual(DOMAINS,("wiki_valid","news_valid"))
        self.assertEqual(ARMS,("FP32","CB59","CB140_v141_LUT"))
        self.assertEqual(len(pairs(torch.arange(256))),WINDOWS_PER_DOMAIN)
        self.assertEqual(pairs(torch.arange(256))[0].shape,(1,WINDOW))
        with self.assertRaises(ValueError):
            pairs(torch.arange(15))

    def test_same_model_teacher_NLL_and_KL(self):
        teacher=torch.tensor([[[1.,2.,3.,0.],[0.,1.,4.,2.],
                               [3.,0.,1.,0.]]])
        input_ids=torch.tensor([[0,2,1]])
        result=compare_logits(teacher,teacher,input_ids)
        self.assertLess(result["sum_teacher_to_student_KL"],1e-8)
        self.assertAlmostEqual(result["sum_NLL"],result["sum_teacher_NLL"],places=7)
        self.assertEqual(result["exact_FP32_next_token_argmax_count"],2)
        aggregate_1=aggregate([result])
        self.assertEqual(aggregate_1["tokens_scored"],2)
        self.assertTrue(aggregate_1["perplexity"]>1)

    def test_nll_regression_detected_on_true_next_tokens(self):
        teacher=torch.tensor([[[0.,0.,8.],[0.,8.,0.],[1.,1.,1.]]])
        worse=torch.tensor([[[8.,0.,0.],[8.,0.,0.],[1.,1.,1.]]])
        inputs=torch.tensor([[0,2,1]])
        ref=compare_logits(teacher,teacher,inputs)
        student=compare_logits(teacher,worse,inputs)
        self.assertGreater(student["sum_NLL"],ref["sum_NLL"])
        self.assertGreater(student["sum_teacher_to_student_KL"],0)
        comparison=contrast({"wiki_valid":[ref]},{"wiki_valid":[student]})
        self.assertGreater(comparison["wiki_valid"]["candidate_minus_baseline_NLL"],0)
        self.assertFalse(comparison["wiki_valid"]["candidate_vs_CB59_NLL_better"])

    def test_bad_geometry_fails_explicitly(self):
        with self.assertRaises(ValueError):
            compare_logits(torch.zeros(1,2,3),torch.zeros(1,3,3),torch.tensor([[1,2]]))
        with self.assertRaises(ValueError):
            aggregate([])


if __name__=="__main__":
    unittest.main()
