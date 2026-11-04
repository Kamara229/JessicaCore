"""Causal comparison fixtures: an improved local weight may not undo input drift."""
import unittest
import torch
from v158.protocol import (check,FIT_DOMAINS,WINDOW,TOKENS_EACH,
                           OPERATOR,KEY)
from v158.audit import similarity,decompose,summary


class OriginalVprojDriftTests(unittest.TestCase):
    def test_exact_original_FIT_scope_and_module(self):
        self.assertTrue(check())
        self.assertEqual(FIT_DOMAINS,("wiki_fit","news_fit"))
        self.assertEqual(TOKENS_EACH,256)
        self.assertEqual(WINDOW,64)
        self.assertEqual(OPERATOR,"model.layers.6.self_attn.v_proj")
        self.assertTrue(KEY.endswith(".v_proj.weight"))

    def test_same_input_weight_only_error_can_improve_while_combined_worsens(self):
        torch.manual_seed(158)
        original=torch.randn(256,8)
        drifted=original*2.5
        # Reference FP32 original linear W = identity on eight dimensions.
        teacher_output=original.clone()
        f32_of_student=drifted.clone()
        # Native 5-bit local approximation is conservative while the
        # paid one-bit output is closer on teacher inputs, but scale-
        # dependent upstream drift can still dominate the REAL result.
        cb59_teacher=original*.8
        cb59_student=drifted*.8
        paid_teacher=original*.95
        paid_student=drifted*.95
        outcome=decompose(
            original,drifted,teacher_output,f32_of_student,
            cb59_teacher,cb59_student,paid_teacher,paid_student)
        self.assertLess(
            outcome["paid_1bit_weight_ONLY_error_at_FP32_teacher_input"]["relative_RMSE"],
            outcome["original_CB59_weight_ONLY_error_at_FP32_teacher_input"]["relative_RMSE"])
        self.assertGreater(
            outcome["paid_1bit_COMBINED_input_and_weight_error_against_FP32"]["relative_RMSE"],
            outcome["native_CB59_COMBINED_input_and_weight_error_against_FP32"]["relative_RMSE"])
        findings=summary({k:outcome for k in FIT_DOMAINS})
        for domain in FIT_DOMAINS:
            item=findings["original_TRAlN_FIT_only_domain_findings"][domain]
            self.assertTrue(item["paid_1bit_reduces_original_weight_error_on_teacher_inputs"])
            self.assertFalse(item["paid_1bit_reduces_combined_actual_original_operator_error"])
            self.assertTrue(item["upstream_drift_dominates_original_weight_only_error"])

    def test_finite_shape_and_complete_domain_checks(self):
        a=torch.ones(256,8)
        b=torch.zeros(256,8)
        self.assertAlmostEqual(similarity(a,a)["relative_RMSE"],0)
        self.assertAlmostEqual(similarity(a,b)["relative_RMSE"],1)
        with self.assertRaises(ValueError):
            similarity(a,b[:-1])
        with self.assertRaises(ValueError):
            similarity(a,torch.full((256,8),float("nan")))
        with self.assertRaises(ValueError):
            summary({FIT_DOMAINS[0]:{}})

if __name__=="__main__":
    unittest.main()
