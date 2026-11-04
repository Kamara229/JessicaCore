"""Synthetic causal-swap and fixed scope assertions. No original TEST."""
import unittest
import torch
from torch import nn

from v152.protocol import (check,ARMS,LAYERS,
                           DIAGNOSTIC_QUESTION_INDICES,locked_questions)
from v152.oracles import replace_one_layer
from v152.scoring import comparison,aggregate


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList(
            [nn.Linear(576,576,bias=False) for i in range(30)])


class FixedLayerFaultLocalizationTests(unittest.TestCase):
    def test_frozen_original_30_blocks(self):
        self.assertTrue(check())
        self.assertEqual((LAYERS,len(ARMS)),(30,32))
        self.assertEqual(DIAGNOSTIC_QUESTION_INDICES,(0,2,4,6,8,10,12,14))
        self.assertEqual(len(locked_questions()),8)

    def test_swap_restored_even_after_exception(self):
        fake=DummyModel()
        teacher=DummyModel()
        original=tuple(fake.model.layers)
        with self.assertRaisesRegex(ValueError,"synthetic evaluation failure"):
            with replace_one_layer(fake,teacher,17) as status:
                self.assertTrue(status["physical_bpp_requirement_is_VIOLATED"])
                self.assertIs(fake.model.layers[17],teacher.model.layers[17])
                raise ValueError("synthetic evaluation failure")
        self.assertTrue(all(a is b for a,b in zip(fake.model.layers,original)))
        with self.assertRaises(ValueError):
            with replace_one_layer(fake,teacher,30):
                pass

    def test_KL_and_matching_teacher_first_top1_on_synthetic_vocab(self):
        reference=torch.zeros(49152,dtype=torch.float32)
        reference[450]=3.
        duplicate=reference.clone()
        changed=reference.clone()
        changed[220]=5.
        equal=comparison(reference,duplicate,2)
        wrong=comparison(reference,changed,2)
        self.assertTrue(equal["FP32_first_token_argmax_agreement"])
        self.assertFalse(wrong["FP32_first_token_argmax_agreement"])
        self.assertLess(equal["teacher_to_student_KL"],1e-8)
        self.assertGreater(wrong["teacher_to_student_KL"],0.001)
        with self.assertRaises(ValueError):
            comparison(reference,torch.zeros(2048),2)

    def test_scope_errors_not_silent(self):
        records={arm:[{
            "teacher_to_student_KL":9.,
            "FP32_first_token_argmax_agreement":False,
            "first_EOS":False} for i in range(8)]
            for arm in ARMS}
        report=aggregate(records)
        self.assertEqual(len(report["all_30_single_layer_FP32_oracle_ablation_ranking"]),30)
        records["oracle_FP32_layer_00"].pop()
        with self.assertRaises(ValueError):
            aggregate(records)
        records.pop("oracle_FP32_layer_01")
        with self.assertRaises(ValueError):
            aggregate(records)


if __name__=="__main__":
    unittest.main()
