"""Oscar v1D no-leak teacher-on-student-prefix and causal EOS-loss contracts."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from oscar.v1d.onpolicy_teacher_distill import (
    teacher_on_actual_student_prefixes,
    loss_against_real_student_policy,
    MAX_STUDENT_PREFIX,
)


class FakeTokenizer:
    vocab_size=32
    eos_token_id=3

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages[0]["role"]=="user"
        assert tokenize is False and add_generation_prompt is True
        return "synthetic-local-only"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [1,2]


class FakeIndependentPretrainedTeacher(torch.nn.Module):
    """Next-token teacher argmax encodes the *actually given* last token."""
    def forward(self,input_ids,*,use_cache=False):
        n=int(input_ids.shape[1])
        result=torch.full((1,n,32),-5.,dtype=torch.float32)
        chosen=(int(input_ids[0,-1])+1)%32
        result[0,-1,chosen]=6.
        return SimpleNamespace(logits=result)


class OscarRealOnpolicyContracts(unittest.TestCase):
    def test_teacher_conditions_on_student_not_on_external_target(self):
        t=FakeIndependentPretrainedTeacher()
        tok=FakeTokenizer()
        row={"question":"not an external benchmark answer"}
        result=teacher_on_actual_student_prefixes(
            t,tok,row,[10,11,tok.eos_token_id])
        self.assertEqual(len(result),3)
        self.assertEqual([r["source_actual_student_token_id"] for r in result],
                         [10,11,3])
        # First teacher sees prompt ending in 2, then student-greedy 10 and 11.
        self.assertEqual([r["original_teacher_target_id_on_student_prefix"]
                           for r in result],[3,11,12])
        self.assertTrue(result[2]["actual_student_emitted_EOS"])
        self.assertFalse(result[2]["teacher_EXPECTED_EOS"])
        with self.assertRaises(AssertionError):
            teacher_on_actual_student_prefixes(t,tok,row,[10,3,11])

    def test_onpolicy_objective_gets_actual_trainable_grad_and_first_anchor(self):
        torch.manual_seed(103)
        first=torch.randn(32,requires_grad=True)
        second=torch.randn(32,requires_grad=True)
        old=first.detach().clone()*1.05
        logits=[first,second]
        teacher_rows=[]
        for i,predicted in enumerate((7,8)):
            teacher=torch.randn(32)
            teacher[predicted]=7
            teacher_rows.append({
                "whole_vocab_target":teacher,
                "original_teacher_target_id_on_student_prefix":predicted,
                "source_actual_student_token_id":[6,3][i],
            })
        objective,trace=loss_against_real_student_policy(
            logits,teacher_rows,old,FakeTokenizer.eos_token_id)
        self.assertTrue(torch.isfinite(objective))
        objective.backward()
        self.assertGreater(float(first.grad.norm()),0)
        self.assertGreater(float(second.grad.norm()),0)
        self.assertEqual(trace[0]["position_objective_weight"],3.0)
        self.assertGreater(trace[0]["original_v1B_first_token_anchor_KL"],0)
        self.assertEqual(trace[1]["original_v1B_first_token_anchor_KL"],0)
        self.assertGreater(trace[1]["EOS_correction"],0)

    def test_source_no_fabricated_post_eos_trajectory(self):
        tok=FakeTokenizer()
        with self.assertRaises(AssertionError):
            teacher_on_actual_student_prefixes(
                FakeIndependentPretrainedTeacher(),tok,
                {"question":"source"},[3,5])

    def test_max_student_prefix_is_bounded(self):
        self.assertEqual(MAX_STUDENT_PREFIX,4)


if __name__=="__main__":
    unittest.main()
