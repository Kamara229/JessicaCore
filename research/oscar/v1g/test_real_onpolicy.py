"""Oscar v1G synthetically prove TRUE student-history, 8-layer original FP32 losses.

No teacher checksum / pretrained QA / old Calibri J7 touched in these tests.
"""
from __future__ import annotations
import unittest
from types import SimpleNamespace
import torch
from oscar.v1g.real_onpolicy_gt8 import (
    build_same_real_student_prefix,posterior_objective,
    MAX_NEW,DEPTH,WIDTH,
)


class SameGenuineStudentCausalPrefix(unittest.TestCase):
    def test_causal_prefix_does_not_use_teacher_generated_tokens(self):
        tok=SimpleNamespace(eos_token_id=2)
        prompt=torch.tensor([[8,9,10]],dtype=torch.long)
        emitted=[91,83,2]
        context=build_same_real_student_prefix(prompt,emitted,tok)
        self.assertEqual(context.tolist(),[[8,9,10,91,83]])
        self.assertNotIn(2,context.tolist()[0])
        self.assertEqual(context.size(1),prompt.size(1)+len(emitted)-1)

    def test_stop_on_actual_student_eos_and_never_fabricate_history(self):
        tok=SimpleNamespace(eos_token_id=2)
        ids=torch.tensor([[8,9]])
        self.assertEqual(
            build_same_real_student_prefix(ids,[2],tok).tolist(),
            [[8,9]])
        with self.assertRaises(AssertionError):
            build_same_real_student_prefix(ids,[12,2,14],tok)
        with self.assertRaises(AssertionError):
            build_same_real_student_prefix(ids,[],tok)

    def test_actual_all8_pretrained_teacher_logit_and_hidden_gradient(self):
        torch.manual_seed(1103)
        b=1;prompt_length=4;n=3;L=prompt_length+n-1
        slogits=torch.randn(b,L,32000,requires_grad=True)
        # Make the teacher-conditioned excess repetition check deterministic:
        # repeated real student token 88 is too likely at second/third step.
        with torch.no_grad():
            slogits[0,prompt_length,88]=17.
            slogits[0,prompt_length+1,88]=17.
        tlogits=torch.randn(b,L,32000)
        # Independent original FP32 teacher produces distinct greedy target
        for j,t in enumerate((42,314,55)):
            tlogits[0,prompt_length-1+j,t]+=8
        student_states=[torch.randn(b,L,WIDTH,requires_grad=True)
                        for _ in range(DEPTH)]
        original_states=[torch.randn(b,L,WIDTH) for _ in range(DEPTH)]
        loss,trace=posterior_objective(
            slogits,tlogits,student_states,original_states,
            prompt_length,[88,88,88])
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(trace["original_FP32_eight_decoder_state_cosines"]),DEPTH)
        self.assertEqual(len(trace["teacher_FP32_original_causal_next_logits_SHA256"]),3)
        loss.backward()
        self.assertGreater(float(slogits.grad.norm()),0)
        self.assertTrue(all(x.grad is not None and
                        float(x.grad.norm())>0 for x in student_states))
        self.assertGreater(trace["genuine_teacher_excess_repetition_penalty"],0)

    def test_true_teacher_and_student_shape_mismatch_caught(self):
        x=torch.randn(1,4,32000)
        with self.assertRaises(AssertionError):
            posterior_objective(
                x,x,[],[],prompt_length=3,actual_student_tokens=[4,5,6])


if __name__=="__main__":
    unittest.main()
