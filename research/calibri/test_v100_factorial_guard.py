"""Check v100 FP32 hidden and head factors are reversible on toy decoders."""
from __future__ import annotations
from contextlib import contextmanager
from unittest.mock import patch
import unittest
import torch
from torch import nn
from v100.factors import factor_logits
from v99.patch import native_logits

class Body(nn.Module):
    def __init__(self,value):
        super().__init__()
        self.embed=nn.Embedding(10,3)
        self.norm=nn.Identity()
        with torch.no_grad():self.embed.weight.fill_(value)
    def forward(self,input_ids):
        return self.norm(self.embed(input_ids))

class Tiny(nn.Module):
    def __init__(self,value,head_value):
        super().__init__()
        self.model=Body(value)
        self.lm_head=nn.Linear(3,5,bias=False)
        with torch.no_grad():self.lm_head.weight.fill_(head_value)
    def forward(self,input_ids,use_cache=False):
        hidden=self.model(input_ids)
        return type("Output",(),{"logits":self.lm_head(hidden)})()

@contextmanager
def pretend_FP32_module_swap(student,teacher,paths):
    if paths!=("lm_head",):raise AssertionError("head only")
    previous=student.lm_head
    try:
        student.lm_head=teacher.lm_head
        yield
    finally:student.lm_head=previous

class Tests(unittest.TestCase):
    def setUp(self):
        self.teacher=Tiny(3.,.4)
        self.student=Tiny(1.,.25)
        self.ids=torch.ones((1,3),dtype=torch.long)
    def test_all_four_factorial_states_different_and_restored(self):
        original_head=self.student.lm_head
        original_weight=original_head.weight.detach().clone()
        norms=[]
        with patch("v100.factors.temporary_oracle_modules",pretend_FP32_module_swap):
            for factor in ("native","norm_only","head_only","norm_and_head"):
                output=factor_logits(self.teacher,self.student,self.ids,factor)
                norms.append(float(output.max().item()))
                self.assertIs(self.student.lm_head,original_head)
                self.assertFalse(self.student.model.norm._forward_hooks)
        self.assertEqual(len(set(norms)),4)
        self.assertTrue(torch.equal(original_weight,original_head.weight))
        self.assertTrue(torch.allclose(factor_logits(self.teacher,self.student,self.ids,"native"),
                                      native_logits(self.student,self.ids)))
    def test_invalid_factor_rejected_before_mutation(self):
        with self.assertRaises(AssertionError):
            factor_logits(self.teacher,self.student,self.ids,"something_else")
        self.assertFalse(self.student.model.norm._forward_hooks)

if __name__=="__main__":unittest.main()
