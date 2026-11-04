"""Guarantee v99 never mutates original native modules nor leaves hooks installed."""
from __future__ import annotations
import unittest
import torch
from torch import nn
from v99.patch import capture_one,overwrite_one_student_stage,paired_one_token_logits

class Toy(nn.Module):
    def __init__(self,scale):
        super().__init__()
        self.layer=nn.Linear(3,3,bias=False)
        self.head=nn.Linear(3,5,bias=False)
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(3)*scale)
            self.head.weight.fill_(.17)
    def forward(self,input_ids,use_cache=False):
        y=self.layer(input_ids.float())
        return type("Output",(),{"logits":self.head(y)})()

class Tests(unittest.TestCase):
    def setUp(self):
        self.teacher=Toy(2.)
        self.native=Toy(1.)
        self.ids=torch.ones(1,2,3)

    def test_original_and_native_heads_remain_unchanged(self):
        old=self.native.layer.weight.detach().clone()
        t,q=paired_one_token_logits(self.teacher,self.native,self.ids,"layer")
        self.assertTrue(torch.allclose(t,q))
        self.assertTrue(torch.equal(old,self.native.layer.weight))
        self.assertFalse(self.native.layer._forward_hooks)
        self.assertFalse(self.teacher.layer._forward_hooks)

    def test_restore_when_raised_during_capture(self):
        with self.assertRaisesRegex(RuntimeError,"fail"):
            with capture_one(self.teacher,"layer"):
                raise RuntimeError("fail")
        self.assertFalse(self.teacher.layer._forward_hooks)

    def test_restore_when_raised_during_patch(self):
        ref=torch.ones(1,2,3)
        with self.assertRaisesRegex(RuntimeError,"fail"):
            with overwrite_one_student_stage(self.native,"layer",ref):
                self.native(input_ids=self.ids)
                raise RuntimeError("fail")
        self.assertFalse(self.native.layer._forward_hooks)

    def test_bad_geometry_does_not_leave_patch(self):
        ref=torch.ones(1,3,3)
        with self.assertRaises(AssertionError):
            with overwrite_one_student_stage(self.native,"layer",ref):
                self.native(input_ids=self.ids)
        self.assertFalse(self.native.layer._forward_hooks)

    def test_patched_stage_called_exactly_once(self):
        ref=torch.ones(1,2,3)
        with self.assertRaisesRegex(AssertionError,"never patched"):
            with overwrite_one_student_stage(self.native,"layer",ref):
                pass
        self.assertFalse(self.native.layer._forward_hooks)

if __name__=="__main__":
    unittest.main()
