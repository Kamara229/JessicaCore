"""Fast regression of the frozen model's read-only CPU layer diagnostics."""
from __future__ import annotations
import unittest
import torch
from torch import nn
from v98.metrics import drift,logits_topk,layer_delta,rank_first_divergences
from v98.hooks import capture_stages

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.a=nn.Linear(3,3,bias=False)
        self.b=nn.ReLU()
    def forward(self,x):return self.b(self.a(x))

class T(unittest.TestCase):
    def test_identical_hidden_zero_error(self):
        a=torch.tensor([[[1.,2.,-1.]]])
        v=drift(a,a.clone())
        self.assertLess(v["relative_rms_error"],1e-9)
        self.assertAlmostEqual(v["cosine_similarity"],1.,places=6)
        self.assertEqual(layer_delta(v,v),0.)

    def test_known_ratio_and_first_crossing(self):
        a=torch.ones((1,3))
        value=drift(a,a*1.5)
        self.assertAlmostEqual(value["relative_rms_error"],.5,places=6)
        previous={"relative_rms_error":.1}
        rows=[{"stage":"embedding","metrics":previous,"delta_from_previous_stage":None},
              {"stage":"layer0","metrics":value,"delta_from_previous_stage":.4}]
        ranked=rank_first_divergences(rows)
        self.assertEqual(ranked["first_relative_rms_above_0p25"],"layer0")
        self.assertEqual(ranked["first_relative_rms_above_0p50"],"layer0")
        self.assertEqual(ranked["top_positive_drift_jumps"][0]["stage"],"layer0")

    def test_nonfinite_or_mismatched_residual_rejected(self):
        a=torch.ones(2)
        with self.assertRaises(AssertionError):drift(a,torch.ones(3))
        with self.assertRaises(AssertionError):drift(a,torch.tensor([float("nan"),0.]))

    def test_hook_cleanup_when_success_and_exception(self):
        m=Tiny()
        with capture_stages(m,["a","b"]) as captured:
            m(torch.ones(1,3))
            self.assertEqual(set(captured),{"a","b"})
        self.assertEqual(len(m.a._forward_hooks),0)
        self.assertEqual(len(m.b._forward_hooks),0)
        with self.assertRaisesRegex(RuntimeError,"forced"):
            with capture_stages(m,["a"]):
                raise RuntimeError("forced")
        self.assertEqual(len(m.a._forward_hooks),0)

    def test_logit_entropy_finite_and_topk(self):
        class Tok:
            def decode(self,ids):return str(ids[0])
        result=logits_topk(torch.tensor([0.,1.,2.]),Tok())
        self.assertEqual(result["top10_token_ids"][0],2)
        self.assertAlmostEqual(result["top10_probability_mass"],1.,places=5)
        self.assertGreater(result["entropy_nats"],0)
        with self.assertRaises(AssertionError):
            logits_topk(torch.tensor([0.,float("inf")]),Tok())

if __name__=="__main__":unittest.main()
