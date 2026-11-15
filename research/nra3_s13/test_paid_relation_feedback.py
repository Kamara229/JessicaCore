"""S13 physical rank8 relation readout CPU causal/gold-free controls."""
from __future__ import annotations
import hashlib
import math
import unittest

import torch
from torch.nn import functional as F

from research.nra3_s13.paid_relation_feedback import (
    MAGIC,RANK,WIDTH,OUT,PARAMETERS,PHYSICAL_BYTES,UPOS_DIM,ROLE_DIM,
    PaidRelationFeedback
)


class PaidCausalRelationFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(313023)
        torch.set_num_threads(1)

    def source(self,b=2,t=9):
        r=torch.randn(b,t,128)
        up=torch.randn(b,t,17)
        sem=torch.randn(b,t,9)
        dep=torch.randn(b,t,t)
        v=torch.ones((b,t),dtype=torch.bool)
        allow=torch.tril(torch.ones((t,t),dtype=torch.bool))[None]
        dep=dep.masked_fill(~allow,-1e4)
        return r,up,sem,dep,v

    def test_physically_charges_exactly_1266_half_fields_sha_and_version(self):
        p=PaidRelationFeedback()
        self.assertEqual(PARAMETERS,1266)
        self.assertEqual(PHYSICAL_BYTES,2581)
        self.assertEqual(sum(x.numel() for x in p.parameters()),PARAMETERS)
        a=p.physical_bytes()
        self.assertEqual(len(a),PHYSICAL_BYTES)
        self.assertEqual(a[:8],MAGIC)
        x=PaidRelationFeedback.from_physical(a)
        self.assertEqual(x.physical_bytes(),a)
        self.assertFalse(any(z.requires_grad for z in x.parameters()))
        for bad in (a[:-1],a+b"\\x00",bytes([a[0]^8])+a[1:],
                    a[:100]+bytes([a[100]^16])+a[101:]):
            with self.assertRaises(ValueError):
                PaidRelationFeedback.from_physical(bad)
        self.assertFalse(x.candidate_accepted)
        self.assertEqual(x.output_gate,0)

    def test_initial_additive_path_exactly_preserves_original_outputs(self):
        model=PaidRelationFeedback().eval()
        r,u,s,d,v=self.source()
        o=model(r,u,s,d,v)
        self.assertTrue(torch.equal(o.upos_logits,u))
        self.assertTrue(torch.equal(o.semantic_role_logits,s))
        self.assertTrue(torch.equal(o.dependency_logits,d))
        self.assertEqual(o.relation_delta.shape,(2,9,26))

    def test_frozen_physical_reload_makes_real_changes_in_all_three_heads(self):
        p=PaidRelationFeedback().eval()
        with torch.no_grad():
            p.output_weight.copy_(torch.randn_like(p.output_weight)*.15)
            p.output_bias.copy_(torch.randn_like(p.output_bias)*.02)
            p.dependency_diagonal.copy_(torch.linspace(-.4,.5,RANK))
        q=PaidRelationFeedback.from_physical(p.physical_bytes())
        r,u,s,d,v=self.source()
        a=p(r,u,s,d,v);b=q(r,u,s,d,v)
        for x,y in zip((a.upos_logits,a.semantic_role_logits,a.dependency_logits),
                       (b.upos_logits,b.semantic_role_logits,b.dependency_logits)):
            self.assertTrue(torch.equal(x,y))
        self.assertFalse(torch.equal(a.upos_logits,u))
        self.assertFalse(torch.equal(a.semantic_role_logits,s))
        self.assertFalse(torch.equal(a.dependency_logits,d))
        future=torch.triu(torch.ones(d.shape[-2:],dtype=torch.bool),diagonal=1)
        self.assertTrue(torch.equal(a.dependency_logits[:,future],d[:,future]))

    def test_prefix_invariance_no_future_text_global_context(self):
        p=PaidRelationFeedback().eval()
        with torch.no_grad():
            p.output_weight.copy_(torch.randn_like(p.output_weight)*.08)
            p.dependency_diagonal.copy_(torch.randn_like(p.dependency_diagonal)*.6)
        r,u,s,d,v=self.source(b=1,t=9)
        old=p(r[:,:5],u[:,:5],s[:,:5],d[:,:5,:5],v[:,:5])
        new=p(r,u,s,d,v)
        self.assertTrue(torch.equal(old.upos_logits,new.upos_logits[:,:5]))
        self.assertTrue(torch.equal(old.semantic_role_logits,new.semantic_role_logits[:,:5]))
        self.assertTrue(torch.equal(old.dependency_logits,new.dependency_logits[:,:5,:5]))
        self.assertTrue(torch.equal(old.relation_rank,new.relation_rank[:,:5]))

    def test_padded_rows_cannot_corrupt_active_prefix(self):
        p=PaidRelationFeedback().eval()
        with torch.no_grad():
            p.output_weight.normal_(0,.04)
            p.dependency_diagonal.fill_(.1)
        r,u,s,d,v=self.source(b=1,t=8)
        v[0,5:]=False
        a=p(r[:,:5],u[:,:5],s[:,:5],d[:,:5,:5],v[:,:5])
        b=p(r,u,s,d,v)
        self.assertTrue(torch.equal(a.upos_logits,b.upos_logits[:,:5]))
        self.assertTrue(torch.equal(a.semantic_role_logits,b.semantic_role_logits[:,:5]))
        self.assertTrue(torch.equal(a.dependency_logits,b.dependency_logits[:,:5,:5]))
        self.assertTrue(bool((b.relation_rank[:,5:]==0).all()))

    def test_rejects_invalid_source_and_activation(self):
        p=PaidRelationFeedback()
        r,u,s,d,v=self.source()
        p.output_gate=1.
        with self.assertRaises(RuntimeError):
            p(r,u,s,d,v)
        with self.assertRaises(RuntimeError):
            p.physical_bytes()
        p.output_gate=0.
        p.candidate_accepted=True
        with self.assertRaises(RuntimeError):
            p(r,u,s,d,v)
        p.candidate_accepted=False
        with self.assertRaises(ValueError):
            p(r,u,s[:,:,:-1],d,v)
        with self.assertRaises(ValueError):
            p(r,u,s,d,v.float())
        with self.assertRaises(ValueError):
            p(r,u,s,d[:,:,:-1],v)
        bad=v.clone()
        bad[:,0]=False
        with self.assertRaises(ValueError):
            p(r,u,s,d,bad)

    def test_only_paid_FP16_matmul_gets_gradients(self):
        p=PaidRelationFeedback()
        r,u,s,d,v=self.source(b=1,t=8)
        assert not any(z.requires_grad for z in (r,u,s,d))
        y=p(r,u,s,d,v)
        loss=F.cross_entropy(y.upos_logits[:,3],torch.tensor([1]))+F.cross_entropy(
            y.semantic_role_logits[:,5],torch.tensor([2]))+(
            y.dependency_logits[:,4,2]-2.3).square().mean()*.01
        loss.backward()
        self.assertTrue(bool((p.output_weight.grad!=0).any()))
        self.assertTrue(bool((p.output_bias.grad!=0).any()))
        self.assertTrue(bool((p.dependency_diagonal.grad!=0).any()))
        self.assertEqual(sum(z.numel() for z in p.parameters()),1266)


if __name__=="__main__":
    unittest.main()
