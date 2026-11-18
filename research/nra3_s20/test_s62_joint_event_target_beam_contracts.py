"""Six S62 source-only structural tests; NOT HELD/J7 evidence."""
import unittest
import torch
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    K,PAIR_D,SOURCE_D,PAID,Beam,S62JointEventRecipient,
)

def sample():
    n,w=2,5
    source=torch.ones((n,K,SOURCE_D))
    source[:,1,:]=2.
    feats=torch.ones((n,K,w,PAIR_D))
    old=torch.full((n,K,w+1),-1e9)
    old[:,0,3]=2.
    old[:,0,w]=0.
    old[:,1,4]=3.
    old[:,1,w]=-.5
    polar=torch.zeros((n,K,w))
    polar[:,1,4]=1.
    allowed=torch.zeros((n,K),dtype=torch.bool)
    allowed[:,:2]=True
    safe=torch.zeros((n,K,w),dtype=torch.bool)
    safe[:,0,3]=True
    safe[:,1,4]=True
    sources=torch.zeros((n,K),dtype=torch.long)
    sources[:,0]=1;sources[:,1]=2
    base=torch.zeros((n,K));base[:,0]=1.
    return Beam(sources,allowed,base,source,feats,old,polar,safe,w)

class TestS62Joint(unittest.TestCase):
    def test_abi_fits_physical2166(self):
        self.assertEqual((SOURCE_D,PAIR_D,K,PAID),(693,387,8,2166))
        m=S62JointEventRecipient()
        self.assertEqual(m.fp16().shape,(PAID,))

    def test_safe_joint_source_target_event(self):
        b=sample()
        m=S62JointEventRecipient().eval().requires_grad_(False)
        got=m.decode(b)
        self.assertEqual(int(got["directed_recipient"].sum()),2)
        self.assertTrue(bool(got["directed_recipient"][:,2,4].all()))
        self.assertTrue(bool(got["directed_rejected"][:,2,4].all()))
        self.assertEqual(got["output_gate"],0)
        self.assertFalse(got["candidate_accepted"])

    def test_none_event_candidate_is_valid_abstention(self):
        b=sample()
        m=S62JointEventRecipient()
        with torch.no_grad():
            m.none_bias.fill_(200.)
        got=m.eval().requires_grad_(False).decode(b)
        self.assertEqual(got["no_target_abstentions"],2)
        self.assertEqual(int(got["directed_recipient"].sum()),0)

    def test_joint_event_reordering_from_pair_not_fixed_event(self):
        b=sample()
        m=S62JointEventRecipient()
        with torch.no_grad():
            m.pair_bias.fill_(10.)
            # source 2 gets larger original pair despite eventbase source1=1.
        result=m.eval().requires_grad_(False).decode(b)
        self.assertTrue(bool(result["directed_recipient"][:,2,4].all()))

    def test_fp16_exact_reload(self):
        m=S62JointEventRecipient()
        with torch.no_grad():
            m.event[21]=.125;m.pair[31]=-.25;m.polar[8]=.015
        z=m.fp16()
        obj=S62JointEventRecipient.from_fp16(z)
        self.assertTrue(torch.equal(obj.fp16(),z))
        self.assertTrue(all(not p.requires_grad for p in obj.parameters()))

    def test_gradient_all_heads_and_fail_closed(self):
        b=sample()
        m=S62JointEventRecipient()
        v,p=m(b)
        loss=torch.nn.functional.cross_entropy(
            v[:,:2].reshape(2,-1),torch.tensor([4,4]))
        loss=loss+torch.nn.functional.binary_cross_entropy_with_logits(
            p[:,1,4],torch.tensor([1.,0.]))
        loss.backward()
        self.assertTrue(all(z.grad is not None and bool(torch.isfinite(z.grad).all())
                            for z in m.parameters()))
        m.candidate_accepted=True
        with self.assertRaisesRegex(ValueError,'gate cannot be activated'):
            m.decode(b)

if __name__=="__main__":unittest.main()
