"""S193 synthetic-only unit contracts; NO real 43M weights or CPU J7 here.

Mocked typed certificates exercise future independently measured 456+128+32
full CPU gate and exact Adam rollback logic. Never describe as J7 result.
"""
from __future__ import annotations
import copy
import hashlib
import unittest
import torch
from calibri2.s125.j7_atomic_physical_retention import identical_tree
from calibri2.s193.atomic_one_step_584 import J7Retention584,one_protected_step


def digest(b):
    return hashlib.sha256(b).hexdigest()


class SyntheticProof:
    """Synthetic callback only. Real J7 runner must replace this entirely."""
    def __init__(self,p,kind="good",bad_baseline=False):
        self.p=p
        self.kind=kind
        self.initial=p.detach().clone()
        self.calls=0
        self.bad_baseline=bad_baseline

    def __call__(self):
        self.calls+=1
        if self.kind=="throw" and self.calls==2:
            raise RuntimeError("synthetic candidate CPU decoder failure")
        candidate=not torch.equal(self.p.detach(),self.initial)
        if self.bad_baseline and not candidate:
            return self._proof(True,False,False)
        return self._proof(candidate,True,True)

    def _proof(self,candidate,good,real_flag):
        # SHA of actual small mock tensor bytes, NOT neural model/checkpoint.
        chk=digest(bytes(self.p.detach().cpu().contiguous().view(torch.uint8).tolist()))
        old_dig=digest(b"immutable original 584 generated full token IDs")
        if candidate and self.kind=="regress":
            old_dig=digest(b"unsafe protected 584 generated token changes")
        if candidate and self.kind=="no_gain":
            new_full=0
            new_first=16
        elif candidate and self.kind=="lose_first":
            new_full=1
            new_first=15
        elif candidate:
            new_full=1
            new_first=17
        else:
            new_full=0
            new_first=16
        if candidate and self.kind=="too_slow":
            speed=.89
        else:
            speed=1.02
        if candidate and self.kind=="bpp_regress":
            bpp=.705
        else:
            bpp=.692
        return J7Retention584(
            actual_cpu_j7=real_flag,
            old_checked=456,
            old_full_eos_correct=456,
            new_checked=128,
            new_full_eos_correct=128,
            old_first_lost=0,
            old_full_lost=0,
            new_first_lost=0,
            new_full_lost=0,
            protected_584_generation_digest=old_dig,
            train32_checked=32,
            train32_first_correct=new_first,
            train32_full_eos_correct=new_full,
            train32_generation_digest=digest(
                b"new32 baseline" if not candidate else b"new32 candidate"),
            serialized_fp16_reload_verified=good,
            frozen_parent_weights_unchanged=True,
            no_reference_data_at_inference=True,
            source_only_train_no_QA_TEST=True,
            same_CPU_speed_vs_S187=speed,
            physical_all_inclusive_bpp=bpp,
            checkpoint_bytes_sha256=chk,
        )


class S193SingleStepAtomic(unittest.TestCase):
    def make(self):
        p=torch.nn.Parameter(torch.tensor([0.5,-0.5],dtype=torch.float32))
        opt=torch.optim.Adam([p],lr=.004)
        # Create real optimizer moment tensors before the transaction so
        # rollback verifies the full internal Adam state, not empty dict.
        opt.zero_grad()
        (p.square().sum()).backward()
        opt.step()
        initial=p.detach().clone()
        state=copy.deepcopy(opt.state_dict())
        def step():
            opt.zero_grad(set_to_none=True)
            ((p-1).square().sum()).backward()
            opt.step()
            return 1
        return p,opt,initial,state,step

    def test_single_synthetic_success_guard_commits_only_gain(self):
        p,opt,before,adam,step=self.make()
        gate=SyntheticProof(p,"good")
        out=one_protected_step([p],opt,step,gate)
        self.assertTrue(out.committed)
        self.assertFalse(out.model_accepted)
        self.assertEqual(out.new_train_full_eos_gain,1)
        self.assertEqual(gate.calls,2)
        self.assertFalse(torch.equal(p.detach(),before))
        self.assertFalse(identical_tree(opt.state_dict(),adam))

    def test_synthetic_regression_no_gain_speed_bpp_and_first_rollback(self):
        for kind in ("regress","no_gain","too_slow","bpp_regress","lose_first"):
            with self.subTest(kind=kind):
                p,opt,before,adam,step=self.make()
                gate=SyntheticProof(p,kind)
                result=one_protected_step([p],opt,step,gate)
                self.assertFalse(result.committed)
                self.assertTrue(result.rollback_verified)
                self.assertFalse(result.model_accepted)
                self.assertEqual(result.protected_count,584)
                self.assertEqual(gate.calls,3)
                self.assertTrue(torch.equal(p.detach(),before))
                restored=copy.deepcopy(adam)
                restored["param_groups"][0]["lr"] *= .5
                self.assertTrue(identical_tree(opt.state_dict(),restored))

    def test_candidate_physical_j7_exception_rolls_back_before_re_raise(self):
        p,opt,before,adam,step=self.make()
        gate=SyntheticProof(p,"throw")
        with self.assertRaisesRegex(AssertionError,"physically rolled back"):
            one_protected_step([p],opt,step,gate)
        self.assertTrue(torch.equal(p.detach(),before))
        self.assertTrue(identical_tree(opt.state_dict(),adam))
        self.assertEqual(gate.calls,3)

    def test_missing_real_j7_baseline_rejected_without_training(self):
        p,opt,before,adam,step=self.make()
        gate=SyntheticProof(p,"good",bad_baseline=True)
        with self.assertRaisesRegex(AssertionError,"baseline physical real CPU"):
            one_protected_step([p],opt,step,gate)
        self.assertTrue(torch.equal(p.detach(),before))
        self.assertTrue(identical_tree(opt.state_dict(),adam))
        self.assertEqual(gate.calls,1)

    def test_bounded_one_step_exact_enforced_with_atomic_rollback(self):
        p,opt,before,adam,step=self.make()
        gate=SyntheticProof(p,"good")
        def two():
            step()
            return 2
        with self.assertRaisesRegex(AssertionError,"physically rolled back"):
            one_protected_step([p],opt,two,gate)
        self.assertTrue(torch.equal(p.detach(),before))
        self.assertTrue(identical_tree(opt.state_dict(),adam))

    def test_refuses_fake_584_certificate_even_at_full_count(self):
        p,opt,before,adam,step=self.make()
        gate=SyntheticProof(p,"good")
        parent=gate()
        invalid=[
            {"actual_cpu_j7":False},
            {"old_checked":455},
            {"new_full_eos_correct":127},
            {"old_full_lost":1},
            {"old_first_lost":1},
            {"new_first_lost":1},
            {"new_full_lost":1},
            {"physical_all_inclusive_bpp":.701},
            {"same_CPU_speed_vs_S187":.89},
            {"serialized_fp16_reload_verified":False},
            {"frozen_parent_weights_unchanged":False},
            {"no_reference_data_at_inference":False},
            {"train32_checked":31},
            {"checkpoint_bytes_sha256":"not-sha"},
        ]
        for change in invalid:
            with self.subTest(change=change):
                fields={**parent.__dict__,**change}
                self.assertFalse(J7Retention584(**fields).compatible())

if __name__=="__main__":
    unittest.main()
