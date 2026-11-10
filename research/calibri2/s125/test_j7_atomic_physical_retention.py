"""Synthetic unit tests ONLY for S125's S87-J7 Adam+parameter transaction.

Never claim these toy tests establish real 456/128 CPU quality.
"""
import copy
import unittest

import torch
from calibri2.s125.j7_atomic_physical_retention import (
    PhysicalRetentionCheck,atomic_train_block,identical_tree,
)


def mock_complete_guard(*,lost_full=0,lost_first=0,
                        reload_ok=True,bpp=.69984,speed=.99):
    return PhysicalRetentionCheck(
        original_456_checked=456,
        original_full_correct=353,
        original_first_correct=387,
        lost_original_full=lost_full,
        lost_original_first=lost_first,
        newly_correct_original_full=0,
        new_S122_train_full=15,
        new_S122_train_both_correct_pairs=0,
        physical_all_inclusive_bpp=bpp,
        paired_original_S101_CPU_speed_ratio=speed,
        physical_FP16_reload_verified=reload_ok,
        immutable_original_S101_bytes_verified=True,
        S95_S98_unchanged=True,
    )


class TestJ7AtomicRollback(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(125)
        self.weight=torch.nn.Parameter(torch.tensor([1.,-2.,.5]))
        self.optimizer=torch.optim.Adam([self.weight],lr=.01)

    def steps(self):
        for _ in range(2):
            self.optimizer.zero_grad(set_to_none=True)
            loss=(self.weight**2).sum()
            loss.backward()
            self.optimizer.step()
        return 2

    def test_safe_physical_guard_commits(self):
        original=self.weight.detach().clone()
        checked=[]
        def guard():
            checked.append(self.weight.detach().clone())
            return mock_complete_guard()
        result=atomic_train_block(
            [self.weight],self.optimizer,self.steps,guard)
        self.assertTrue(result.committed)
        self.assertFalse(result.rollback_verified)
        self.assertEqual(result.block_steps,2)
        self.assertNotEqual(0,float((original-self.weight).abs().sum()))
        self.assertEqual(len(checked),1)
        self.assertEqual(result.learning_rate_after,.01)
        self.assertFalse(result.model_accepted)

    def test_lost_old_EOS_causes_exact_parameter_and_adam_rollback(self):
        original=self.weight.detach().clone()
        old_opt=copy.deepcopy(self.optimizer.state_dict())
        snapshots=[]
        def guard():
            snapshots.append(self.weight.detach().clone())
            return mock_complete_guard(
                lost_full=1 if len(snapshots)==1 else 0)
        result=atomic_train_block([self.weight],self.optimizer,
                                  self.steps,guard)
        self.assertFalse(result.committed)
        self.assertTrue(result.rollback_verified)
        self.assertEqual(len(snapshots),2)
        self.assertFalse(torch.equal(snapshots[0],original))
        self.assertTrue(torch.equal(snapshots[1],original))
        self.assertTrue(torch.equal(self.weight.detach(),original))
        # Full Adam state is exactly restored, not just its step counter.
        latest=copy.deepcopy(self.optimizer.state_dict())
        for group in latest["param_groups"]:
            group["lr"]=.01
        self.assertTrue(identical_tree(latest,old_opt))
        self.assertEqual(result.learning_rate_after,.005)

    def test_fail_closed_on_first_old_token_or_bpp_cpu_speed(self):
        for bad in (
            dict(lost_first=1),dict(bpp=.70001),
            dict(speed=.899),dict(reload_ok=False),
        ):
            with self.subTest(bad=bad):
                weight=torch.nn.Parameter(torch.tensor([1.,-2.,.5]))
                opt=torch.optim.Adam([weight],lr=.01)
                snapshots=[]
                def run():
                    opt.zero_grad()
                    (weight**2).sum().backward()
                    opt.step()
                    return 1
                def guard():
                    snapshots.append(weight.detach().clone())
                    return mock_complete_guard(**bad) if len(snapshots)==1 else mock_complete_guard()
                outcome=atomic_train_block([weight],opt,run,guard)
                self.assertFalse(outcome.committed)
                self.assertTrue(outcome.rollback_verified)
                self.assertTrue(torch.equal(weight.detach(),torch.tensor([1.,-2.,.5])))
                self.assertEqual(len(snapshots),2)

    def test_physical_guard_fails_after_rollback_must_raise(self):
        original=self.weight.detach().clone()
        def guard():
            return mock_complete_guard(lost_full=1)
        with self.assertRaisesRegex(AssertionError,"rollback physically failed"):
            atomic_train_block([self.weight],self.optimizer,
                               self.steps,guard)
        self.assertTrue(torch.equal(original,self.weight.detach()))

    def test_guard_exception_never_leaves_updated_weights(self):
        original=self.weight.detach().clone()
        def boom():
            raise RuntimeError("physical readback error")
        with self.assertRaisesRegex(RuntimeError,"physical readback error"):
            atomic_train_block([self.weight],self.optimizer,self.steps,boom)
        self.assertTrue(torch.equal(original,self.weight.detach()))
        self.assertEqual(self.optimizer.param_groups[0]["lr"],.01)

    def test_no_unregistered_parameters_in_transaction(self):
        stranger=torch.nn.Parameter(torch.tensor([9.]))
        with self.assertRaises(ValueError):
            atomic_train_block([stranger],self.optimizer,self.steps,
                               mock_complete_guard)


if __name__=="__main__":
    unittest.main(verbosity=2)
