"""S102: S101 quality anchor + J7 atomic rollback, research ONLY.

This is a generic transaction gate to be called by a future MPS TRAIN step.
It never writes LIVE weights, learns from sealed QA, or uses prompt IDs in
inference. It protects all existing correct TRAIN answers from the frozen
S101 autoregressive policy, not only S82's 196; it additionally protects all
previously correct FIRST tokens. No numerical model weight is promoted here.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Sequence
import torch


MAX_PHYSICAL_BPP = 0.7
MIN_CPU_RATIO = 0.90
EXPECTED_TRAIN_ROWS = 456
EXPECTED_PROTECTED_FULL = 353


def strict_train_retention(
    expected: Mapping[str, Sequence[int]],
    baseline: Mapping[str, Sequence[int]],
    proposed: Mapping[str, Sequence[int]],
    *,
    physical_bpp: float,
    cpu_ratio: float | None = None,
    speed_required: bool = False,
) -> dict:
    """Compare FULL token IDs incl EOS for EVERY authorized TRAIN identity.

    The 353 correct S101 full+EOS responses are a minimum protected set.
    Every previously correct first token must ALSO remain correct. The
    reference is immutable S101, not S82, so later gains cannot erase S99.
    """
    keys = set(expected)
    if len(keys) != EXPECTED_TRAIN_ROWS or keys != set(baseline) or keys != set(proposed):
        raise AssertionError("Complete identical 456 TRAIN identities required")
    original_full = {k for k in keys if list(baseline[k]) == list(expected[k])}
    original_first = {k for k in keys if baseline[k] and baseline[k][0] == expected[k][0]}
    if len(original_full) != EXPECTED_PROTECTED_FULL:
        raise AssertionError("Frozen S101 353 full+EOS protected baseline changed")
    lost_full = sorted(k for k in original_full if list(proposed[k]) != list(expected[k]))
    lost_first = sorted(k for k in original_first if not proposed[k] or
                        proposed[k][0] != expected[k][0])
    gained_full = sorted(k for k in keys - original_full
                         if list(proposed[k]) == list(expected[k]))
    if physical_bpp < 0 or not float("-inf") < physical_bpp < float("inf"):
        raise AssertionError("Invalid physical bpp")
    if speed_required and (cpu_ratio is None or not float("-inf") < cpu_ratio < float("inf")):
        raise AssertionError("Missing measured CPU ratio")
    passed = (not lost_full and not lost_first and
              physical_bpp <= MAX_PHYSICAL_BPP and
              (not speed_required or cpu_ratio >= MIN_CPU_RATIO))
    return {
        "all_456_verified": True,
        "old_full_correct_protected": len(original_full),
        "old_first_correct_protected": len(original_first),
        "lost_full": len(lost_full),
        "lost_first": len(lost_first),
        "gained_full": len(gained_full),
        "lost_full_identities": lost_full,
        "lost_first_identities": lost_first,
        "gained_full_identities": gained_full,
        "physical_bpp": physical_bpp,
        "cpu_ratio": cpu_ratio,
        "speed_required": speed_required,
        "transaction_can_commit": passed,
        "not_production_acceptance": True,
    }


class AtomicProtectedUpdate:
    """Snapshots ALL trainable tensors + full Adam state before a transaction.

    Call once per small group of GPU steps; run a serialized CB59+real S101
    CPU greedy full+EOS 456-row guard; commit only if every old identity
    remains correct and complete physical bpp holds. On any regression, undo
    both weights and momentum and verify the numerical rollback.
    """

    def __init__(self, parameters: Sequence[torch.nn.Parameter], optimizer):
        self.parameters = tuple(parameters)
        if not self.parameters or len({id(x) for x in self.parameters}) != len(self.parameters):
            raise AssertionError("Nonempty unique trainable parameter list required")
        self.optimizer = optimizer
        self._original = [p.detach().clone() for p in self.parameters]
        self._optimizer_state = copy.deepcopy(optimizer.state_dict())
        self._done = False

    def rollback(self):
        if self._done:
            raise RuntimeError("Transaction has already been finalized")
        with torch.no_grad():
            for tensor, old in zip(self.parameters, self._original):
                tensor.copy_(old)
                tensor.grad = None
        self.optimizer.load_state_dict(copy.deepcopy(self._optimizer_state))
        for tensor, old in zip(self.parameters, self._original):
            if not torch.equal(tensor.detach(), old):
                raise AssertionError("J7 rollback failed to restore learned weights")
        # Adam step/moments must roll back as well; no stale optimizer state.
        def exact(left, right):
            if isinstance(left, torch.Tensor):
                return (isinstance(right, torch.Tensor) and torch.equal(left, right)
                        and left.dtype == right.dtype)
            if isinstance(left, dict):
                return (isinstance(right, dict) and left.keys() == right.keys()
                        and all(exact(left[k], right[k]) for k in left))
            if isinstance(left, (list, tuple)):
                return (type(left) is type(right) and len(left) == len(right)
                        and all(exact(a,b) for a,b in zip(left,right)))
            return left == right
        if not exact(self.optimizer.state_dict(), self._optimizer_state):
            raise AssertionError("J7 rollback failed to restore Adam state")
        self._done = True

    def finalize(self, guard: dict, *, half_lr_if_rejected=True) -> dict:
        if self._done:
            raise RuntimeError("Transaction has already been finalized")
        if not guard.get("all_456_verified"):
            raise AssertionError("Refusing unverified 456-row guard")
        if guard.get("old_full_correct_protected") != EXPECTED_PROTECTED_FULL:
            raise AssertionError("Incorrect frozen 353 baseline")
        passed = bool(guard.get("transaction_can_commit"))
        if not passed:
            self.rollback()
            if half_lr_if_rejected:
                for group in self.optimizer.param_groups:
                    group["lr"] *= 0.5
        else:
            self._done = True
        return {
            "committed": passed,
            "rollback_verified": not passed,
            "lost_full": guard["lost_full"],
            "lost_first": guard["lost_first"],
            "gained_full": guard["gained_full"],
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "not_production_acceptance": True,
        }


def numerical_rollback_selftest() -> dict:
    """Only a synthetic optimizer transaction test; never a Jessica/QA test."""
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = torch.optim.Adam([p], lr=0.03)
    p.grad = torch.tensor([.5, -.25])
    opt.step()
    opt.zero_grad(set_to_none=True)
    base_p = p.detach().clone()
    base_state = copy.deepcopy(opt.state_dict())
    tx = AtomicProtectedUpdate([p], opt)
    p.grad = torch.tensor([.1, .8])
    opt.step()
    opt.zero_grad(set_to_none=True)
    expected = {str(i): [1, 2, 3] for i in range(456)}
    old = {str(i): [1, 2, 3] if i < 353 else [1, 0, 3] for i in range(456)}
    candidate = dict(old)
    candidate["0"] = [1, 5, 3]
    candidate["400"] = [1, 2, 3]
    guarded = strict_train_retention(
        expected, old, candidate, physical_bpp=.69957, cpu_ratio=1.01,
        speed_required=True)
    if guarded["lost_full"] != 1 or guarded["gained_full"] != 1:
        raise AssertionError("Synthetic full+EOS guard did not detect forgetting")
    result = tx.finalize(guarded)
    if result["committed"] or not result["rollback_verified"] or not torch.equal(p,base_p):
        raise AssertionError("Synthetic rollback did not restore model")
    for k,v in base_state["state"].items():
        for name, value in v.items():
            current = opt.state_dict()["state"][k][name]
            if isinstance(value,torch.Tensor) and not torch.equal(value,current):
                raise AssertionError("Synthetic rollback did not restore Adam moments")
    # A non-regressive new gain must be allowed without mutating inference ABI.
    candidate["0"] = old["0"]
    success = strict_train_retention(
        expected, old, candidate, physical_bpp=.69957, cpu_ratio=1.01,
        speed_required=True)
    if not success["transaction_can_commit"] or success["gained_full"] != 1:
        raise AssertionError("Synthetic safe learning should be eligible")
    return {"event":"S102_SYNTHETIC_ATOMIC_ROLLBACK_PASS",
            "full_protected":353,"first_protected":456,
            "detected_lost":1,"rollback_weights_and_Adam":True,
            "safe_gain_eligible":True}
