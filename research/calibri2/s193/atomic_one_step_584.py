"""S193 ONE real-CPU-J7-gated transaction primitive: preflight only.

S125's PhysicalRetentionCheck.valid() hardcodes 353/387 and CANNOT
protect S187's complete 456/456 + S122 128/128. This new strict 584
contract requires FULL per-example prediction identity digests, full
first token and EOS, physical FP16 reload, unchanged legacy parent
weights, same CPU speed and all-in bpp before committing an update.
Run *synthetic* rollback/commit/exception cases in S193 CI only.
S193 must never train the real 43M model or claim neural gain.
Future actual runner must compute every certificate via fresh true CPU J7.
"""
from __future__ import annotations
import copy
from dataclasses import dataclass
import math

import torch
from calibri2.s125.j7_atomic_physical_retention import identical_tree

PROTECTED_OLD=456
PROTECTED_NEW=128
NEW_TRAIN=32

@dataclass(frozen=True)
class J7Retention584:
    """Supplied by independent *physical CPU* evaluator, not trainer/labels router.

    Digests must cover generated full token ID sequences in a stable
    predeclared prompt order, not only correctness totals.
    """
    actual_cpu_j7: bool
    old_checked: int
    old_full_eos_correct: int
    new_checked: int
    new_full_eos_correct: int
    old_first_lost: int
    old_full_lost: int
    new_first_lost: int
    new_full_lost: int
    protected_584_generation_digest: str
    train32_checked: int
    train32_first_correct: int
    train32_full_eos_correct: int
    train32_generation_digest: str
    serialized_fp16_reload_verified: bool
    frozen_parent_weights_unchanged: bool
    no_reference_data_at_inference: bool
    source_only_train_no_QA_TEST: bool
    same_CPU_speed_vs_S187: float
    physical_all_inclusive_bpp: float
    checkpoint_bytes_sha256: str

    def compatible(self) -> bool:
        hexchars=set("0123456789abcdef")
        digests=(self.protected_584_generation_digest,
                 self.train32_generation_digest,
                 self.checkpoint_bytes_sha256)
        sha_valid=all(isinstance(s,str) and len(s)==64
                      and set(s)<=hexchars for s in digests)
        numeric=(self.same_CPU_speed_vs_S187,
                 self.physical_all_inclusive_bpp)
        return (
            self.actual_cpu_j7 is True
            and self.old_checked==PROTECTED_OLD
            and self.old_full_eos_correct==PROTECTED_OLD
            and self.new_checked==PROTECTED_NEW
            and self.new_full_eos_correct==PROTECTED_NEW
            and self.old_first_lost==0 and self.old_full_lost==0
            and self.new_first_lost==0 and self.new_full_lost==0
            and self.train32_checked==NEW_TRAIN
            and 0<=self.train32_first_correct<=NEW_TRAIN
            and 0<=self.train32_full_eos_correct<=NEW_TRAIN
            and self.serialized_fp16_reload_verified is True
            and self.frozen_parent_weights_unchanged is True
            and self.no_reference_data_at_inference is True
            and self.source_only_train_no_QA_TEST is True
            and all(math.isfinite(x) for x in numeric)
            and self.same_CPU_speed_vs_S187>=.9
            and self.physical_all_inclusive_bpp<=.7
            and sha_valid
        )


@dataclass(frozen=True)
class OneBlockResult:
    one_step_executed: bool
    committed: bool
    rollback_verified: bool
    protected_count: int
    new_train_full_eos_gain: int
    baseline_bpp: float
    candidate_bpp: float
    speed_ratio: float
    model_accepted: bool=False


def _identical_params(parameters,snapshots):
    return all(torch.equal(p.detach(),v) for p,v in zip(parameters,snapshots))


def _same_recovered_predictions(parent,recovered):
    return (
        recovered.compatible()
        and recovered.protected_584_generation_digest==parent.protected_584_generation_digest
        and recovered.train32_generation_digest==parent.train32_generation_digest
        and recovered.checkpoint_bytes_sha256==parent.checkpoint_bytes_sha256
        and recovered.train32_first_correct==parent.train32_first_correct
        and recovered.train32_full_eos_correct==parent.train32_full_eos_correct
    )


def one_protected_step(
    params: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    train_one_step,
    independent_frozen_real_cpu_j7_guard,
    *,
    lr_reduction: float=.5,
) -> OneBlockResult:
    """Single bounded optimizer step, real external full CPU evidence or rollback.

    The callback MUST physically persist+reload FP16 and decode 456+128+32
    on the actual 43M model on CPU, freshly for parent/proposal/recovery.
    Any failure rejects the step and EXACTLY restores weights+all Adam state.
    Only successful proposal with strictly positive new32 full/EOS gain
    and no regression can commit. No model accepted or deployment implied.
    """
    if (not params or len({id(p) for p in params})!=len(params)
            or not 0<lr_reduction<1 or not math.isfinite(lr_reduction)):
        raise ValueError("S193 invalid protected transaction scope")
    optim_ids={id(p) for g in optimizer.param_groups for p in g["params"]}
    if not optimizer.param_groups or any(id(p) not in optim_ids for p in params):
        raise ValueError("S193 trainable params are not in optimizer")
    if any(not math.isfinite(float(g["lr"])) or g["lr"]<=0
           for g in optimizer.param_groups):
        raise ValueError("S193 invalid Adam learning rates")

    # Baseline physical CPU BEFORE changes. Refuse empty or inferred evidence.
    parent=independent_frozen_real_cpu_j7_guard()
    if not isinstance(parent,J7Retention584) or not parent.compatible():
        raise AssertionError("S193 baseline physical real CPU J7 584+32 not verified")
    prior=[p.detach().clone() for p in params]
    adam=copy.deepcopy(optimizer.state_dict())
    original_lrs=[float(g["lr"]) for g in optimizer.param_groups]
    changed=False
    candidate=None
    failure=None
    try:
        steps=train_one_step()
        changed=True
        if type(steps) is not int or steps!=1:
            raise AssertionError("S193 exactly ONE optimizer step is authorized")
        candidate=independent_frozen_real_cpu_j7_guard()
        if not isinstance(candidate,J7Retention584):
            raise TypeError("S193 physical CPU J7 guard must return typed proof")
        accepted=(candidate.compatible()
                  and candidate.protected_584_generation_digest==
                      parent.protected_584_generation_digest
                  and candidate.train32_first_correct>=parent.train32_first_correct
                  and candidate.train32_full_eos_correct>
                      parent.train32_full_eos_correct)
        if accepted:
            return OneBlockResult(True,True,False,584,
                    candidate.train32_full_eos_correct-parent.train32_full_eos_correct,
                    parent.physical_all_inclusive_bpp,
                    candidate.physical_all_inclusive_bpp,
                    candidate.same_CPU_speed_vs_S187)
    except Exception as exc:
        failure=exc
    # Non-positive gain, regression, or exception: EXACT restore of Adam
    # AND trainable weights (no other trainables authorized).
    with torch.no_grad():
        for p,v in zip(params,prior):
            p.copy_(v)
    optimizer.load_state_dict(adam)
    if not _identical_params(params,prior) or not identical_tree(optimizer.state_dict(),adam):
        raise AssertionError("S193 rollback failed exact weight/optimizer parity")
    recovered=independent_frozen_real_cpu_j7_guard()
    if not isinstance(recovered,J7Retention584) or not _same_recovered_predictions(parent,recovered):
        raise AssertionError("S193 rollback failed physically identical original CPU J7 584+32")
    if failure is not None:
        raise AssertionError("S193 one-step rejected and physically rolled back after error") from failure
    # Decrease LR only AFTER full physical rollback proof.
    for group,lr in zip(optimizer.param_groups,original_lrs):
        group["lr"]=lr*lr_reduction
    return OneBlockResult(changed,False,True,584,0,
        parent.physical_all_inclusive_bpp,
        candidate.physical_all_inclusive_bpp if candidate else parent.physical_all_inclusive_bpp,
        candidate.same_CPU_speed_vs_S187 if candidate else parent.same_CPU_speed_vs_S187)
