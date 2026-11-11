"""S135 isolated stage-2 five-cycle training contract preflight tests."""
from __future__ import annotations
import inspect
import unittest
import torch

from calibri2.s132.progressive_core import ProgressiveRouteCore
from calibri2.s132.growth_schedule import (
    ActualJ7Cycle, authorize_growth,architecture_params,
)
from calibri2.s134.stage2_bootstrap import (
    mask_gradient_to_new_channels_only,verify_frozen_inherited,
    STAGE2_TOTAL_CORE,STAGE2_TRAINING_CYCLES,
    NEW_CHANNELS,
)


class TestS135FiveCycle43M(unittest.TestCase):
    def test_physical_rank_and_stage2_cycle_lengths(self):
        self.assertEqual(STAGE2_TOTAL_CORE,43_261_760)
        self.assertEqual(STAGE2_TRAINING_CYCLES,5)
        self.assertEqual(architecture_params(1344),46_702_400)
        self.assertEqual(NEW_CHANNELS,tuple(range(448,472)))

    def test_previous_weights_cannot_be_modified_by_training_step(self):
        core=ProgressiveRouteCore(rank=896)
        with torch.no_grad():
            core.up[0,0,0]=.015625
        original=core.up.detach().clone()
        core.up.grad=torch.ones_like(core.up)
        mask_gradient_to_new_channels_only(core)
        opt=torch.optim.Adam([core.up],lr=.002)
        opt.step()
        with torch.no_grad():
            core.up[:,:,448:472].clamp_(-0.05,0.05)
        verify_frozen_inherited(core,original)
        self.assertNotEqual(float(core.up[0,0,448]),0.0)
        self.assertEqual(float(core.up[0,0,0]),.015625)

    def test_five_audited_cycles_only_then_46m(self):
        records=[
            ActualJ7Cycle(
                cycle=i,old_456_checked=456,
                lost_previously_correct_full=0,
                lost_previously_correct_first=0,
                physical_bpp=.68,paired_cpu_ratio=.98,
                physical_reloaded=True,source_immutable=True,
                old_full_count=358,old_first_count=388,
                new_full_count=15,new_pairs_both_count=0,
                transaction_committed=True,
                transaction_rollback_verified=False,
            ) for i in range(1,6)
        ]
        self.assertEqual(authorize_growth(records,896),1344)
        with self.assertRaises(AssertionError):
            authorize_growth(records[:4],896)

    def test_train_source_masks_new_channels_only(self):
        # Structural inspection ensures production cannot silently remove
        # the new-only mask or switch back to training inherited channels.
        from calibri2.s135 import train_stage2_43m_five_cycles as program
        source=inspect.getsource(program.main)
        self.assertIn("mask_gradient_to_new_channels_only(expansion)",source)
        self.assertIn("verify_frozen_inherited(expansion,immutable_before)",source)
        self.assertIn("expansion.up[:,:,448:472].clamp_",source)
        self.assertIn("verified_s132_stage2_entry(",source)

if __name__=="__main__":
    unittest.main()
