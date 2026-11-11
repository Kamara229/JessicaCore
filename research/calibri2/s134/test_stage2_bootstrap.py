"""S134: stage-2 inherits trained S132 weights and masks old gradient exactly."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from calibri2.s132.growth_schedule import architecture_params
from calibri2.s132.progressive_core import (
    ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY,initial_active_indices,
)
from calibri2.s129.additive_core_expansion import UP_KEY
from calibri2.s133.verify_stage2_entry import sha256_file
from calibri2.s134.stage2_bootstrap import (
    prepare_stage2,mask_gradient_to_new_channels_only,verify_frozen_inherited,
    STAGE2_TOTAL_CORE,
)


class TestS134Stage2Bootstrap(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        folder=Path(self.tmp.name)
        self.s131=folder/"S131_Physical.safetensors"
        self.s132=folder/"S132_Grown.safetensors"
        self.report=folder/"S132.json"
        ancestor=torch.zeros((12,320,24),dtype=torch.float16)
        ancestor[0,0,0]=0.03125
        grown=torch.zeros((12,320,48),dtype=torch.float16)
        grown[:,:,:24]=ancestor
        grown[0,0,0]=0.125  # learned additional update during five S132 cycles
        save_file({"s101.frozen":torch.tensor([0,1,2],dtype=torch.uint8),
                   UP_KEY:ancestor},str(self.s131))
        save_file({"s101.frozen":torch.tensor([0,1,2],dtype=torch.uint8),
                   IDS_KEY:torch.tensor(initial_active_indices(896),dtype=torch.int32),
                   WEIGHTS_KEY:grown},str(self.s132))
        self.payload={
            "stage":"S132_FIVE_CYCLES_THEN_ZERO_GROWTH",
            "next_rank":896,
            "previous_core_architecture":architecture_params(448),
            "next_core_architecture":architecture_params(896),
            "new_channels_trained":False,
            "new_architecture_initial_up_exact_zero":True,
            "old_456_exact_EOS_parity":True,
            "new_128_exact_EOS_parity":True,
            "old_456_full_correct":358,
            "new_128_full_correct":14,
            "not_promoted_to_Jessica_or_MAIN":True,
            "CPU_speed_ratio":.955,
            "rate_bpp_all_runtime_included":.676,
            "physical_sha256":sha256_file(self.s132),
            "five_J7_cycles_verified":[{
                "cycle":i,
                "old_456_checked":456,
                "lost_previously_correct_full":0,
                "lost_previously_correct_first":0,
                "physical_bpp":.676,
                "paired_cpu_ratio":.954,
                "physical_reloaded":True,
                "source_immutable":True,
                "old_full_count":358,
                "old_first_count":388,
                "new_full_count":14,
                "new_pairs_both_count":0,
                "transaction_committed":False,
                "transaction_rollback_verified":True,
            } for i in range(1,6)]
        }
        self.report.write_text(json.dumps(self.payload))

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        return prepare_stage2(s132_report=self.report,
                              s132_physical=self.s132,
                              s131_physical=self.s131)

    def test_real_inherited_43m_full_checkpoint(self):
        core,parent,immutable=self.load()
        self.assertEqual(parent.actual_rank,896)
        self.assertEqual(parent.new_parameter_count,6_881_280)
        self.assertEqual(STAGE2_TOTAL_CORE,43_261_760)
        self.assertEqual(float(core.up[0,0,0]),.125)
        self.assertTrue(torch.equal(immutable,core.up))
        verify_frozen_inherited(core,immutable)

    def test_stage2_gradient_mask_protects_s132_old_weights(self):
        core,parent,immutable=self.load()
        core.up.grad=torch.ones_like(core.up)
        counts=mask_gradient_to_new_channels_only(core)
        self.assertEqual(counts["trainable_new_channels"],24)
        self.assertEqual(counts["nonzero_new_gradient_coefficients"],92160)
        self.assertEqual(int(torch.count_nonzero(core.up.grad[:,:,:448])),0)
        opt=torch.optim.Adam([core.up],lr=.001)
        opt.step()
        verify_frozen_inherited(core,immutable)
        self.assertGreater(int(torch.count_nonzero(core.up[:,:,448:472])),0)

    def test_stage2_fails_if_original_learned_weights_change(self):
        core,parent,immutable=self.load()
        with torch.no_grad():
            core.up[0,0,0]+=0.5
        with self.assertRaises(AssertionError):
            verify_frozen_inherited(core,immutable)

    def test_stage2_rejects_unaudited_input(self):
        self.payload["five_J7_cycles_verified"]=[]
        self.report.write_text(json.dumps(self.payload))
        with self.assertRaises(AssertionError):self.load()

if __name__=="__main__":unittest.main()
