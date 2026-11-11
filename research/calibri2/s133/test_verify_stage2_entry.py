"""S133 require genuine S132 five-cycle evidence before stage-2 GPU training."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from calibri2.s129.additive_core_expansion import UP_KEY
from calibri2.s132.growth_schedule import architecture_params
from calibri2.s132.progressive_core import IDS_KEY,WEIGHTS_KEY,initial_active_indices
from calibri2.s133.verify_stage2_entry import (
    sha256_file,verified_s132_stage2_entry,
)

class Stage2EntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        p=Path(self.tmp.name)
        self.parent=p/"S131_FP16.safetensors"
        self.grown=p/"S132_43M.safetensors"
        self.report=p/"S132_43M.json"
        learned=torch.zeros((12,320,24),dtype=torch.float16)
        learned[0,0,0]=0.03125
        save_file({"original.frozen":torch.ones(4,dtype=torch.uint8),
                   UP_KEY:learned},str(self.parent))
        weights=torch.zeros((12,320,48),dtype=torch.float16)
        weights[:,:,:24]=learned
        ids=torch.tensor(initial_active_indices(896),dtype=torch.int32)
        save_file({"original.frozen":torch.ones(4,dtype=torch.uint8),
                   IDS_KEY:ids,WEIGHTS_KEY:weights},str(self.grown))
        self.expected={
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
            "CPU_speed_ratio":.97,
            "rate_bpp_all_runtime_included":.68,
            "physical_sha256":sha256_file(self.grown),
            "five_J7_cycles_verified": [{
                "cycle":i,"old_456_checked":456,
                "lost_previously_correct_full":0,
                "lost_previously_correct_first":0,
                "physical_bpp":.68,"paired_cpu_ratio":.97,
                "physical_reloaded":True,"source_immutable":True,
                "old_full_count":358,"old_first_count":388,
                "new_full_count":14,"new_pairs_both_count":0,
                "transaction_committed":True,
                "transaction_rollback_verified":False,
            } for i in range(1,6)],
        }
    def tearDown(self):
        self.tmp.cleanup()

    def check(self,report=None):
        self.report.write_text(json.dumps(self.expected if report is None else report))
        return verified_s132_stage2_entry(self.report,self.grown,self.parent)

    def test_verifies_all_five_cycles_and_physical_reconstruction(self):
        manifest=self.check()
        self.assertEqual(manifest["stage2_architecture_params"],43_261_760)
        self.assertEqual(manifest["completed_s132_j7_cycles"],5)
        self.assertTrue(manifest["original_and_s131_weights_verified_immutable"])

    def test_fails_on_missing_one_cycle(self):
        self.expected["five_J7_cycles_verified"].pop()
        with self.assertRaises(AssertionError):self.check()

    def test_fails_on_old_loss_or_bpp_or_cpu(self):
        self.expected["five_J7_cycles_verified"][1]["lost_previously_correct_full"]=1
        with self.assertRaises(AssertionError):self.check()
        self.expected["five_J7_cycles_verified"][1]["lost_previously_correct_full"]=0
        self.expected["rate_bpp_all_runtime_included"]=.701
        with self.assertRaises(AssertionError):self.check()
        self.expected["rate_bpp_all_runtime_included"]=.68
        self.expected["CPU_speed_ratio"]=.89
        with self.assertRaises(AssertionError):self.check()

    def test_fails_on_forged_or_changed_physical_source(self):
        self.expected["physical_sha256"]="0"*64
        with self.assertRaises(AssertionError):self.check()
        self.expected["physical_sha256"]=sha256_file(self.grown)
        changed=torch.zeros((12,320,48),dtype=torch.float16)
        changed[0,0,24]=.1
        save_file({"original.frozen":torch.ones(4,dtype=torch.uint8),
                   IDS_KEY:torch.tensor(initial_active_indices(896),dtype=torch.int32),
                   WEIGHTS_KEY:changed},str(self.grown))
        self.expected["physical_sha256"]=sha256_file(self.grown)
        with self.assertRaises(AssertionError):self.check()

if __name__=="__main__":
    unittest.main()
