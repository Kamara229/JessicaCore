"""S136 CPU-only fail-closed synthetic evidence tests (not QA / real run)."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from calibri2.s132.growth_schedule import architecture_params
from calibri2.s132.progressive_core import IDS_KEY,WEIGHTS_KEY,initial_active_indices
from calibri2.s136.verify_stage3_entry import (
    sha256_file,verified_s135_stage3_entry,TARGET_PARAM_COUNT,
)


class Stage3EntryTests(unittest.TestCase):
    def setUp(self):
        self._tmp=tempfile.TemporaryDirectory()
        d=Path(self._tmp.name)
        self.parent=d/"S132_REAL_43M.safetensors"
        self.physical=d/"S135_REAL_46M.safetensors"
        self.report=d/"S135_GROWTH_REPORT.json"

        old=torch.zeros((12,320,48),dtype=torch.float16)
        old[0,0,0]=0.015625
        old[0,0,24]=0.005
        newer=torch.zeros((12,320,72),dtype=torch.float16)
        newer[:,:,:48]=old
        side={"source.immutable":torch.tensor([17,29],dtype=torch.int32)}
        save_file({**side,IDS_KEY:torch.tensor(initial_active_indices(896),dtype=torch.int32),
                   WEIGHTS_KEY:old},str(self.parent))
        save_file({**side,IDS_KEY:torch.tensor(initial_active_indices(1344),dtype=torch.int32),
                   WEIGHTS_KEY:newer},str(self.physical))

        self.record={
            "stage":"S135_STAGE2_FIVE_CYCLES_THEN_ZERO_GROWTH",
            "previous_core_architecture":architecture_params(896),
            "next_core_architecture":architecture_params(1344),
            "next_rank":1344,
            "new_channels_trained":False,
            "new_architecture_initial_up_exact_zero":True,
            "old_456_exact_EOS_parity":True,
            "new_128_exact_EOS_parity":True,
            "old_456_full_correct":361,
            "new_128_full_correct":17,
            "additional_structural_core_params":
                architecture_params(1344)-architecture_params(896),
            "newly_learned_params_from_structural_growth":0,
            "physically_persisted_active_coefficients":12*320*72,
            "inherited_trainable_channel_count":24,
            "stage2_trainable_new_channel_count":24,
            "not_promoted_to_Jessica_or_MAIN":True,
            "physical_sha256":sha256_file(self.physical),
            "rate_bpp_all_runtime_included":0.68,
            "CPU_speed_ratio":0.95,
            "five_J7_cycles_verified":[{
                "cycle":i,"old_456_checked":456,
                "lost_previously_correct_full":0,
                "lost_previously_correct_first":0,
                "physical_bpp":0.69,"paired_cpu_ratio":0.96,
                "physical_reloaded":True,"source_immutable":True,
                "old_full_count":361,"old_first_count":389,
                "new_full_count":17,"new_pairs_both_count":0,
                "transaction_committed":i==1,
                "transaction_rollback_verified":i!=1,
            } for i in range(1,6)],
        }

    def tearDown(self):
        self._tmp.cleanup()

    def check(self,record=None):
        self.report.write_text(json.dumps(self.record if record is None else record))
        return verified_s135_stage3_entry(self.report,self.physical,self.parent)

    def test_five_verified_cycles_and_real_reconstructed_geometry(self):
        result=self.check()
        self.assertEqual(result["stage3_architecture_params"],TARGET_PARAM_COUNT)
        self.assertEqual(result["completed_s135_J7_cycles"],5)
        self.assertEqual(result["stage3_rank"],1344)
        self.assertTrue(result["stage3_new_24_coefficients_exact_zero"])

    def test_reject_forgotten_old_full_and_new_or_missing_cycle(self):
        for key,bad in [("old_full_count",360),("old_first_count",388),
                        ("new_full_count",16),("lost_previously_correct_full",1),
                        ("lost_previously_correct_first",1)]:
            candidate=copy.deepcopy(self.record)
            candidate["five_J7_cycles_verified"][2][key]=bad
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(candidate)
        candidate=copy.deepcopy(self.record)
        candidate["five_J7_cycles_verified"].pop()
        with self.assertRaises(AssertionError):
            self.check(candidate)

    def test_reject_bpp_cpu_or_modified_archive(self):
        for key,bad in [("rate_bpp_all_runtime_included",.71),
                        ("CPU_speed_ratio",.899),
                        ("physical_sha256","0"*64),
                        ("newly_learned_params_from_structural_growth",1)]:
            candidate={**self.record,key:bad}
            with self.subTest(key=key),self.assertRaises(AssertionError):
                self.check(candidate)

    def test_reject_altered_inherited_learned_stage1_channels(self):
        frozen=load_file(str(self.physical))
        w=frozen[WEIGHTS_KEY].clone()
        w[0,0,0]=0.33
        frozen[WEIGHTS_KEY]=w
        save_file(frozen,str(self.physical))
        self.record["physical_sha256"]=sha256_file(self.physical)
        with self.assertRaises(AssertionError):
            self.check()

    def test_reject_altered_inherited_stage2_channels_24_to_47(self):
        # Regression: stage2 trained channels were previously not checked.
        frozen=load_file(str(self.physical))
        w=frozen[WEIGHTS_KEY].clone()
        w[0,0,24]=0.006
        frozen[WEIGHTS_KEY]=w
        save_file(frozen,str(self.physical))
        self.record["physical_sha256"]=sha256_file(self.physical)
        with self.assertRaises(AssertionError):
            self.check()

    def test_reject_nonzero_untrained_stage3_columns(self):
        frozen=load_file(str(self.physical))
        w=frozen[WEIGHTS_KEY].clone()
        w[0,0,48]=0.125
        frozen[WEIGHTS_KEY]=w
        save_file(frozen,str(self.physical))
        self.record["physical_sha256"]=sha256_file(self.physical)
        with self.assertRaises(AssertionError):
            self.check()

    def test_reject_modified_s101_q2_immutable_bytes(self):
        frozen=load_file(str(self.physical))
        frozen["source.immutable"]=torch.tensor([17,30],dtype=torch.int32)
        save_file(frozen,str(self.physical))
        self.record["physical_sha256"]=sha256_file(self.physical)
        with self.assertRaises(AssertionError):
            self.check()


if __name__=="__main__":
    unittest.main()
