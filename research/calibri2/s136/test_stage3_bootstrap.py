"""S136 rank1344 new-channel-only preflight; synthetic CPU NOT a GPU result."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file,save_file

from calibri2.s132.progressive_core import (
    IDS_KEY,WEIGHTS_KEY,initial_active_indices,
)
from calibri2.s136.attest_stage3_entry import attest_stage3
from calibri2.s136.stage3_bootstrap import (
    prepare_stage3,mask_gradient_to_stage3_new_only,
    verify_frozen_stage1_stage2,
    STAGE3_NEW_CHANNELS,FROZEN_STAGE1_CHANNELS,FROZEN_STAGE2_CHANNELS,
    STAGE3_TRAINING_CYCLES,
)
from calibri2.s136.test_verify_stage3_entry import Stage3EntryTests


class Stage3BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.fixture=Stage3EntryTests(
            methodName="test_five_verified_cycles_and_real_reconstructed_geometry")
        self.fixture.setUp()
        self.fixture.check()

    def tearDown(self):
        self.fixture.tearDown()

    def load(self,device="cpu"):
        f=self.fixture
        return prepare_stage3(
            s135_report=f.report,s135_physical=f.physical,
            s132_physical=f.parent,device=device)

    def test_only_new_stage3_capacity_is_initially_zero(self):
        core,parent,immutable=self.load()
        self.assertEqual(parent.stage3_rank,1344)
        self.assertEqual(STAGE3_TRAINING_CYCLES,5)
        self.assertEqual(parent.old_full_correct,361)
        self.assertEqual(parent.new_full_correct,17)
        self.assertEqual(core.architectural_new_parameter_count(),10_321_920)
        self.assertEqual(FROZEN_STAGE1_CHANNELS,tuple(range(24)))
        self.assertEqual(FROZEN_STAGE2_CHANNELS,tuple(range(448,472)))
        self.assertEqual(STAGE3_NEW_CHANNELS,tuple(range(896,920)))
        self.assertEqual(tuple(core.active_indices),initial_active_indices(1344))
        self.assertEqual(
            int(torch.count_nonzero(core.up[:,:,list(STAGE3_NEW_CHANNELS)])),0)
        self.assertEqual(float(core.up[0,0,0]),.015625)
        self.assertAlmostEqual(float(core.up[0,0,448]),float(
            torch.tensor(.006,dtype=torch.float16)))
        self.assertTrue(torch.equal(core.up,immutable))
        verify_frozen_stage1_stage2(core,immutable)

    def test_fresh_gradient_masks_all_inherited_coefficients(self):
        core,parent,immutable=self.load()
        core.up.grad=torch.ones_like(core.up)
        report=mask_gradient_to_stage3_new_only(core)
        self.assertEqual(report["nonzero_fresh_gradient_coefficients"],12*320*24)
        self.assertEqual(int(torch.count_nonzero(core.up.grad[:,:,:896])),0)
        self.assertEqual(int(torch.count_nonzero(core.up.grad[:,:,920:])),0)
        optimizer=torch.optim.Adam([core.up],lr=.002)
        optimizer.step()
        with torch.no_grad():
            core.up[:,:,896:920].clamp_(-.05,.05)
        verify_frozen_stage1_stage2(core,immutable)
        self.assertNotEqual(float(core.up[0,0,896]),0.)

        # A real FP16 save/reload, not a claim that synthetic training
        # has passed the protected 456+128 exact-EOS CPU evaluation.
        physical=load_file(str(self.fixture.physical))
        old_sidecar=dict(physical)
        physical.update(core.export_active_fp16())
        fp=self.fixture.physical.parent/"S136_SYNTHETIC_FP16_SIDE_EFFECTS.safetensors"
        save_file(physical,str(fp))
        after=load_file(str(fp))
        for k in old_sidecar:
            if k not in {IDS_KEY,WEIGHTS_KEY}:
                self.assertTrue(torch.equal(after[k],old_sidecar[k]))
        self.assertTrue(torch.equal(
            after[WEIGHTS_KEY][:,:,:48],old_sidecar[WEIGHTS_KEY][:,:,:48]))
        self.assertTrue(torch.any(after[WEIGHTS_KEY][:,:,48:]!=0))
        self.assertEqual(tuple(after[IDS_KEY].tolist()),initial_active_indices(1344))
        self.assertFalse(parent.grown_source_archive==fp)

    def test_missing_or_nonfinite_new_gradients_are_rejected(self):
        core,_,_=self.load()
        with self.assertRaises(AssertionError):
            mask_gradient_to_stage3_new_only(core)
        core.up.grad=torch.zeros_like(core.up)
        with self.assertRaises(AssertionError):
            mask_gradient_to_stage3_new_only(core)
        core.up.grad[0,0,896]=float("nan")
        with self.assertRaises(AssertionError):
            mask_gradient_to_stage3_new_only(core)

    def test_altered_inherited_stage1_or_stage2_is_rejected(self):
        for col in (0,448):
            core,_,immutable=self.load()
            with torch.no_grad():
                core.up[0,0,col]+=.125
            with self.subTest(channel=col),self.assertRaises(AssertionError):
                verify_frozen_stage1_stage2(core,immutable)

    def test_no_stage3_entry_without_five_real_protected_cycles(self):
        candidate=copy.deepcopy(self.fixture.record)
        candidate["five_J7_cycles_verified"].pop()
        self.fixture.report.write_text(json.dumps(candidate))
        with self.assertRaises(AssertionError):
            self.load()

    def test_real_gate_cli_attests_verified_inputs_only(self):
        f=self.fixture
        proof_path=f.physical.parent/"S136_RESEARCH_ONLY_ATTESTATION.json"
        evidence=attest_stage3(
            report=f.report,physical=f.physical,parent=f.parent,
            output=proof_path)
        self.assertEqual(evidence["completed_s135_J7_cycles"],5)
        self.assertFalse(evidence["S136_stage3_GPU_training_started"])
        self.assertTrue(evidence["five_real_J7_required"])
        self.assertTrue(proof_path.exists())
        with self.assertRaises(AssertionError):
            attest_stage3(
                report=f.report,physical=f.physical,
                parent=proof_path.parent/"missing-S132.safetensors",
                output=proof_path)
        with self.assertRaises(AssertionError):
            attest_stage3(
                report=f.report,physical=f.physical,parent=f.parent,
                output=f.physical)

    def test_real_s136_stage3_trainer_compiles_and_freezes_all_old_coefficients(self):
        source=Path(__file__).with_name(
            "train_stage3_46m_five_cycles.py").read_text()
        compile(source,"train_stage3_46m_five_cycles.py","exec")
        for required in (
            "verified_s135_stage3_entry(",
            "a.S135_parent_sidecar",
            "a.S132_parent_sidecar",
            "mask_gradient_to_stage3_new_only(expansion)",
            "verify_frozen_stage1_stage2(expansion,immutable_before)",
            "expansion.up[:,:,896:920].clamp_",
            "ProgressiveRouteCore(rank=1344)",
            "authorize_growth(cycle_evidence,1344)",
            "S136_GROWN_50M_ZERO_NEW_CHANNELS.safetensors",
            "S136_NEXT_STAGE_50M_PHYSICAL_GROWTH_REPORT.json",
            "strict_train_retention(",
            "PhysicalRetentionCheck.valid(",
            "atomic_train_block(",
            "new_128_exact_EOS_parity",
            "old_456_exact_EOS_parity",
            "rate_bpp_all_runtime_included",
        ):
            with self.subTest(required=required):
                self.assertIn(required,source)
        for forbidden in (
            "ProgressiveRouteCore(rank=896)",
            "mask_gradient_to_new_channels_only(expansion)",
            "a.S131_parent_sidecar",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden,source)

    def test_stage3_gpu_workflow_is_fail_closed_until_s135_real_evidence(self):
        workflow=(Path(__file__).resolve().parents[3]/
            ".github"/"workflows"/
            "calibri2-s136-stage3-five-J7-cycles-46-to-50m-physical.yml"
        ).read_text()
        for required in (
            "s135-real-physical-approved",
            "require real", # verified below by exact real gate source
        ):
            if required=="require real":
                self.assertIn("verified_s135_stage3_entry(",workflow)
            else:
                self.assertIn(required,workflow)
        self.assertIn(' --jq .conclusion)" = "success"',workflow)
        self.assertIn("S135_GROWN_46M_ZERO_NEW_CHANNELS.safetensors",workflow)
        self.assertIn("S132_GROWN_43M_ZERO_NEW_CHANNELS.safetensors",workflow)
        self.assertIn("S136_GROWN_50M_ZERO_NEW_CHANNELS",Path(__file__).with_name(
            "train_stage3_46m_five_cycles.py").read_text())
        self.assertIn("research/calibri2/s136/train_stage3_46m_five_cycles.py",workflow)

    def test_mps_fp16_loader_has_no_unsupported_device_index_copy(self):
        source=(Path(__file__).resolve().parents[1]/
                "s132"/"progressive_core.py").read_text()
        loader=source.split("def load_active_fp16",1)[1].split(
            "def expand_preserving_previous",1)[0]
        self.assertNotIn("self.up.index_copy_",loader)
        self.assertIn("dense.index_copy_",loader)
        self.assertIn("self.up.copy_",loader)


if __name__=="__main__":
    unittest.main()
