"""S127 SYNTHETIC mechanism tests ONLY: NOT a real CALIBRI training/456 audit.

Original S101 TRAIN and independent QA are never accessed here.
Full prefill-only Q2 gate trainability, physical reload and J7 rollback
are local dummy-model tests. The real physical 456+128 guard is mandatory
for any subsequent actual S127 candidate, never inferred from these.
"""
from __future__ import annotations
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from safetensors.torch import save_file,load_file
from calibri2.s126.full_layer_state_tunnel import FullLayerStateTunnel
from calibri2.s126.compact_rank4_q2 import export_compact_q2,load_compact_q2,PREFIX
from calibri2.s126.test_full_layer_state_tunnel import TwelveLayerModel
from calibri2.s125.j7_atomic_physical_retention import (
    PhysicalRetentionCheck,atomic_train_block,
)
from calibri2.s127.q2_prefill_gate_control import (
    prefill_only_tunnel,restrict_to_gate_only,
)


class SyntheticCachedModel(TwelveLayerModel):
    def forward(self,x,*,past_key_values=None,use_cache=True):
        return super().forward(x)


class TestS127Q2Preflight(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1270111)
        self.model=SyntheticCachedModel().eval()
        self.tunnel=FullLayerStateTunnel(width=320,depth=12,rank=4).eval()
        archive=export_compact_q2(self.tunnel)
        load_compact_q2(self.tunnel,archive)
        self.tunnel.install(self.model)
        self.x=torch.randn(1,3,320)
    def tearDown(self):
        self.tunnel.uninstall()

    def test_prefill_only_prevents_cached_token_leakage(self):
        baseline=self.model(self.x,past_key_values=None)
        with torch.no_grad():
            self.tunnel.gates.fill_(0.22)
        with prefill_only_tunnel(self.model,self.tunnel) as flags:
            prefill=self.model(self.x,past_key_values=None)
            decoded=self.model(self.x,past_key_values=object())
            following=self.model(self.x,past_key_values=None)
            self.assertEqual(flags,[True,False,True])
            self.assertTrue(torch.equal(prefill,following))
        self.assertFalse(torch.equal(baseline,prefill))
        self.assertTrue(torch.equal(baseline,decoded))
        self.assertTrue(torch.equal(baseline,self.model(self.x,past_key_values=None)))
        self.assertFalse(self.tunnel._active)
        self.assertIsNone(self.tunnel._carry)

    def test_reject_ambiguous_cache_call(self):
        with prefill_only_tunnel(self.model,self.tunnel):
            with self.assertRaises(AssertionError):
                self.model(self.x)
        self.assertFalse(self.tunnel._active)

    def test_only_paid_q2_gates_receive_gradient_and_archive_roundtrip(self):
        params=restrict_to_gate_only(self.tunnel)
        self.assertEqual([p.numel() for p in params],[12])
        with prefill_only_tunnel(self.model,self.tunnel):
            loss=self.model(self.x,past_key_values=None).square().mean()
        loss.backward()
        self.assertGreater(int(torch.count_nonzero(self.tunnel.gates.grad)),0)
        self.assertIsNone(self.tunnel.encoder.grad)
        self.assertIsNone(self.tunnel.decoder.grad)
        self.assertIsNone(self.tunnel.mix_logits.grad)
        before=export_compact_q2(self.tunnel)
        optimizer=torch.optim.Adam(params,lr=0.001)
        optimizer.step()
        after=export_compact_q2(self.tunnel)
        self.assertTrue(torch.equal(before[PREFIX+"encoder_codes"],
                                    after[PREFIX+"encoder_codes"]))
        self.assertTrue(torch.equal(before[PREFIX+"decoder_codes"],
                                    after[PREFIX+"decoder_codes"]))
        with TemporaryDirectory() as folder:
            path=Path(folder)/"candidate.safetensors"
            save_file(after,str(path))
            reloaded=load_file(str(path))
            clone=FullLayerStateTunnel(width=320,depth=12,rank=4)
            load_compact_q2(clone,reloaded)
            self.assertTrue(all(torch.equal(reloaded[k],v)
                              for k,v in export_compact_q2(clone).items()))

    def test_j7_rollback_restores_gate_and_adam_with_reloaded_archive(self):
        params=restrict_to_gate_only(self.tunnel)
        optimizer=torch.optim.Adam(params,lr=0.02)
        frozen=self.tunnel.gates.detach().clone()
        orig=self.model(self.x,past_key_values=None)
        with TemporaryDirectory() as folder:
            path=Path(folder)/"physical-test.safetensors"
            def step():
                optimizer.zero_grad(set_to_none=True)
                with prefill_only_tunnel(self.model,self.tunnel):
                    value=self.model(self.x,past_key_values=None).square().mean()
                value.backward()
                self.assertGreater(int(torch.count_nonzero(self.tunnel.gates.grad)),0)
                optimizer.step()
                return 1
            def mock_real_guard():
                # The numbers below are MOCKED ABI inputs used ONLY to test
                # the preexisting J7 rollback implementation. They are NOT
                # observational evidence for old/new 456+128 answers.
                save_file(export_compact_q2(self.tunnel),str(path))
                restored=load_file(str(path))
                t2=FullLayerStateTunnel(width=320,depth=12,rank=4)
                load_compact_q2(t2,restored)
                identical=bool(torch.equal(t2.gates,frozen))
                no_change=bool(torch.equal(orig,self.model(
                    self.x,past_key_values=None)))
                return PhysicalRetentionCheck(
                    original_456_checked=456,original_full_correct=353,
                    original_first_correct=387,
                    lost_original_full=0 if identical and no_change else 1,
                    lost_original_first=0,
                    newly_correct_original_full=0,
                    new_S122_train_full=14,
                    new_S122_train_both_correct_pairs=0,
                    physical_all_inclusive_bpp=0.6998,
                    paired_original_S101_CPU_speed_ratio=0.98,
                    physical_FP16_reload_verified=True,
                    immutable_original_S101_bytes_verified=True,
                    S95_S98_unchanged=True)
            result=atomic_train_block(params,optimizer,step,
                                      mock_real_guard,reduction_factor=0.5)
            self.assertFalse(result.committed)
            self.assertTrue(result.rollback_verified)
            self.assertTrue(torch.equal(self.tunnel.gates,frozen))
            self.assertEqual(optimizer.param_groups[0]["lr"],0.01)


if __name__=="__main__":
    unittest.main()
