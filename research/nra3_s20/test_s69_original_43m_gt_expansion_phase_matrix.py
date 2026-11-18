"""S69 fail-closed original model forward hook phase ABI tests."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from research.nra3_s20.s69_original_43m_gt_expansion_phase_matrix import (
    MODES,final_audit,frozen_original_expansion_all_steps,
)


class Dummy(nn.Module):
    def __init__(self,exp):
        super().__init__()
        self.exp=exp
        self.seen=[]
        self.eval()
    def forward(self,*,input_ids,past_key_values,use_cache):
        self.seen.append((bool(self.exp._active),past_key_values is None))
        return input_ids


class TestS69PhaseMatrix(unittest.TestCase):
    def test_expansion_installed_only_temporal_active_restored(self):
        expansion=SimpleNamespace(_active=False,_handles=[object()],_visited=[])
        model=Dummy(expansion)
        with torch.inference_mode():
            with frozen_original_expansion_all_steps(model,expansion) as visits:
                first=model(input_ids=torch.ones(1,4,dtype=torch.long),
                            past_key_values=None,use_cache=True)
                second=model(input_ids=torch.ones(1,1,dtype=torch.long),
                             past_key_values=object(),use_cache=True)
        self.assertEqual(first.shape,(1,4))
        self.assertEqual(second.shape,(1,1))
        self.assertEqual(model.seen,[(True,True),(True,False)])
        self.assertEqual(visits,[True,False])
        self.assertFalse(expansion._active)

    def test_expansion_all_steps_requires_explicit_cache(self):
        exp=SimpleNamespace(_active=False,_handles=[object()],_visited=[])
        model=Dummy(exp)
        with torch.inference_mode():
            with self.assertRaisesRegex(ValueError,"explicit KV"):
                with frozen_original_expansion_all_steps(model,exp):
                    model(input_ids=torch.ones(1,1),past_key_values=None,use_cache=False)
        self.assertFalse(exp._active)

    def test_modes_factor_tunnel_and_expansion_not_model_training(self):
        self.assertEqual(len(MODES),3)
        self.assertIn("tunnel_all_expansion_all",MODES[0])
        self.assertIn("tunnel_all_expansion_prefill",MODES[1])
        self.assertIn("tunnel_prefill_expansion_all",MODES[2])

    def test_reports_prohibit_gate_or_online_claims(self):
        def record(mode):
            return {"mode":mode,"cut":7,"output_gate":0,"candidate_accepted":False,
                    "activation_allowed":False,"normal_online_generation_certified":False,
                    "full_vs_KV2_H8_next2":{"max_abs":0.01},
                    "full_vs_KV2_logits_next2":{"max_abs":0.04},
                    "KV2_vs_KV1plus1_logits_next2":{"max_abs":0.0}}
        rows=[{m:record(m) for m in MODES} for _ in range(8)]
        report=final_audit(rows)
        self.assertEqual(report["n_rows"],8)
        self.assertFalse(report["online_causality_certified"])
        self.assertEqual(report["mode_h8_KV2_mismatch_gt_1e3"][MODES[0]],8)
        self.assertEqual(report["mode_logits_KV2_vs_KV1plus1_mismatch_gt_1e3"][MODES[0]],0)
        rows[0][MODES[0]]["activation_allowed"]=True
        with self.assertRaisesRegex(ValueError,"research only"):
            final_audit(rows)

if __name__=="__main__":
    unittest.main()
