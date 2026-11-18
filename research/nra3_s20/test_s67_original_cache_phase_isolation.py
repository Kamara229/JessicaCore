"""S67 synthetic fail-closed diagnostics only; physical CI is separate."""
import unittest
from types import SimpleNamespace
import torch

from research.nra3_s20.s67_original_cache_phase_isolation import (
    MODES,cache_length,delta,diagnostic_result,
)


class TestS67OriginalCacheIsolation(unittest.TestCase):
    def test_modes_are_three_predeclared_nontraining_ablations(self):
        self.assertEqual(MODES,(
            "core_only_gt_disabled",
            "s93_original_prefill_only",
            "s93_bridge_all_steps_tunnel_prefill_only"))

    def test_dynamic_cache_position_reads_actual_KV(self):
        d=SimpleNamespace(get_seq_length=lambda:12)
        self.assertEqual(cache_length(d),12)
        self.assertEqual(cache_length(((torch.zeros(1,1,7,2),
                                        torch.zeros(1,1,7,2)),)),7)
        with self.assertRaisesRegex(ValueError,"missing KV"):
            cache_length(None)

    def test_real_differences_never_clamped_or_claimed_equal(self):
        a=torch.zeros((2,320),dtype=torch.float16)
        b=a.clone()
        b[1,17]=.0625
        self.assertEqual(delta(a,b)["max_abs"],.0625)
        self.assertFalse(delta(a,b)["exact_fp16_or_logits"])
        self.assertTrue(delta(a,a.clone())["exact_fp16_or_logits"])

    def test_nonfinite_and_shape_drift_fail_closed(self):
        x=torch.zeros((2,320),dtype=torch.float16)
        with self.assertRaisesRegex(ValueError,"differing tensor"):
            delta(x,x[:1])
        y=x.clone()
        y[0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            delta(x,y)

    def test_report_requires_complete_eight_rows_and_no_activation(self):
        def item(mode):
            return {"mode":mode,"output_gate":0,"candidate_accepted":False,
                    "activation_allowed":False,
                    "normal_online_generation_certified":False}
        good=[{m:item(m) for m in MODES} for _ in range(8)]
        report=diagnostic_result(good)
        self.assertEqual(len(report["rows"]),8)
        self.assertFalse(report["online_causality_not_certified"] is False)
        with self.assertRaisesRegex(ValueError,"complete"):
            diagnostic_result(good[:7])
        good[0][MODES[0]]["candidate_accepted"]=True
        with self.assertRaisesRegex(ValueError,"fail-closed"):
            diagnostic_result(good)

if __name__=="__main__":
    unittest.main()
