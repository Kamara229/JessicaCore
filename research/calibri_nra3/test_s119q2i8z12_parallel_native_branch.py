"""Z12 locked fresh-DEV and target-blind native CPU lookahead unit tests."""
import unittest
from collections import Counter
import torch

from research.calibri_nra3.s119q2i8z12_parallel_native_new14_source import (
    rows,verify,digest,KINDS,
)
from research.calibri_nra3.s119q2i8z12_native_parallel_branch_contract import (
    native_lookahead_second_third,NATIVE_VOCAB,WIDTH,THIRD_WIDTH,
)

class NativeOracle:
    def __init__(self,second_logits,third_by_second):
        self.second=second_logits
        self.third=third_by_second
        self.calls=[]
    def __call__(self,prefix):
        self.calls.append(prefix)
        if len(prefix)==4:
            return self.second.clone()
        return self.third.get(prefix[-1],torch.zeros((1,NATIVE_VOCAB))).clone()

class TestZ12(unittest.TestCase):
    def test_source_sha_and_14_new_unscored_seven_domains(self):
        v=verify()
        self.assertEqual(v["fresh_DEV_rows"],14)
        self.assertEqual(v["DEV14_SHA256"],digest(rows()))
        self.assertEqual(Counter(x["kind"] for x in rows()),{k:2 for k in KINDS})
        self.assertTrue(v["disjoint_prior_E_G_K_R_T_W_X_Z2_Z9"])
        self.assertEqual((v["BRANCH_WIDTH"],v["THIRD_WIDTH"],v["HORIZON"]),(8,2,2))
        for bad in ("TRAIN","J7","GOLD","EVAL128","TEST"):
            with self.assertRaises(ValueError):rows(bad)

    def test_lookahead_can_override_greedy_via_actual_future_logits(self):
        lp2=torch.full((1,NATIVE_VOCAB),-40.)
        lp2[0,10]=6.0
        lp2[0,11]=5.9
        # Fix all remaining top-8 ranks explicitly: torch.topk tie ordering
        # is backend-dependent, and EOS must be outside this non-EOS fixture.
        lp2[0,20:26]=torch.tensor([-30.,-31.,-32.,-33.,-34.,-35.])
        # For greedy BPE2=10, future logits are flat => low third confidence.
        # For alternative 11, one third token is highly probable.
        lp3=torch.zeros((1,NATIVE_VOCAB))
        lp3alt=torch.full((1,NATIVE_VOCAB),-15.)
        lp3alt[0,31]=15.
        f=NativeOracle(lp2,{11:lp3alt,10:lp3})
        d=native_lookahead_second_third(f,(100,101,102,103))
        self.assertEqual(d["own_greedy_continuation_2native_BPE"][0],10)
        self.assertEqual(d["lookahead_selected_continuation_2native_BPE"],[11,31])
        self.assertEqual(d["actual_candidate_count"],16)
        self.assertEqual(d["actual_original43M_native_forward_calls"],9)
        self.assertTrue(d["no_gold_used_in_search"])
        self.assertTrue(d["candidate_generation_is_sequential_CPU_not_parallel_wallclock"])

    def test_EOS_included_never_banned_and_shorter_candidate_possible(self):
        lp2=torch.full((1,NATIVE_VOCAB),-40.)
        lp2[0,2]=10.
        lp2[0,17]=9.
        lp2[0,20:26]=torch.tensor([-30.,-31.,-32.,-33.,-34.,-35.])
        f=NativeOracle(lp2,{})
        d=native_lookahead_second_third(f,(100,101,102,103))
        self.assertEqual(d["top8_second_native_ids"][0],2)
        self.assertEqual(d["own_greedy_continuation_2native_BPE"],[2])
        self.assertEqual(d["lookahead_selected_continuation_2native_BPE"],[2])
        self.assertEqual(d["actual_candidate_count"],15)
        self.assertEqual(d["actual_original43M_native_forward_calls"],8)

    def test_deny_non_native_wrongshape_nonfinite_and_grad(self):
        wrong=torch.zeros((1,NATIVE_VOCAB))
        bad=NativeOracle(wrong,{})
        with self.assertRaises(ValueError):
            native_lookahead_second_third(bad,[100,101,102,49152])
        with self.assertRaises(ValueError):
            native_lookahead_second_third(lambda p:torch.ones((1,123)),(1,2,3,4))
        with self.assertRaises(ValueError):
            native_lookahead_second_third(lambda p:torch.full((1,NATIVE_VOCAB),float("nan")),(1,2,3,4))
        with self.assertRaises(ValueError):
            native_lookahead_second_third(lambda p:torch.zeros((1,NATIVE_VOCAB),requires_grad=True),(1,2,3,4))
if __name__=="__main__":
    unittest.main()
