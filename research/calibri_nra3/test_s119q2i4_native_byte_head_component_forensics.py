"""S119q2i4 fail-closed causal native byte-ID30/155 head attribution tests."""
from __future__ import annotations

import copy
from unittest import TestCase,main

import torch
from torch import nn

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    NativeHeadAlignment,temporary_learned_native_head,
)
from research.calibri_nra3.s119q2i4_native_byte_head_component_forensics import (
    STAGE,MASKS,VARIANTS,N_PHYSICAL_DECODERS,N_SOURCE,TOP_K,
    COLLAPSE_IDS,HEAD_GAIN,ADDITIVE_PARITY_TOL,
    temporary_head_component_mask,additive_readout_identity,
    inspect_native,summarize_masks,
)


class OriginalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head=nn.Linear(320,49152,bias=False)
        self.lm_head.requires_grad_(False)
    def forward(self,x):
        return self.lm_head(x)


class FakeNative:
    def decode(self,ids,skip_special_tokens=False):
        return "".join(chr(65+(v%26)) for v in ids)
    def convert_ids_to_tokens(self,id):
        return "<0x"+format(id,"02X")+">"


def synthetic_native(v,*,baseline=0.0):
    z=torch.full((1,49152),baseline,dtype=torch.float32)
    z[0,v]=7.
    return z


def make_observations():
    result=[]
    for i in range(N_SOURCE):
        masks={}
        for m in MASKS:
            masks[m]={"metrics":{
                "native_top1_id":155 if m=="FULL" else 30,
                "correct_first_native_rank_49152":500+i,
                "correct_first_native_in_top12":False,
                "top1_minus_correct_first_logit_margin":4.,
                "collapsed_30_raw_native_logit":1.,
                "collapsed_155_raw_native_logit":2.,
            }}
        result.append({"source_id":"old_"+str(i),"masks":masks})
    return result


class S119Q2I4NativeByteReadoutCausalContracts(TestCase):
    def test_01_head_component_four_controlled_masks(self):
        self.assertEqual(MASKS,(
            "FULL","NO_BIAS","NO_LOWRANK","HEAD_OFF"))
        self.assertEqual(VARIANTS,(
            "GT_ZERO_BASELINE","GT_RECURRENT","GT_NOCARRY_LOCAL"))
        self.assertEqual(N_SOURCE,12)
        self.assertEqual(N_PHYSICAL_DECODERS,144)

    def test_02_known_collapsed_partial_byte_native_155_and_dot_30(self):
        self.assertEqual(COLLAPSE_IDS,(30,155))
        self.assertEqual(TOP_K,12)
        self.assertEqual(HEAD_GAIN,.25)

    def test_03_same_true_original_frozen_LM_head_counterfactual_parity(self):
        torch.manual_seed(33)
        original=OriginalHead()
        head=NativeHeadAlignment()
        with torch.no_grad():
            head.output_bias[30]=3.0
            head.output_bias[155]=1.0
            head.down.weight.fill_(.01)
            head.up.weight[30].fill_(.5)
            head.up.weight[155].fill_(1.)
        x=torch.randn(1,3,320)
        actual={}
        old=original(x).detach().clone()
        for mode in MASKS:
            with temporary_head_component_mask(head,mode) as witness:
                with temporary_learned_native_head(original,head) as audit:
                    actual[mode]=original(x).detach().float()[:,-1]
            self.assertEqual(witness["calls"],1)
            self.assertEqual(audit["original_head_calls"],1)
        p=additive_readout_identity(
            actual["FULL"],actual["NO_BIAS"],
            actual["NO_LOWRANK"],actual["HEAD_OFF"])
        self.assertLess(p["native_49k_additive_physical_parity_Linf"],
                        ADDITIVE_PARITY_TOL)
        self.assertTrue(torch.equal(old,original(x)))
        self.assertTrue(torch.equal(
            actual["HEAD_OFF"],old[:,-1]))
        self.assertEqual(len(head._forward_hooks),0)
        self.assertEqual(len(original.lm_head._forward_hooks),0)

    def test_04_mask_scopes_are_readonly_state_dict(self):
        head=NativeHeadAlignment()
        with torch.no_grad():
            head.output_bias[155]=7.
        before={k:v.detach().clone() for k,v in head.state_dict().items()}
        for m in MASKS:
            with temporary_head_component_mask(head,m):
                head(torch.randn(1,2,320))
        self.assertTrue(all(torch.equal(v,head.state_dict()[k])
                            for k,v in before.items()))

    def test_05_bias_only_vs_lowrank_only_mathematically_distinct(self):
        head=NativeHeadAlignment()
        with torch.no_grad():
            head.output_bias[155]=8.
        x=torch.randn(1,2,320)
        values={}
        for mode in MASKS:
            with temporary_head_component_mask(head,mode):
                values[mode]=head(x).detach().clone()
        self.assertEqual(values["NO_BIAS"][0,-1,155].item(),0.)
        self.assertAlmostEqual(
            values["NO_LOWRANK"][0,-1,155].item(),2.,places=5)
        self.assertAlmostEqual(
            values["FULL"][0,-1,155].item(),2.,places=5)
        self.assertTrue(torch.equal(values["HEAD_OFF"],
                                    torch.zeros_like(values["HEAD_OFF"])))

    def test_06_rank_off_does_not_mutate_original_full_vocab(self):
        head=NativeHeadAlignment()
        original=OriginalHead()
        x=torch.randn(1,3,320)
        before=original(x).detach().clone()
        with temporary_head_component_mask(head,"NO_LOWRANK"):
            with temporary_learned_native_head(original,head):
                original(x)
        self.assertTrue(torch.equal(original(x).detach(),before))

    def test_07_restore_hook_even_when_exception(self):
        head=NativeHeadAlignment()
        with self.assertRaisesRegex(RuntimeError,"intentional"):
            with temporary_head_component_mask(head,"NO_BIAS"):
                raise RuntimeError("intentional")
        self.assertFalse(head._forward_hooks)

    def test_08_invalid_mode_fails_closed(self):
        head=NativeHeadAlignment()
        with self.assertRaisesRegex(GateFailure,"fixed 0.25"):
            with temporary_head_component_mask(head,"SUPPRESS_NON_RUSSIAN"):
                pass

    def test_09_concurrent_head_ablation_nesting_denied(self):
        head=NativeHeadAlignment()
        with temporary_head_component_mask(head,"FULL"):
            with self.assertRaisesRegex(GateFailure,"fixed 0.25"):
                with temporary_head_component_mask(head,"NO_BIAS"):
                    pass

    def test_10_model_accepted_is_denied(self):
        head=NativeHeadAlignment()
        head.candidate_accepted=True
        with self.assertRaisesRegex(GateFailure,"fixed 0.25"):
            with temporary_head_component_mask(head,"FULL"):
                pass

    def test_11_exact_full_native_49k_head_logits_required(self):
        a=synthetic_native(155)
        result=inspect_native(FakeNative(),a,201)
        self.assertEqual(result["native_top1_id"],155)
        self.assertEqual(result["correct_first_native_rank_49152"],2)
        self.assertFalse(result["correct_first_native_in_top12"])
        self.assertEqual(result["native_top1_tokenizer_raw"],"<0x9B>")

    def test_12_native_gold_out_of_vocab_or_NaN_fails_closed(self):
        with self.assertRaisesRegex(GateFailure,"49k"):
            inspect_native(FakeNative(),torch.zeros(1,1234),1)
        with self.assertRaisesRegex(GateFailure,"49k"):
            inspect_native(FakeNative(),torch.zeros(1,49152),49152)
        x=synthetic_native(155);x[0,0]=float("nan")
        with self.assertRaisesRegex(GateFailure,"49k"):
            inspect_native(FakeNative(),x,201)

    def test_13_additive_counterfactual_parity_detects_wrong_logits(self):
        x=synthetic_native(155)
        y=x.clone(); y[0,30]+=1.
        with self.assertRaisesRegex(GateFailure,"decomposition"):
            additive_readout_identity(y,x,x,x)

    def test_14_additive_parity_fails_on_proxy_vocab(self):
        x=torch.zeros((1,32))
        with self.assertRaisesRegex(GateFailure,"same original full native"):
            additive_readout_identity(x,x,x,x)

    def test_15_twelve_old_dev_source_native_summary(self):
        s=summarize_masks(make_observations())
        self.assertEqual(s["FULL"][
            "first_native_token_155_partial_byte_greedy"],12)
        self.assertEqual(s["HEAD_OFF"][
            "first_native_token_30_dot_greedy"],12)
        self.assertEqual(s["FULL_to_HEAD_OFF_first_native_greedy_changed"],12)
        self.assertEqual(s["FULL_to_NO_BIAS_first_native_greedy_changed"],12)
        self.assertEqual(s["FULL_to_NO_LOWRANK_first_native_greedy_changed"],12)

    def test_16_short_readonly_source_case_set_rejected(self):
        with self.assertRaisesRegex(GateFailure,"12 previously"):
            summarize_masks(make_observations()[:10])

    def test_17_missing_head_mask_kind_rejected(self):
        cases=make_observations()
        cases[0]["masks"].pop("NO_BIAS")
        with self.assertRaisesRegex(GateFailure,"12 previously"):
            summarize_masks(cases)

    def test_18_final_stage_is_forensic_not_optimizer(self):
        self.assertIn("CAUSAL_HEAD_ATTRIBUTION",STAGE)
        self.assertEqual(HEAD_GAIN,.25)


if __name__=="__main__":
    main()
