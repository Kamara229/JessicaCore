"""S119n fast fail-closed contract suite; original43M separate physical CI."""
import unittest
from unittest.mock import patch
import torch
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import NativeFullHeadShadow
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import LearnedOnPathInterlayer
from research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts import (
    PROMPTS,MAX_NEW,stream,first_original_pass,
    actual_original_onpath_second_pass,verify_zero_gate_exact_onpath,
)

class FakeNativeTokenizer:
    vocab_size=19
    eos_token_id=8
    def decode(self,ids,skip_special_tokens=True):
        return " ".join(map(str,ids))


class S119NRealInterlayerAutoregressiveSmoke(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119127)
        torch.set_num_threads(2)
        self.tok=FakeNativeTokenizer()
        self.adapter=LearnedOnPathInterlayer(NativeFullHeadShadow()).eval()
        self.original_prefixes=[]
        self.intervened_prefixes=[]

    def original(self,*args):
        prefix=list(args[5])
        self.original_prefixes.append(prefix)
        k=len(prefix)
        token=(4 if k==3 else 6 if k==4 else 8)
        logits=torch.zeros(1,self.tok.vocab_size)
        logits[0,token]=10.
        return logits,torch.zeros(1,12,k,320)

    def intervened(self,*args):
        prefix=list(args[6])
        self.intervened_prefixes.append(prefix)
        token=(5 if len(prefix)==3 else 7 if len(prefix)==4 else 8)
        logits=torch.zeros(1,self.tok.vocab_size)
        logits[0,token]=10.
        return logits

    def run_mock(self,trained):
        with patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.strict_chat_source_ids",
            return_value=[1,2,3]),patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.prompt_features",
            return_value=torch.zeros(512)),patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.first_original_pass",
            side_effect=self.original),patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.actual_original_onpath_second_pass",
            side_effect=self.intervened),patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.verify_zero_gate_exact_onpath"
            ) as zero:
            result=stream(self.tok,None,None,None,None,None,self.adapter,
                          "unlabeled",trained_onpath=trained)
        return result,zero.call_count

    def test_01_six_diverse_unlabeled_source_only_requests(self):
        self.assertEqual(len(PROMPTS),6)
        self.assertEqual(len(set(k for k,_ in PROMPTS)),6)
        self.assertEqual(len(set(v for _,v in PROMPTS)),6)
        self.assertTrue(all(v.strip()==v and v for _,v in PROMPTS))
        self.assertEqual(MAX_NEW,8)

    def test_02_baseline_real_greedy_native_feedback(self):
        result,zero=self.run_mock(False)
        self.assertEqual(result["generated_native_token_ids"],[4,6,8])
        self.assertEqual(result["original_full12layer_prefill_calls"],3)
        self.assertEqual(result["real_onpath_second_decoder_calls"],0)
        self.assertEqual(zero,0)
        self.assertEqual(self.original_prefixes,
                         [[1,2,3],[1,2,3,4],[1,2,3,4,6]])
        self.assertFalse(self.intervened_prefixes)
        self.assertTrue(result["emitted_EOS"])

    def test_03_trained_actual_onpath_token_is_fed_to_next_original(self):
        result,zero=self.run_mock(True)
        self.assertEqual(result["generated_native_token_ids"],[5,7,8])
        self.assertEqual(result["original_full12layer_prefill_calls"],3)
        self.assertEqual(result["real_onpath_second_decoder_calls"],3)
        self.assertEqual(zero,1)
        self.assertEqual(self.original_prefixes,
                         [[1,2,3],[1,2,3,5],[1,2,3,5,7]])
        self.assertEqual(self.intervened_prefixes,self.original_prefixes)
        self.assertTrue(result["emitted_EOS"])

    def test_04_one_frozen_route_per_whole_response(self):
        with patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.strict_chat_source_ids",
            return_value=[1,2,3]),patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.prompt_features",
            return_value=torch.zeros(512)) as frozen,patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.first_original_pass",
            side_effect=self.original),patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.actual_original_onpath_second_pass",
            side_effect=self.intervened),patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.verify_zero_gate_exact_onpath"):
            stream(self.tok,None,None,None,None,None,self.adapter,
                   "one user question",trained_onpath=True)
        self.assertEqual(frozen.call_count,1)
        self.assertEqual(frozen.call_args.args[1],"one user question")

    def test_05_zero_gate_exact_control_restores_learned_gates(self):
        with torch.no_grad():self.adapter.midlayer_gates.fill_(0.077)
        prior=self.adapter.midlayer_gates.detach().clone()
        original=torch.randn(1,19)
        with patch(
            "research.calibri_nra3.s119n_real43m_onpath_autoregressive_sixprompts.actual_original_onpath_second_pass",
            return_value=original):
            verify_zero_gate_exact_onpath(
                None,None,None,None,None,self.adapter,[1,2,3],
                torch.zeros(512),original,torch.randn(1,12,3,320))
        self.assertTrue(torch.equal(self.adapter.midlayer_gates,prior))

    def test_06_unverified_candidate_activation_always_blocked(self):
        with self.assertRaises(GateFailure):
            first_original_pass(None,None,None,None,None,[1],torch.zeros(512))
        self.assertFalse(self.adapter.candidate_accepted)
        self.assertFalse(self.adapter.activation_allowed)
        self.assertEqual(self.adapter.output_gate,0)
        self.assertEqual(sum(p.numel() for p in
            self.adapter.shadow.unified.nra3_neural_weights.parameters()),53751)

    def test_07_logits_compare_real_native_margins_not_accuracy_fabrication(self):
        result,_=self.run_mock(True)
        self.assertEqual(len(result["actual_native_steps"]),3)
        self.assertTrue(all(z["original_top1_top2_logit_margin"]>=0
                            for z in result["actual_native_steps"]))
        self.assertTrue(all(z["actual_onpath_max_abs_logit_change"]>=0
                            for z in result["actual_native_steps"]))

if __name__=="__main__":unittest.main()
