"""S119n fail-closed self-feeding ON-PATH native original decoder contracts.

No synthetic mock here is a 43M result: only SHA-pinned real CI run
is permitted to establish actual decoder and GT on-path generation.
"""
import unittest
from unittest.mock import patch
import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import NativeFullHeadShadow
from research.calibri_nra3.s119m_real43m_gt_interlayer_onpath_joint import LearnedOnPathInterlayer
from research.calibri_nra3.s119n_true_onpath_autoregressive_native_ab import (
    PROMPTS,MAX_NEW,greedy_stream,authentic_two_pass_step,
)

class TinyTokenizer:
    vocab_size=19
    eos_token_id=8
    def decode(self,ids,skip_special_tokens=True):
        return " ".join(map(str,ids))

class S119NTrueDeepOnpathNativeAutoregression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119127)
        torch.set_num_threads(2)
        self.adapter=LearnedOnPathInterlayer(NativeFullHeadShadow()).eval()

    def test_01_only_precommitted_unlabeled_prompts(self):
        self.assertEqual(len(PROMPTS),4)
        self.assertEqual(len(set(q for _,q in PROMPTS)),4)
        self.assertEqual(MAX_NEW,5)
        self.assertTrue(all(q and kind for kind,q in PROMPTS))

    @staticmethod
    def dummy_step(_model,_src,_br,_full,_exp,_tok,_ad,prefix):
        base=torch.zeros(1,19)
        trial=torch.zeros(1,19)
        if len(prefix)==3:
            b,t=4,5
        elif len(prefix)==4:
            b,t=6,7
        else:
            b,t=8,8
        base[0,b]=10
        trial[0,t]=10
        return base,trial,{
            "full_vocab_top1_original":b,
            "full_vocab_top1_true_onpath":t,
            "true_interlayer_residual_l2":0.01,
            "max_abs_native_full_logits_delta":1.0,
        }

    def run_mocked(self,onpath):
        seen=[]
        def observe(*args):
            seen.append(list(args[-1]))
            return self.dummy_step(*args)
        tok=TinyTokenizer()
        with patch(
            "research.calibri_nra3.s119n_true_onpath_autoregressive_native_ab.strict_chat_source_ids",
            return_value=[1,2,3]), patch(
            "research.calibri_nra3.s119n_true_onpath_autoregressive_native_ab.authentic_two_pass_step",
            side_effect=observe):
            result=greedy_stream(tok,None,None,None,None,None,
                                 self.adapter,"unlabeled",onpath=onpath)
        return result,seen

    def test_02_original_generated_native_ids_are_feedback_not_gold(self):
        result,seen=self.run_mocked(False)
        self.assertEqual(result["generated_native_token_ids"],[4,6,8])
        self.assertTrue(result["eos_generated"])
        self.assertEqual(seen,[[1,2,3],[1,2,3,4],[1,2,3,4,6]])
        self.assertEqual(result["actual_original_cpu_forward_calls"],6)

    def test_03_onpath_generated_native_ids_are_OWN_independent_feedback(self):
        result,seen=self.run_mocked(True)
        self.assertEqual(result["generated_native_token_ids"],[5,7,8])
        self.assertEqual(seen,[[1,2,3],[1,2,3,5],[1,2,3,5,7]])
        self.assertEqual(result["actual_original_cpu_forward_calls"],6)

    def test_04_hard_stop_original_vocab_and_zero_research_gate(self):
        self.adapter.output_gate=1
        with self.assertRaises(GateFailure):
            authentic_two_pass_step(None,None,None,None,None,
                                    TinyTokenizer(),self.adapter,[1,2,3])

    def test_05_original_old_53751_not_modifiable_during_generation(self):
        old=self.adapter.shadow.unified.nra3_neural_weights
        self.assertEqual(sum(p.numel() for p in old.parameters()),53751)
        self.assertTrue(all(not p.requires_grad for p in old.parameters()))
        self.assertFalse(self.adapter.candidate_accepted)
        self.assertEqual(self.adapter.output_gate,0)
        self.assertEqual(self.adapter.shadow.output_gate,0)

    def test_06_exact_native_prefix_only_never_injected_unknown_vocab(self):
        with self.assertRaises(GateFailure):
            authentic_two_pass_step(None,None,None,None,None,
                                    TinyTokenizer(),self.adapter,[1,190])

if __name__=="__main__":
    unittest.main()
