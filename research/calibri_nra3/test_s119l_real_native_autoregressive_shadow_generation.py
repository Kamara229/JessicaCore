"""S119l autoregressive feedback and original no-output-mutation safety tests."""
import unittest
from unittest.mock import patch
import torch
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import (
    NativeFullHeadShadow)
from research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation import (
    autoregressive_stream,strict_chat_source_ids,prompt_id_digest,
    PROMPTS,MAX_NEW,true_fullprefix_step,
)

class TinyNativeTokenizer:
    vocab_size=19
    eos_token_id=8
    def decode(self,ids,skip_special_tokens=True):
        return " ".join(map(str,ids))


def dummy_next_logits(model,sources,bridge,full,expansion,features,ids,shadow):
    """Two independent original full vocab streams, lexical IDs only."""
    base=torch.zeros(1,19)
    trial=torch.zeros(1,19)
    if len(ids)==3:
        ib,it=4,5
    elif len(ids)==4:
        ib,it=(6,7) if ids[-1] in (4,5) else (9,9)
    else:
        ib,it=8,8
    base[0,ib]=10
    trial[0,it]=10
    return base,trial

class S119LRealNativeAutoregressiveContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119125)
        torch.set_num_threads(2)

    def test_01_research_source_fixed_three_no_gold_answer(self):
        self.assertEqual(len(PROMPTS),3)
        self.assertEqual(len({prompt_id_digest(x[1]) for x in PROMPTS}),3)
        self.assertTrue(all(x[1].strip()==x[1] and x[1] for x in PROMPTS))
        self.assertLessEqual(MAX_NEW,5)

    def test_02_baseline_native_GREEDY_reenters_real_original_next_forward(self):
        t=TinyNativeTokenizer()
        child=NativeFullHeadShadow()
        with patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.strict_chat_source_ids",
            return_value=[1,2,3]), patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.s160.prompt_only_vector",
            return_value=torch.zeros(512)), patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.true_fullprefix_step",
            side_effect=dummy_next_logits) as fn:
            r=autoregressive_stream(t,None,None,None,None,None,
                                    child,"source only",candidate=False)
        self.assertEqual(r["new_generated_token_ids"],[4,6,8])
        self.assertEqual(r["n_forward_actual_original43m_calls"],3)
        self.assertTrue(r["eos_generated"])
        prefixes=[call.args[6] for call in fn.call_args_list]
        self.assertEqual(prefixes,[[1,2,3],[1,2,3,4],[1,2,3,4,6]])

    def test_03_shadow_chosen_tokens_feed_back_not_baseline(self):
        t=TinyNativeTokenizer()
        child=NativeFullHeadShadow()
        with patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.strict_chat_source_ids",
            return_value=[1,2,3]), patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.s160.prompt_only_vector",
            return_value=torch.zeros(512)), patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.true_fullprefix_step",
            side_effect=dummy_next_logits) as fn:
            r=autoregressive_stream(t,None,None,None,None,None,
                                    child,"source only",candidate=True)
        self.assertEqual(r["new_generated_token_ids"],[5,7,8])
        self.assertEqual(r["n_forward_actual_original43m_calls"],3)
        self.assertEqual([call.args[6] for call in fn.call_args_list],
                         [[1,2,3],[1,2,3,5],[1,2,3,5,7]])

    def test_04_initial_question_route_held_fixed_during_generation(self):
        t=TinyNativeTokenizer()
        child=NativeFullHeadShadow()
        with patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.strict_chat_source_ids",
            return_value=[1,2,3]), patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.s160.prompt_only_vector",
            return_value=torch.zeros(512)) as vec, patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.true_fullprefix_step",
            side_effect=dummy_next_logits):
            autoregressive_stream(t,None,None,None,None,None,
                                  child,"fixed question",candidate=True)
        self.assertEqual(vec.call_count,1)
        self.assertEqual(vec.call_args.args[1],"fixed question")

    def test_05_strict_native_original_prompt_contract(self):
        t=TinyNativeTokenizer()
        with patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.s160.prompt_tokens",
            return_value=[1,2,3]):
            self.assertEqual(strict_chat_source_ids(t,"one"),[1,2,3])
        with patch(
            "research.calibri_nra3.s119l_real_native_autoregressive_shadow_generation.s160.prompt_tokens",
            return_value=[1,99]):
            with self.assertRaises(GateFailure):
                strict_chat_source_ids(t,"invalid native tokenizer ID")

    def test_06_true_actual_fullprefix_forward_rejects_invalid_activation(self):
        child=NativeFullHeadShadow()
        child.output_gate=1
        with self.assertRaises(GateFailure):
            true_fullprefix_step(None,None,None,None,None,torch.zeros(512),
                                 [1,2,3],child)

    def test_07_no_silent_truncation_beyond_five_tokens(self):
        self.assertEqual(MAX_NEW,5)
        self.assertFalse(NativeFullHeadShadow().candidate_accepted)

if __name__=="__main__":
    unittest.main()
