"""S66 six gold-blind source-only ABI tests, no causal certification."""
import unittest
import torch
from types import SimpleNamespace
from research.nra3_s20.s66_original_gt_token_prefix_kv_parity import (
    stable_native_prefix_cut,compare_real_native,SUBSET,forensic_source_row,
)
from research.nra3_s20.s66_frozen_new32_raw_only_DO_NOT_TRAIN import (
    SOURCE_ROWS,FAMILIES,raw_only_inputs,raw_sha256,
)

class TestS66RealCausalProbeABI(unittest.TestCase):
    def test_S66_new32_four_8case_strata_raw_only(self):
        self.assertEqual(len(raw_only_inputs()),32)
        self.assertEqual(len(SUBSET),8)
        self.assertEqual(len(raw_sha256()),64)
        self.assertEqual([sum(z[1]==f for z in SOURCE_ROWS) for f in FAMILIES],[8]*4)

    def test_source_native_word_cut_only(self):
        words=[SimpleNamespace(native_token_positions=(j+2,),
                 causally_safe=True,shared_native_tokens=())
                 for j in range(8)]
        c=stable_native_prefix_cut(words,list(range(18)))
        self.assertTrue(c["prefix_selection_from_source_only"])
        self.assertGreaterEqual(c["cut"],4)
        self.assertLessEqual(c["cut"],15)

    def test_shared_bpe_never_chosen(self):
        w=[SimpleNamespace(native_token_positions=(i+2,),
              causally_safe=False,shared_native_tokens=(2,)) for i in range(8)]
        with self.assertRaisesRegex(ValueError,"no S66 BPE-safe"):
            stable_native_prefix_cut(w,list(range(20)))

    def test_true_float32_prefix_KV_mismatch_measured_not_repaired(self):
        full=torch.zeros((12,320),dtype=torch.float16)
        prefix=torch.zeros((6,320),dtype=torch.float16)
        kv=torch.zeros((2,320),dtype=torch.float16)
        kv[0,5]=.25
        p=compare_real_native(full,prefix,kv,6)
        self.assertTrue(p["prefill_hidden320_at_strict_word_end"]["exact_fp16_all_320"])
        self.assertEqual(p["KV_cache_incremental_hidden320_next2"]["max_abs"],.25)
        self.assertTrue(p["cannot_certify_autoregressive_streaming_from_this_probe"])

    def test_exact_match_still_NOT_online_causal_certified(self):
        full=torch.zeros((12,320),dtype=torch.float16)
        v=compare_real_native(full,full[:6].clone(),full[6:8].clone(),6)
        self.assertEqual(v["KV_cache_incremental_hidden320_next2"]["max_abs"],0.)
        self.assertTrue(v["original_GT_full_context_future_source_not_masked_by_prompt_features"])
        self.assertTrue(v["fail_closed_online_activation"])

    def test_source_row_precommitted_gates_not_mutated(self):
        info={"cut":6,"word_index":1,"original_native_token_count":17,
              "prefix_selection_from_source_only":True}
        met={"cannot_certify_autoregressive_streaming_from_this_probe":True}
        d=forensic_source_row("T01",SOURCE_ROWS[0][2],info,met)
        self.assertEqual(d["output_gate"],0)
        self.assertFalse(d["candidate_accepted"])
        self.assertTrue(d["independent_human_gold_NEVER_LOADED_during_probe"])

if __name__=="__main__":unittest.main()
