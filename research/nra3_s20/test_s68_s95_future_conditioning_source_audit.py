"""S68 source-only frozen S95 function and no-gold future-feature tests."""
import ast
import unittest
from pathlib import Path
import numpy as np
from research.nra3_s20.s68_s95_future_conditioning_source_audit import (
    ByteTokenSurrogate,audit_spent_raw_only,original_s95_function_from_ast,
    source_word_mid_cut,vector_pair_drift,
)


class TestS68S95FutureSource(unittest.TestCase):
    def test_really_extracts_original_S95_not_reimplemented(self):
        f,sha,fnsha=original_s95_function_from_ast()
        self.assertEqual(len(sha),64)
        self.assertEqual(len(fnsha),64)
        self.assertEqual(f(ByteTokenSurrogate(),"А Б").shape,(512,))

    def test_shared_same_prefix_two_futures_change_hashed_prompt(self):
        f,_,_=original_s95_function_from_ast()
        s="Курьер сообщил, что деньги передали Анне, но это были только слухи."
        cut=source_word_mid_cut(s)
        r=vector_pair_drift(f,ByteTokenSurrogate(),s,cut)
        self.assertFalse(r["full_vs_prefix_prompt_features"]["exact_equal"])
        self.assertFalse(
            r["shared_prefix_suffixA_vs_suffixB_features"]["exact_equal"])
        self.assertFalse(r["native_calibri_tokenizer_used"])
        self.assertTrue(r["surrogate_only_no_native_H8"])

    def test_cut_is_source_only_and_not_native_proof(self):
        s="Это был сложный рассказ о заявлении, но никто не подтверждал факт."
        cut=source_word_mid_cut(s)
        self.assertEqual(s[cut]," ")
        self.assertGreater(cut,4)
        f,_,_=original_s95_function_from_ast()
        with self.assertRaisesRegex(ValueError,"boundary"):
            vector_pair_drift(f,ByteTokenSurrogate(),s,len(s))

    def test_direct_original_vector_is_unchanged_and_normalized(self):
        f,_,_=original_s95_function_from_ast()
        v=f(ByteTokenSurrogate(),"Один участник рассказал о передаче.")
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertAlmostEqual(float(np.linalg.norm(v)),1,places=5)
        self.assertTrue(np.array_equal(
            v,f(ByteTokenSurrogate(),"Один участник рассказал о передаче.")))

    def test_no_duplicate_raw_or_training_or_online_activation(self):
        raw=tuple(f"Сначала это обсуждалось в зале номер {i}, затем вопрос закрыли."
                  for i in range(32))
        out=audit_spent_raw_only(raw)
        self.assertEqual(out["n_rows"],32)
        self.assertEqual(out["n_full_vs_prefix_feature_change"],32)
        self.assertEqual(out["n_equal_prefix_suffix_swap_feature_change"],32)
        self.assertTrue(out["no_fresh_held_or_EVAL32_128_or_gold"])
        self.assertFalse(out["causal_original_streaming_root_cause_or_safety_proven"])
        self.assertEqual(out["output_gate"],0)
        with self.assertRaisesRegex(ValueError,"exact already-spent"):
            audit_spent_raw_only(raw[:-1])

if __name__=="__main__":
    unittest.main()
