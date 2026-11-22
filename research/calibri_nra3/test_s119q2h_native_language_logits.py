"""S119q2h fail-closed original native head diagnostic tests.

No model training, GOLD/EVAL, hidden J7 or future-answer tokens in inputs.
"""
from __future__ import annotations

import copy
from collections import Counter
from unittest import TestCase,main
from unittest.mock import patch

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2g_fresh_own_token_source import (
    rows as authored_rows,validate as authored_provenance,
    EXPECTED_SOURCE_SHA256,
)
from research.calibri_nra3.s119q2h_native_language_logits import (
    TOP_K,EXPECTED_PHYSICAL_CALLS,STAGE,native_piece_bucket,
    first_answer_native,physical_rank,summarize_kind,
    MODELS,VOCAB_SIZE,
)


class FakeNative:
    vocab_size=49152
    eos_token_id=2
    def apply_chat_template(self,messages,tokenize=False,
                            add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "<user>"+messages[0]["content"]+"</user><assistant>"
    def encode(self,text,add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=False):
        return "".join(chr(x) for x in ids)


def dummy_spent_cases():
    result=[]
    for x in authored_rows():
        result.append({
            "source_id":x["source_id"],
            "kind":x["source_kind"],
            "rank_evidence":{
                "gold_FIRST_native_rank_49152":112,
                "top1_minus_gold_first_logit_margin":3.5,
                "top16_native_contains_gold_FIRST_token":False,
                "top1_native_piece_bucket":"latin_piece",
            }
        })
    return result


class S119Q2HFullNativeFirstTokenAuditContracts(TestCase):
    def test_01_exact_spent_q2g_source_fixed_sha(self):
        e=authored_provenance(authored_rows())
        self.assertEqual(e["source_example_count"],12)
        self.assertEqual(e["twelve_source_sha256"],EXPECTED_SOURCE_SHA256)
        self.assertEqual(len(EXPECTED_SOURCE_SHA256),64)
        self.assertTrue(e["not_independent_blind_holdout"])

    def test_02_actual_three_models_x12_prompts_budget(self):
        self.assertEqual(MODELS,(
            "frozen_original","trained_contextual","trained_token_local"))
        self.assertEqual(EXPECTED_PHYSICAL_CALLS,36)
        self.assertEqual(TOP_K,16)
        self.assertEqual(VOCAB_SIZE,49152)

    def test_03_native_script_latin(self):
        self.assertEqual(native_piece_bucket("yellow"),"latin_piece")
        self.assertEqual(native_piece_bucket(" Yes"),"latin_piece")

    def test_04_native_script_cyrillic(self):
        self.assertEqual(native_piece_bucket("Ответ"),"cyrillic_piece")
        self.assertEqual(native_piece_bucket(" Илья"),"cyrillic_piece")

    def test_05_native_script_mixed_and_punctuation(self):
        self.assertEqual(native_piece_bucket("AБ"),"mixed_script_piece")
        self.assertEqual(native_piece_bucket("!),  "),
                         "other_punctuation_special_or_incomplete_byte_piece")
        self.assertEqual(native_piece_bucket("�"),
                         "other_punctuation_special_or_incomplete_byte_piece")

    def test_06_native_true_assistant_boundary_causal_answer(self):
        tok=FakeNative()
        original_question="Вопрос?"
        answer="Да."
        with patch(
            "research.calibri_nra3.s119q2h_native_language_logits.strict_chat_source_ids",
            return_value=tok.encode(
                tok.apply_chat_template(
                    [{"role":"user","content":original_question}]))
        ):
            prompt,gold=first_answer_native(tok,original_question,answer)
        self.assertEqual(gold,[ord(x) for x in answer])
        self.assertEqual(len(prompt)+len(gold),len(tok.encode(
            tok.apply_chat_template(
                [{"role":"user","content":original_question}])+answer)))
        self.assertNotIn(ord("Д"),prompt)

    def test_07_boundary_merging_must_fail_not_guess_first_gold(self):
        class BoundaryMerges(FakeNative):
            def encode(self,text,add_special_tokens=False):
                out=super().encode(text,add_special_tokens=add_special_tokens)
                if text.endswith("Да."):
                    out.pop(0)
                return out
        tok=BoundaryMerges()
        with patch(
            "research.calibri_nra3.s119q2h_native_language_logits.strict_chat_source_ids",
            return_value=FakeNative().encode(
                tok.apply_chat_template([{"role":"user","content":"В?"}]))
        ):
            with self.assertRaisesRegex(GateFailure,"preserve causal prompt"):
                first_answer_native(tok,"В?","Да.")

    def test_08_raw_full_head_rank_gold_equal_top1(self):
        tok=FakeNative()
        logits=torch.zeros(1,VOCAB_SIZE)
        logits[0,89]=7.
        logits[0,90]=2.
        z=physical_rank(tok,logits,89,89)
        self.assertEqual(z["gold_FIRST_native_rank_49152"],1)
        self.assertEqual(z["top1_minus_gold_first_logit_margin"],0.)
        self.assertTrue(z["top16_native_contains_gold_FIRST_token"])
        self.assertEqual(z["top1_native_text_piece"],"Y")
        self.assertGreater(z[
            "gold_first_BPE_probability_under_full_native_head"],0.)

    def test_09_raw_full_head_rank_gold_below_top1(self):
        tok=FakeNative()
        logits=torch.zeros(1,VOCAB_SIZE)
        logits[0,89]=7.
        logits[0,90]=5.
        logits[0,91]=4.
        z=physical_rank(tok,logits,91,89)
        self.assertEqual(z["gold_FIRST_native_rank_49152"],3)
        self.assertAlmostEqual(z["top1_minus_gold_first_logit_margin"],3.)

    def test_10_reject_first_id_mismatch_with_original_q2g(self):
        tok=FakeNative()
        logits=torch.zeros(1,VOCAB_SIZE)
        logits[0,89]=1.
        with self.assertRaisesRegex(GateFailure,"reproduce q2g"):
            physical_rank(tok,logits,89,90)

    def test_11_reject_nonfinite_original_logits(self):
        tok=FakeNative()
        logits=torch.zeros(1,VOCAB_SIZE)
        logits[0,0]=float("nan")
        with self.assertRaisesRegex(GateFailure,"native 49,152"):
            physical_rank(tok,logits,89,89)

    def test_12_reject_partial_vocab_or_wrong_gold(self):
        tok=FakeNative()
        with self.assertRaisesRegex(GateFailure,"native 49,152"):
            physical_rank(tok,torch.zeros(1,32),1,1)
        with self.assertRaisesRegex(GateFailure,"native 49,152"):
            physical_rank(tok,torch.zeros(1,VOCAB_SIZE),49152,0)

    def test_13_spent_source_script_ranking_summary_not_accuracy(self):
        z=summarize_kind(dummy_spent_cases())
        self.assertEqual(z["n_source_prompts"],12)
        self.assertEqual(z["median_gold_first_native_rank"],112.)
        self.assertEqual(z["mean_gold_first_native_rank"],112.)
        self.assertEqual(z["gold_native_FIRST_in_top16"],0)
        self.assertEqual(z["top1_piece_hints"],{"latin_piece":12})
        self.assertEqual(len(z["by_source_domain"]),6)
        self.assertTrue(z[
            "not_blind_and_no_new_training_or_hyperparameter_selection"])

    def test_14_reject_incomplete_or_duplication_of_domain_source(self):
        with self.assertRaisesRegex(GateFailure,"spent 12"):
            summarize_kind(dummy_spent_cases()[:-1])
        a=dummy_spent_cases()
        a[0]["kind"]="negation"
        with self.assertRaisesRegex(GateFailure,"spent 12"):
            summarize_kind(a)

    def test_15_audit_must_not_claim_language_of_word_from_bpe(self):
        piece=native_piece_bucket("�")
        self.assertNotEqual(piece,"latin_piece")
        self.assertNotEqual(piece,"cyrillic_piece")

    def test_16_provenance_and_no_new_learning_at_this_stage(self):
        self.assertIn("LANGUAGE_LOGIT_FORENSICS",STAGE)
        self.assertEqual(len(authored_rows()),12)
        self.assertEqual(Counter(r["source_kind"] for r in authored_rows()),
                         {k:2 for k in (
                             "roles_word_order","negation",
                             "transitive_comparison","event_time",
                             "arithmetic","paraphrase")})


if __name__=="__main__":
    main()
