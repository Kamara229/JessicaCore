"""S119q2g static provenance and own-token BPE no-leakage contracts.

Fake tokenizer/decoder prove generation uses ONLY previous native model
predictions, never answer labels. Real S142 original Core+GT and trained q2f
physical FP16 evaluation is an independent CI stage from these unit tests.
"""
from __future__ import annotations

import copy
from collections import Counter
from unittest import TestCase,main
from unittest.mock import patch

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2g_fresh_own_token_source import (
    EXAMPLES,KINDS,EXPECTED_SOURCE_SHA256,rows,validate,normalized,
    source_hash,STAGE as SOURCE_STAGE,
)
from research.calibri_nra3.s119q2g_physical_own_token_generation import (
    MAX_NEW_NATIVE,MAX_POSSIBLE_REAL_ORIGINAL_FORWARDS,
    SAMPLE_COUNT,MODELS,own_token_stream,score_full_untrained_source,
)


class FakeTok:
    vocab_size=49152
    eos_token_id=2
    def decode(self,ids,skip_special_tokens=False):
        return "".join(chr(t) for t in ids if t!=2)


def fake_lm(picked):
    t=torch.full((1,49152),-12.,dtype=torch.float32)
    t[0,picked]=11.
    return t


def fixture_data(answer="AB.",eos=True):
    records=[]
    for row in rows():
        streams={}
        for variant in MODELS:
            ids=[ord(x) for x in answer]
            if eos:
                ids.append(2)
            streams[variant]={
                "variant":variant,
                "generated_native_token_ids":ids,
                "generated_native_decoded_text":answer,
                "native_eos_emitted":eos,
                "generation_truncated_at_token_cap":not eos,
                "real_original_full_decoder_forward_calls":len(ids),
                "no_label_or_future_answer_token_in_decoder_prefix":True,
            }
        records.append({
            "kind":row["source_kind"],"expected_short_answer":answer,
            "results":streams,
        })
    return records


class S119Q2GFreshNativeGreedyContracts(TestCase):
    def test_01_exact_twelve_new_original_author_cases(self):
        r=rows()
        self.assertEqual(len(r),12)
        self.assertEqual(Counter(v["source_kind"] for v in r),
                         {k:2 for k in KINDS})
        self.assertEqual(len({v["question"] for v in r}),12)
        self.assertTrue(all(v["split"]=="FRESH_SOURCE_PROBE_NEVER_TRAIN"
                            for v in r))

    def test_02_source_sha_of_canonical_rows(self):
        r=rows()
        e=validate(r)
        self.assertEqual(e["twelve_source_sha256"],source_hash(r))
        self.assertEqual(len(e["twelve_source_sha256"]),64)
        if EXPECTED_SOURCE_SHA256 is not None:
            self.assertEqual(EXPECTED_SOURCE_SHA256,e["twelve_source_sha256"])
        self.assertTrue(e["not_independent_blind_holdout"])

    def test_03_never_reuse_prior_q2a_q2c1_S119n_examples(self):
        e=validate(rows())
        self.assertTrue(e["not_trained_or_seen_in_prior_traces"])
        self.assertTrue(e["no_old_EVAL128_S118_B_GOLD_or_J7"])

    def test_04_mutated_source_or_answer_fails_closed(self):
        a=copy.deepcopy(rows())
        a[-1]["canonical_short_answer"]="Исправлено."
        with self.assertRaisesRegex(ValueError,"changed prompts"):
            validate(a)

    def test_05_new_test_source_has_no_training_data_splits(self):
        a=rows()
        a[0]["split"]="TRAIN"
        with self.assertRaisesRegex(ValueError,"changed prompts"):
            validate(a)

    def test_06_new_source_normalization_is_fixed_and_nonsemantic(self):
        self.assertEqual(normalized("  ИЛЬЯ!!!   "),normalized("Илья."))
        self.assertNotEqual(normalized("нет"),normalized("да"))
        self.assertNotEqual(normalized("8"),normalized("17"))

    def test_07_real_full_vocab_and_three_predeclared_models(self):
        self.assertEqual(MODELS,(
            "frozen_original","trained_contextual","trained_token_local"))
        self.assertEqual(SAMPLE_COUNT,12)
        self.assertEqual(MAX_NEW_NATIVE,12)
        self.assertEqual(MAX_POSSIBLE_REAL_ORIGINAL_FORWARDS,432)

    def test_08_fake_original_autoregressive_uses_OWN_predictions_only(self):
        observations=[]
        def baseline(_model,_s,_b,_f,_e,ids,features):
            observations.append(list(ids))
            token=65 if len(ids)==2 else 66 if len(ids)==3 else 2
            return fake_lm(token),torch.zeros(1,12,len(ids),320)
        with patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.strict_chat_source_ids",
            return_value=[19,20]
        ),patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.prompt_features",
            return_value=torch.zeros(512)
        ),patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.first_original_pass",
            side_effect=baseline
        ):
            z=own_token_stream(
                FakeTok(),"Вопрос",None,None,None,None,None,
                variant="frozen_original",max_new=5)
        self.assertEqual(observations,[[19,20],[19,20,65],[19,20,65,66]])
        self.assertEqual(z["generated_native_token_ids"],[65,66,2])
        self.assertEqual(z["generated_native_decoded_text"],"AB")
        self.assertTrue(z["native_eos_emitted"])
        self.assertEqual(z["real_original_full_decoder_forward_calls"],3)

    def test_09_fake_contextual_new_original_decoder_hook_autoregresses(self):
        class DummyAdapter:
            output_gate=0
            candidate_accepted=False
            activation_allowed=False
            projection_enabled=True
        observed=[]
        def candidate(model,s,b,f,e,adapter,ids,features,*,contextual):
            observed.append((list(ids),contextual))
            token=88 if len(ids)==2 else 2
            return fake_lm(token).requires_grad_(True),{
                "candidate_forward_calls":1,
            }
        with patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.strict_chat_source_ids",
            return_value=[1,3]
        ),patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.prompt_features",
            return_value=torch.zeros(512)
        ),patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.onepass_full_native_logits",
            side_effect=candidate
        ):
            z=own_token_stream(
                FakeTok(),"Вопрос",None,None,None,None,None,
                variant="trained_contextual",adapter=DummyAdapter(),
                max_new=4)
        self.assertEqual(observed,[([1,3],True),([1,3,88],True)])
        self.assertEqual(z["generated_native_decoded_text"],"X")
        self.assertEqual(z["real_original_full_decoder_forward_calls"],2)

    def test_10_fake_local_switch_stays_false(self):
        class DummyAdapter:
            output_gate=0
            candidate_accepted=False
            activation_allowed=False
            projection_enabled=True
        observed=[]
        def candidate(m,s,b,f,e,a,ids,features,*,contextual):
            observed.append(contextual)
            return fake_lm(2).requires_grad_(True),{
                "candidate_forward_calls":1,
            }
        with patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.strict_chat_source_ids",
            return_value=[4,5]
        ),patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.prompt_features",
            return_value=torch.zeros(512)
        ),patch(
            "research.calibri_nra3.s119q2g_physical_own_token_generation.onepass_full_native_logits",
            side_effect=candidate
        ):
            z=own_token_stream(
                FakeTok(),"Вопрос",None,None,None,None,None,
                variant="trained_token_local",adapter=DummyAdapter())
        self.assertEqual(observed,[False])
        self.assertTrue(z["native_eos_emitted"])

    def test_11_cap_truncation_not_counted_as_finished_correct_answer(self):
        data=fixture_data(eos=False)
        summary=score_full_untrained_source(data)
        self.assertEqual(summary["frozen_original"]["exact_complete_answers_WITH_native_EOS"],0)
        self.assertEqual(summary["frozen_original"]["visible_exact_text_even_if_truncated"],12)
        self.assertEqual(summary["frozen_original"]["native_eos_finished_count"],0)
        self.assertEqual(summary["frozen_original"]["never_finished_token_cap_count"],12)

    def test_12_real_eos_is_required_for_strict_full_answer_correctness(self):
        data=fixture_data(eos=True)
        summary=score_full_untrained_source(data)
        for variant in MODELS:
            self.assertEqual(
                summary[variant]["exact_complete_answers_WITH_native_EOS"],12)
            self.assertEqual(summary[variant]["native_eos_finished_count"],12)

    def test_13_scoring_denies_missing_or_fake_model_result(self):
        data=fixture_data()
        data[0]["results"].pop("frozen_original")
        with self.assertRaisesRegex(GateFailure,"three original streams"):
            score_full_untrained_source(data)

    def test_14_prompt_with_wrong_variant_adapter_denied(self):
        with self.assertRaisesRegex(GateFailure,"one of three"):
            own_token_stream(
                FakeTok(),"Вопрос",None,None,None,None,None,
                variant="frozen_original",adapter=object())

    def test_15_max_native_token_length_fail_closed(self):
        with self.assertRaisesRegex(GateFailure,"one of three"):
            own_token_stream(
                FakeTok(),"Вопрос",None,None,None,None,None,
                variant="frozen_original",max_new=MAX_NEW_NATIVE+1)

    def test_16_this_stage_explicitly_source_only_not_quality_certified(self):
        self.assertIn("FRESH_SOURCE",SOURCE_STAGE)
        self.assertTrue(all(
            "original_author_constructed_untrained_not_blind"==r["provenance"]
            for r in rows()))


if __name__=="__main__":
    main()
