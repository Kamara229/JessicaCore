"""S119q2i8b original source 135M vs 43M strict native tokenizer tests."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase,main

import torch
from torch import nn

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i8b_source135m_vs_original43m_native import (
    STAGE,REPORT,LABELS,MAX_TOKENS,KIND_COUNT,NATIVE_VOCAB,
    selected_already_scored_dev,physical_q2i8a_fitted_child,
    physical_q2i8a_gates,native_rank,source_teacher_step,
    greedy_own_stream,summarize,
)


class FakeToken:
    eos_token_id=2
    all_special_ids=[2]
    def convert_ids_to_tokens(self,x):
        return {155:"Ð",201:"Ĳ",30:".",2:"<|im_end|>",42:"A"}.get(x,".")
    def decode(self,ids,skip_special_tokens=True):
        piece=[]
        for x in ids:
            if x==2 and skip_special_tokens:continue
            if x==155:piece.append("\ufffd")
            elif x==201:piece.append("\ufffd")
            elif x==30:piece.append(".")
            elif x==42:piece.append("A")
            else:piece.append("?")
        return "".join(piece)


class FakeSourceTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls=0
        self.last_inputs=[]
    def forward(self,input_ids,past_key_values=None,use_cache=False):
        self.calls+=1
        self.last_inputs.append(input_ids[0].tolist())
        v=torch.full((1,input_ids.shape[1],NATIVE_VOCAB),-8.0)
        v[0,-1,155]=9.0 if self.calls==1 else -8.0
        v[0,-1,2]=9.0 if self.calls>1 else -8.0
        return SimpleNamespace(logits=v)


def fake_dev(id="S119Q2I8A_test",gold=None):
    return {
        "id":id,"split":"DEV","kind":"unicode_word_copy",
        "prefix":[11,12,13],
        "answer_native_with_eos":gold or [155,201,2],
        "gold_answer_for_scoring_ONLY":"А",
    }


def fake_seven():
    return [{
        "kind":kind,
        "strict_entire_native_BPE_answer_including_EOS_correct":False,
        "native_visible_text_equals_gold":False,
        "native_EOS_emitted":True,
        "native_EOS_after_INVALID_or_INCOMPLETE_UTF8":i%2==0,
        "entire_generated_prefix_actual_GPT2_UTF8_state":
            "VALID_COMPLETE_UTF8" if i%2 else "INCOMPLETE_UTF8_NEEDS_CONTINUATION",
        "true_first_native_BPE_correct":i%2==0,
        "correct_initial_BPE_but_wrong_second":i%2==0,
        "real_original_native_49152_head_forward_calls":2,
    } for i,kind in enumerate(
        ("unicode_word_copy","reversed_agent_roles","scope_negation",
         "numeric_arithmetic","event_chronology",
         "comparative_references","lexical_paraphrase"))]


class TestQ2I8BSourceTeacherVsCompressedOriginal(TestCase):
    def test_01_exact_pinned_source_only_seven_old_DEV(self):
        source=selected_already_scored_dev()
        self.assertEqual(len(source),KIND_COUNT)
        self.assertEqual(Counter(x["kind"] for x in source),
                         {k:1 for k in (
                             "unicode_word_copy","reversed_agent_roles",
                             "scope_negation","numeric_arithmetic",
                             "event_chronology","comparative_references",
                             "lexical_paraphrase")})
        self.assertTrue(all(x["split"]=="DEV" for x in source))
        self.assertTrue(all(x["id"].endswith("_10") for x in source))

    def test_02_only_three_actual_pretrained_native_model_categories(self):
        self.assertEqual(LABELS,(
            "FROZEN_SOURCE_135M_INSTRUCT",
            "FROZEN_S142_43M_ORIGINAL",
            "Q2I8A_FITTED_43M_GT12_NRA3_NPA1"))

    def test_03_identical_real_native_vocab_size(self):
        self.assertEqual(NATIVE_VOCAB,49152)
        self.assertEqual(MAX_TOKENS,18)
        self.assertLess(MAX_TOKENS,24)

    def test_04_real_native_first_correct_rank(self):
        x=torch.full((1,NATIVE_VOCAB),-4.0)
        x[0,155]=8.0;x[0,201]=7.0
        m=native_rank(x,201)
        self.assertEqual(m["original_full_native_gold_first_rank"],2)
        self.assertEqual(m["original_full_native_greedy_first_id"],155)
        self.assertFalse(m["original_full_native_correct_first"])
        self.assertEqual(len(m["top8_native_ids"]),8)

    def test_05_real_native_first_correct_top1(self):
        x=torch.full((1,NATIVE_VOCAB),-10.)
        x[0,201]=7.0
        m=native_rank(x,201)
        self.assertEqual(m["original_full_native_gold_first_rank"],1)
        self.assertTrue(m["original_full_native_correct_first"])

    def test_06_non_native_restricted_logits_REJECTED(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            native_rank(torch.zeros(1,256),30)

    def test_07_nan_native_head_REJECTED(self):
        x=torch.zeros((1,NATIVE_VOCAB));x[0,155]=float("nan")
        with self.assertRaisesRegex(GateFailure,"49,152"):
            native_rank(x,201)

    def test_08_out_of_vocab_gold_REJECTED(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            native_rank(torch.zeros((1,NATIVE_VOCAB)),NATIVE_VOCAB)

    def test_09_teacher_full_model_receives_actual_OWN_prefix_only(self):
        t=FakeSourceTeacher()
        a=source_teacher_step(t,[10,11])
        b=source_teacher_step(t,[10,11,155])
        self.assertEqual(a.shape,(1,NATIVE_VOCAB))
        self.assertEqual(b.shape,(1,NATIVE_VOCAB))
        self.assertEqual(t.last_inputs,[[10,11],[10,11,155]])
        self.assertEqual(torch.argmax(a).item(),155)
        self.assertEqual(torch.argmax(b).item(),2)

    def test_10_teacher_full_vocab_no_truncation_or_fixed_byte(self):
        t=FakeSourceTeacher()
        with self.assertRaisesRegex(GateFailure,"SAME chat prefix"):
            source_teacher_step(t,[49152,20])
        self.assertEqual(t.calls,0)

    def test_11_self_native_teacher_stream_never_sees_correct_gold(self):
        t=FakeSourceTeacher()
        x=fake_dev()
        actual=greedy_own_stream(
            FakeToken(),x,lambda ids:source_teacher_step(t,ids),
            label="FROZEN_SOURCE_135M_INSTRUCT")
        self.assertEqual(actual["native_generated_BPE_ids"],[155,2])
        self.assertEqual(t.last_inputs,[[11,12,13],[11,12,13,155]])
        self.assertFalse(
            actual["strict_entire_native_BPE_answer_including_EOS_correct"])
        self.assertTrue(actual["native_EOS_emitted"])
        self.assertTrue(actual["never_gold_target_BPE_in_own_self_prefix"])

    def test_12_teacher_repeated_native_155_first_byte_not_a_correct_answer(self):
        t=FakeSourceTeacher()
        r=greedy_own_stream(
            FakeToken(),fake_dev(),lambda ids:source_teacher_step(t,ids),
            label="FROZEN_SOURCE_135M_INSTRUCT")
        self.assertTrue(r["true_first_native_BPE_correct"])
        self.assertTrue(r["correct_initial_BPE_but_wrong_second"])
        self.assertTrue(r["native_EOS_after_INVALID_or_INCOMPLETE_UTF8"])

    def test_13_full_correct_native_BPE_sequence_plus_real_EOS_only(self):
        step=[155,201,2]
        count=[0]
        def model(_prefix):
            i=count[0];count[0]+=1
            x=torch.full((1,NATIVE_VOCAB),-9.0)
            x[0,step[i]]=9.0
            return x
        got=greedy_own_stream(
            FakeToken(),fake_dev(),model,
            label="FROZEN_S142_43M_ORIGINAL")
        self.assertTrue(
            got["strict_entire_native_BPE_answer_including_EOS_correct"])
        self.assertEqual(got["native_generated_BPE_ids"],[155,201,2])
        self.assertEqual(got["real_original_native_49152_head_forward_calls"],3)

    def test_14_unsafe_unknown_label_denied(self):
        with self.assertRaises(AssertionError):
            greedy_own_stream(
                FakeToken(),fake_dev(),lambda ids:torch.zeros(1,NATIVE_VOCAB),
                label="UNSAFE_MOCK_TEACHER")

    def test_15_no_current_or_future_answer_in_own_prefix(self):
        target=[30,30,2]
        inputs=[]
        def stub(prefix):
            inputs.append(prefix.copy())
            x=torch.full((1,NATIVE_VOCAB),-10.)
            x[0,target[len(inputs)-1]]=10.
            return x
        result=greedy_own_stream(FakeToken(),fake_dev(),stub,
                                 label="FROZEN_S142_43M_ORIGINAL")
        self.assertEqual(inputs,[[11,12,13],[11,12,13,30],
                                 [11,12,13,30,30]])
        self.assertEqual(result["native_generated_BPE_ids"],target)
        self.assertFalse(result["native_visible_text_equals_gold"])

    def test_16_wrong_native_original_FP16_path_REJECTED(self):
        with self.assertRaisesRegex(GateFailure,"Q2I8a recurrent"):
            physical_q2i8a_fitted_child(
                Path("/tmp/unsafe.safetensors"),
                {"sha256":"x","file_bytes":0},None)

    def test_17_wrong_original_GT12_gate_REJECTED(self):
        with self.assertRaisesRegex(GateFailure,"Q2I8a original"):
            physical_q2i8a_gates(
                Path("/tmp/unsafe.safetensors"),
                {"sha256":"x","file_bytes":0})

    def test_18_diagnostics_count_source_teacher_and_students_separately(self):
        result=summarize(fake_seven())
        self.assertEqual(result["source_count"],7)
        self.assertEqual(result["native_EOS_finished"],7)
        self.assertEqual(result["correct_gold_FIRST_native_BPE_top1"],4)
        self.assertEqual(result["total_actual_full_native_decoder_forwards"],14)

    def test_19_missing_one_of_seven_domain_sources_fails_closed(self):
        with self.assertRaisesRegex(GateFailure,"seven known"):
            summarize(fake_seven()[:-1])

    def test_20_source_135M_is_not_an_oracle_or_external_blind_evaluator(self):
        self.assertIn("SOURCE_135M_VS_REAL_S142_43M_NATIVE_RU",STAGE)
        self.assertTrue(REPORT.endswith(".json"))


if __name__=="__main__":
    main()
