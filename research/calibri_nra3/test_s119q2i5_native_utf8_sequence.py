"""S119q2i5 original 49k native UTF8 causal gold-vs-own contracts.

Synthetic tests enforce exact native token sequencing, not mere decoded
'�' string. Original real physical stage tests same on frozen 43M+GT.
"""
from __future__ import annotations

from collections import Counter
from unittest import TestCase,main

import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2i5_real43m_own_vs_gold_utf8_sequence import (
    utf8_byte_tail_status,first_divergence,decode_step_metrics,
    summarize_sequences,STAGE,SOURCE_COUNT,MAX_SELF_TOKENS,
    MAX_GOLD_TARGET_TOKENS,MODELS,BYTE_PIECE_PATTERN,
    VOCAB_SIZE,
)
from research.calibri_nra3.s119q2i1_authored_ru_train_source import KINDS,rows


class TinyNative:
    def convert_ids_to_tokens(self,x):
        return {
            155:"<0xD0>",201:"<0x90>",202:"<0x91>",
            203:"<0xD1>",204:"<0x81>",30:".",2:"</s>",
            90:"Z",91:"Y",
        }.get(x,str(x))
    def decode(self,ids,skip_special_tokens=True):
        b=bytearray()
        for t in ids:
            raw=self.convert_ids_to_tokens(t)
            if t==2 and skip_special_tokens:
                continue
            m=BYTE_PIECE_PATTERN.fullmatch(raw)
            if m:
                b.append(int(m.group(1),16))
            elif raw==".":
                b.extend(b".")
            else:
                b.extend(raw.encode("ascii",errors="replace"))
        return bytes(b).decode("utf8",errors="replace")


def fake_sequences():
    records=[]
    for k in KINDS:
        records.append({
            "kind":k,
            "first_divergence":{
                "first_wrong_or_missing_native_BPE_position_1based":2,
                "matching_native_BPE_prefix_length":1,
                "first_native_BPE_correct":True,
                "first_native_correct_but_wrong_OR_missing_second":True,
                "full_native_ID_sequence_EXACT":False,
            },
            "teacher_forced_gold_position_details":[{
                "native_target_is_top1":i==0,
                "native_target_CE":float(i+1),
            } for i in range(3)],
            "actual_own_final_UTF8_native_byte_status":{
                "replacement_UFFFD_visible_if_rendered_NOW":True,
            },
            "total_original_S142_Core12_full_native_decoder_forwards":5,
        })
    return records


class S119Q2I5NativeUtf8CausalAuditContracts(TestCase):
    def setUp(self):
        self.tok=TinyNative()

    def test_01_native_source_one_each_six_old_dev_domains(self):
        data=[r for r in rows("DEV")
              if r["id"].rsplit("_",1)[-1]=="06"]
        self.assertEqual(len(data),SOURCE_COUNT)
        self.assertEqual(Counter(r["kind"] for r in data),
                         {k:1 for k in KINDS})

    def test_02_original_three_real_neural_variants(self):
        self.assertEqual(MODELS,(
            "GT_ZERO_BASELINE","GT_RECURRENT","GT_NOCARRY_LOCAL"))
        self.assertEqual(MAX_SELF_TOKENS,12)
        self.assertEqual(MAX_GOLD_TARGET_TOKENS,48)

    def test_03_first_native_utf8_D0_alone_is_INCOMPLETE_not_necessarily_wrong(self):
        a=utf8_byte_tail_status(self.tok,[155])
        self.assertEqual(a["contiguous_trailing_raw_byte_hex"],["D0"])
        self.assertEqual(a["trailing_native_byte_state"],
                         "TRAILING_BYTES_INCOMPLETE_UTF8")
        self.assertTrue(a["replacement_UFFFD_visible_if_rendered_NOW"])
        self.assertTrue(a["incomplete_byte_token_MAY_be_correct_native_prefix"])

    def test_04_correct_continuation_makes_valid_Cyrillic(self):
        a=utf8_byte_tail_status(self.tok,[155,201])
        self.assertEqual(a["decoded_text_if_rendered_NOW"],"А")
        self.assertEqual(a["trailing_native_byte_state"],
                         "TRAILING_BYTES_VALID_UTF8")
        self.assertFalse(a["replacement_UFFFD_visible_if_rendered_NOW"])

    def test_05_repeated_D0_native_start_is_invalid_continuation(self):
        a=utf8_byte_tail_status(self.tok,[155,155])
        self.assertEqual(a["trailing_native_byte_state"],
                         "TRAILING_BYTES_INVALID_UTF8")
        self.assertTrue(a["replacement_UFFFD_visible_if_rendered_NOW"])

    def test_06_more_raw_byte_only_prefix_crosses_Cyrillic_pair_boundary(self):
        a=utf8_byte_tail_status(self.tok,[155,201,203,204])
        self.assertEqual(a["decoded_text_if_rendered_NOW"],"Ас")
        self.assertEqual(a["trailing_native_byte_state"],
                         "TRAILING_BYTES_VALID_UTF8")

    def test_07_nonbyte_period_breaks_contiguous_tail(self):
        a=utf8_byte_tail_status(self.tok,[155,30])
        self.assertEqual(a["contiguous_trailing_raw_byte_hex"],[])
        self.assertEqual(a["trailing_native_byte_state"],
                         "NO_TRAILING_NATIVE_BYTE_TOKENS")
        self.assertTrue(a["replacement_UFFFD_visible_if_rendered_NOW"])

    def test_08_first_mismatch_at_correct_byte_second_token(self):
        d=first_divergence([155,201,30,2],[155,155,30,2])
        self.assertEqual(d["first_wrong_or_missing_native_BPE_position_1based"],2)
        self.assertEqual(d["matching_native_BPE_prefix_length"],1)
        self.assertTrue(d["first_native_correct_but_wrong_OR_missing_second"])
        self.assertFalse(d["full_native_ID_sequence_EXACT"])

    def test_09_first_byte_already_wrong(self):
        d=first_divergence([155,201,2],[30,155,2])
        self.assertEqual(d["first_wrong_or_missing_native_BPE_position_1based"],1)
        self.assertFalse(d["first_native_BPE_correct"])
        self.assertFalse(d["first_native_correct_but_wrong_OR_missing_second"])

    def test_10_missing_EOS_detected_as_incomplete(self):
        d=first_divergence([155,201,2],[155,201])
        self.assertEqual(d["first_wrong_or_missing_native_BPE_position_1based"],3)
        self.assertFalse(d["full_native_ID_sequence_EXACT"])

    def test_11_exact_short_full_native_and_EOS_detected(self):
        d=first_divergence([155,201,2],[155,201,2])
        self.assertIsNone(d["first_wrong_or_missing_native_BPE_position_1based"])
        self.assertTrue(d["full_native_ID_sequence_EXACT"])

    def test_12_overlong_self_added_after_GOLD_EOS_is_divergence(self):
        d=first_divergence([155,201,2],[155,201,2,30])
        self.assertEqual(d["first_wrong_or_missing_native_BPE_position_1based"],4)

    def test_13_native_full_vocab_49152_mandatory(self):
        logits=torch.full((1,VOCAB_SIZE),-7.)
        logits[0,155]=9.
        logits[0,201]=8.
        r=decode_step_metrics(self.tok,logits,201,1)
        self.assertEqual(r["native_target_rank_full_49152"],2)
        self.assertFalse(r["native_target_is_top1"])
        self.assertEqual(r["correct_native_BPE_target_id"],201)
        self.assertEqual(r["native_position_1based"],2)

    def test_14_teacher_forced_correct_target_TOP1(self):
        logits=torch.full((1,VOCAB_SIZE),-7.)
        logits[0,155]=9.
        r=decode_step_metrics(self.tok,logits,155,0)
        self.assertEqual(r["native_target_rank_full_49152"],1)
        self.assertTrue(r["native_target_is_top1"])
        self.assertAlmostEqual(r["native_top1_minus_gold_logit"],0.)

    def test_15_nan_or_proxy_logits_fails_closed(self):
        with self.assertRaisesRegex(GateFailure,"49,152"):
            decode_step_metrics(self.tok,torch.zeros(1,128),2,0)
        x=torch.zeros(1,VOCAB_SIZE)
        x[0,12]=float("nan")
        with self.assertRaisesRegex(GateFailure,"49,152"):
            decode_step_metrics(self.tok,x,2,0)

    def test_16_aggregate_real_native_gold_and_own_full_positions(self):
        s=summarize_sequences(fake_sequences())
        self.assertEqual(s["correct_native_first_BPE"],6)
        self.assertEqual(s["correct_native_first_BPE_BUT_failed_second"],6)
        self.assertEqual(s["total_teacher_forced_full_answer_BPE_plus_EOS_positions"],18)
        self.assertEqual(s["teacher_forced_correct_BPE_top1_positions"],6)
        self.assertEqual(s["original_physical_native_decoder_forwards"],30)
        self.assertTrue(s["already_spent_author_DEV_not_independent_or_blind"])

    def test_17_missing_domain_fail_closed(self):
        with self.assertRaisesRegex(GateFailure,"six original"):
            summarize_sequences(fake_sequences()[:-1])

    def test_18_wrong_source_duplicate_domain_fail_closed(self):
        rows=fake_sequences()
        rows[0]["kind"]=rows[1]["kind"]
        with self.assertRaisesRegex(GateFailure,"six original"):
            summarize_sequences(rows)

    def test_19_only_readonly_old_source_diagnostic_never_PROMOTE(self):
        self.assertIn("FIRST_DIVERGENCE",STAGE)
        self.assertEqual(SOURCE_COUNT,6)
        self.assertEqual(VOCAB_SIZE,49152)


if __name__=="__main__":
    main()
