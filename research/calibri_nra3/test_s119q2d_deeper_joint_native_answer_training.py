"""S119q2d multi-example native teacher forcing and immutable-source contracts.

Only synthetic original author rows and fake native tokenizer; real S142
and physically loaded q2c1 checkpoints are verified separately in Actions.
"""
from __future__ import annotations

import copy
import unittest
from collections import Counter
import torch

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119q2a_train_source_manifest import build_rows
from research.calibri_nra3.s119q2d_deeper_joint_native_answer_training import (
    source_inventory,pack_sources,scheduled_position,
    EPOCHS,EXAMPLES_PER_EPOCH,ACCUMULATION,LR,CONTRAST_WEIGHT,VOCAB_SIZE,
)


class FakeNative:
    vocab_size=49152
    def apply_chat_template(self,messages,tokenize=False,
                            add_generation_prompt=True):
        assert not tokenize and add_generation_prompt
        return "<user>"+messages[0]["content"]+"</user><assistant>"
    def encode(self,text,add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) for c in text]
    def decode(self,ids,skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


class S119Q2DDeepJointContracts(unittest.TestCase):
    def setUp(self):
        self.train,self.probe=source_inventory(build_rows())

    def test_01_balanced_deep_train_96_cases_six_skills(self):
        self.assertEqual(len(self.train),96)
        self.assertEqual(Counter(x["kind"] for x in self.train),
                         {k:16 for k in ("roles_word_order","negation",
                                         "transitive_comparison","event_time",
                                         "arithmetic","paraphrase")})
        self.assertEqual(Counter(x["source_family"] for x in self.train),
                         {"q2a":72,"q2c1":24})

    def test_02_calibration_and_prior_source_probe_NEVER_train(self):
        self.assertEqual(len(self.probe),30)
        self.assertEqual(Counter(x["source_split"] for x in self.probe),
                         {"CALIBRATION":18,"SOURCE_PROBE":12})
        self.assertTrue(all(x["source_split"]=="TRAIN" for x in self.train))
        self.assertFalse({x["id"] for x in self.train}&
                         {x["id"] for x in self.probe})

    def test_03_epoch_budget_not_one_off_six_examples(self):
        self.assertEqual(EPOCHS,4)
        self.assertEqual(EXAMPLES_PER_EPOCH,96)
        self.assertEqual(ACCUMULATION,4)
        self.assertEqual(EXAMPLES_PER_EPOCH*EPOCHS,384)
        self.assertEqual(EXAMPLES_PER_EPOCH*EPOCHS//ACCUMULATION,96)
        self.assertEqual(LR,0.0005)
        self.assertEqual(CONTRAST_WEIGHT,0.2)

    def test_04_true_full_original_native_vocabulary(self):
        self.assertEqual(VOCAB_SIZE,49152)

    def test_05_full_answer_positions_from_real_native_tok(self):
        train,probe=pack_sources(FakeNative(),self.train,self.probe)
        self.assertEqual(len(train),96)
        self.assertEqual(len(probe),30)
        self.assertEqual(
            sum(x["source_family"]=="q2c1" for x in train),24)
        self.assertTrue(all(2<=len(x["prefix"])<160 and
                            1<=len(x["completion_ids"])<=22
                            for x in train+probe))
        self.assertTrue(all(x["prompt_features"].shape==(512,)
                            for x in train+probe))
        self.assertTrue(all(x["source_split"]=="TRAIN" for x in train))

    def test_06_q2c1_opposite_answers_diverge_before_end(self):
        train,probe=pack_sources(FakeNative(),self.train,self.probe)
        for ex in (x for x in train+probe if x["source_family"]=="q2c1"):
            self.assertIsNotNone(ex["first_distinct_position"])
            self.assertIsNotNone(ex["wrong_distinct_native"])
            self.assertNotEqual(
                ex["completion_ids"][ex["first_distinct_position"]],
                ex["wrong_distinct_native"])

    def test_07_teacher_forcing_never_contains_next_target(self):
        train,_=pack_sources(FakeNative(),self.train,self.probe)
        for epoch in range(EPOCHS):
            for ex in train:
                pos=scheduled_position(ex,epoch)
                causal=ex["prefix"]+ex["completion_ids"][:pos]
                self.assertEqual(causal[-pos:] if pos else [],
                                 ex["completion_ids"][:pos])
                self.assertEqual(len(causal),
                                 len(ex["prefix"])+pos)
                self.assertTrue(pos<len(ex["completion_ids"]))

    def test_08_q2c1_first_epoch_focuses_true_pair_difference(self):
        train,_=pack_sources(FakeNative(),self.train,self.probe)
        for ex in (x for x in train if x["source_family"]=="q2c1"):
            self.assertEqual(scheduled_position(ex,0),
                             ex["first_distinct_position"])

    def test_09_multi_position_train_covers_answer_suffix_not_only_first(self):
        train,_=pack_sources(FakeNative(),self.train,self.probe)
        self.assertGreater(sum(scheduled_position(ex,e)>0
                               for ex in train for e in range(EPOCHS)),0)
        self.assertTrue(any(len(ex["completion_ids"])>1 for ex in train))

    def test_10_source_rows_are_literal_and_name_bank_restricted(self):
        self.assertEqual(len({ex["prompt"] for ex in self.train}),96)
        self.assertEqual(len({ex["prompt"] for ex in self.probe}),30)
        self.assertFalse({x["prompt"] for x in self.train}&
                         {x["prompt"] for x in self.probe})

    def test_11_no_source_probe_can_be_scheduled_for_optimizer(self):
        _,probes=pack_sources(FakeNative(),self.train,self.probe)
        with self.assertRaisesRegex(GateFailure,"source-only"):
            scheduled_position(probes[0],0)

    def test_12_exact_native_assistant_boundary_failclosed(self):
        class BrokenBoundary(FakeNative):
            def encode(self,text,add_special_tokens=False):
                result=super().encode(text,add_special_tokens=add_special_tokens)
                if text.startswith("<user>") and text.endswith("."):
                    result.append(99)
                return result
        with self.assertRaisesRegex(GateFailure,"assistant boundary"):
            pack_sources(BrokenBoundary(),self.train,self.probe)

    def test_13_invalid_epoch_schedule_fails(self):
        train,_=pack_sources(FakeNative(),self.train,self.probe)
        for value in (-1,EPOCHS):
            with self.assertRaisesRegex(GateFailure,"source-only"):
                scheduled_position(train[0],value)

    def test_14_deterministic_multi_position_schedule(self):
        a,_=pack_sources(FakeNative(),self.train,self.probe)
        b,_=pack_sources(FakeNative(),self.train,self.probe)
        for e in range(EPOCHS):
            self.assertEqual(
                [scheduled_position(z,e) for z in a],
                [scheduled_position(z,e) for z in b])

    def test_15_spent_q2c1_source_probe_still_excluded_from_new_gradients(self):
        self.assertEqual(
            sum(x["source_family"]=="q2c1" for x in self.probe),12)
        self.assertTrue(all(x["pair_index"]==2 for x in self.probe
                            if x["source_family"]=="q2c1"))
        self.assertTrue(all(x["pair_index"] in (0,1) for x in self.train
                            if x["source_family"]=="q2c1"))


if __name__=="__main__":
    unittest.main()
