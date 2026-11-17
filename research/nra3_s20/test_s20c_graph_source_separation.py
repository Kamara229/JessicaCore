"""S20c source-only ABI tests with FAKE geometry, never real hidden."""
from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import torch

from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s19.s19f_real43m_train9_graph_targets import (
    STATUS, ROLE, EVENT_RELATION,
)
from research.nra3_s20.s20_new96_synthetic_graph_train_source import (
    raw_text_only_train_input,PATTERNS,FAMILY_COUNT,PATTERN_COUNT,
)
from research.nra3_s20.s20c_real43m_new96_graph_target_join import (
    N,build_features,build_train_targets,EXPECTED_COUNTS,
)


def fake_geometry_for_tests_only():
    texts=raw_text_only_train_input()
    counts=[len(raw_model_input(t).tokens) for t in texts]
    width=max(counts)
    h=torch.zeros(N,width,320,dtype=torch.float16)
    m={"wordpiece_alignment":[],"true_train96_token_counts":counts}
    for i,n in enumerate(counts):
        # Explicit geometry fixture: not claimed original physical Calibri.
        for j in range(n):
            h[i,j,:]=float(i+1)/1000+float(j)/10000
        m["wordpiece_alignment"].append({
            "word_last_token_positions":list(range(n)),
            "word_token_positions":[[j] for j in range(n)],
        })
    return texts,h,m


class S20cSourceOnlyGoldInputSeparation(unittest.TestCase):
    def test_ninety_six_original_dimension_source_only_no_gold(self):
        text,h,meta=fake_geometry_for_tests_only()
        raw=build_features(text,h,meta)
        self.assertEqual(raw.raw_word_hidden320_and_raw18.shape[0],96)
        self.assertEqual(raw.raw_word_hidden320_and_raw18.shape[2],338)
        self.assertEqual(int(raw.valid_word_mask.sum()),
                         sum(len(raw_model_input(t).tokens) for t in text))
        self.assertFalse(hasattr(raw,"event_status"))
        self.assertFalse(hasattr(raw,"predicate_argument_role"))
        self.assertFalse(hasattr(raw,"coreferences"))
        self.assertNotIn("gold",inspect.signature(build_features).parameters)
        self.assertTrue(torch.all(raw.raw_word_hidden320_and_raw18[~raw.valid_word_mask]==0))
        self.assertTrue(bool(((raw.raw_word_hidden320_and_raw18[...,320:]==0)|
                              (raw.raw_word_hidden320_and_raw18[...,320:]==1)).all()))

    def test_all_five_gold_graph_training_targets_separate_from_raw_inputs(self):
        texts,h,meta=fake_geometry_for_tests_only()
        raw=build_features(texts,h,meta)
        labels=build_train_targets(raw)
        self.assertEqual(tuple(int(z) for z in (
            (labels.event_status>0).sum(),
            (labels.predicate_argument_role>0).sum(),
            (labels.argument_polarity==1).sum(),
            (labels.coreference==1).sum(),
            (labels.event_links>0).sum())),EXPECTED_COUNTS)
        self.assertEqual(EXPECTED_COUNTS,(160,432,16,64,64))
        self.assertTrue(bool((labels.event_status[~raw.valid_word_mask]==-1).all()))
        self.assertTrue(bool((labels.predicate_argument_role.diagonal(dim1=1,dim2=2)==-1).all()))

    def test_negated_event_is_not_rejected_recipient_role(self):
        texts,h,meta=fake_geometry_for_tests_only()
        target=build_train_targets(build_features(texts,h,meta))
        for family in range(FAMILY_COUNT):
            rejected_i=family*PATTERN_COUNT+PATTERNS.index("rejected_recipient_preposed_time")
            row=raw_model_input(texts[rejected_i]).tokens
            anchor=next(t.index-1 for t in row if t.form in (
                "отправила","передала","показала","направила","переслала"))
            self.assertEqual(target.event_status[rejected_i,anchor].item(),
                             STATUS.index("ASSERTED"))
            self.assertEqual(int((target.argument_polarity[rejected_i]==1).sum()),1)
            negative_i=family*PATTERN_COUNT+PATTERNS.index("negated_inverted_word_order")
            self.assertEqual(int((target.event_status[negative_i]==STATUS.index("NEGATED")).sum()),1)
            self.assertEqual(int((target.argument_polarity[negative_i]==1).sum()),0)

    def test_reversed_condition_order_points_from_subordinate_to_consequence(self):
        texts,h,meta=fake_geometry_for_tests_only()
        gold=build_train_targets(build_features(texts,h,meta))
        for family in range(FAMILY_COUNT):
            index=family*PATTERN_COUNT+PATTERNS.index("conditional_if_subordinate_after_result")
            graph=raw_model_input(texts[index]).tokens
            verbs=[i for i in range(len(graph))
                   if gold.event_status[index,i]==STATUS.index("CONDITIONAL")]
            self.assertEqual(len(verbs),2)
            self.assertEqual(gold.event_links[index,verbs[1],verbs[0]].item(),
                             EVENT_RELATION.index("CONDITION_RESULT"))
            self.assertEqual(int((gold.coreference[index]==1).sum()),1)

    def test_mutated_plain_text_or_duplicate_native_wordpiece_fails_closed(self):
        texts,h,meta=fake_geometry_for_tests_only()
        with self.assertRaises(ValueError):
            build_features(texts[:-1]+("другой текст",),h,meta)
        bad={"wordpiece_alignment":[dict(z) for z in meta["wordpiece_alignment"]],
             "true_train96_token_counts":meta["true_train96_token_counts"]}
        bad["wordpiece_alignment"][0]["word_token_positions"]=[
            list(x) for x in bad["wordpiece_alignment"][0]["word_token_positions"]]
        bad["wordpiece_alignment"][0]["word_token_positions"][1]=[0]
        with self.assertRaises(ValueError):
            build_features(texts,h,bad)
        inputs=build_features(texts,h,meta)
        with self.assertRaises(ValueError):
            build_train_targets(replace(inputs,raw_texts=inputs.raw_texts[::-1]))


if __name__=="__main__":
    unittest.main()
