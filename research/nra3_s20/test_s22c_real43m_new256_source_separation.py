"""S22c source-only tensor ABI checks: FAKE hidden in unit tests, NEVER quality."""
from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

import torch

from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s19.s19f_real43m_train9_graph_targets import STATUS, EVENT_RELATION
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    PATTERNS,FAMILY_COUNT,PATTERN_COUNT,build_s22_train256,text_only_inputs,
)
from research.nra3_s20.s22c_real43m_new256_graph_target_join import (
    N,build_features,build_train_targets,EXPECTED_COUNTS,
)


def fake_geometry_for_tests_only():
    texts=text_only_inputs()
    counts=[len(raw_model_input(t).tokens) for t in texts]
    width=max(counts)
    h=torch.zeros(N,width,320,dtype=torch.float16)
    m={"wordpiece_alignment":[],"true_train256_token_counts":counts}
    for i,n in enumerate(counts):
        for j in range(n):
            h[i,j,:]=float(i+1)/1000+float(j)/10000
        m["wordpiece_alignment"].append({
            "word_last_token_positions":list(range(n)),
            "word_token_positions":[[j] for j in range(n)],
        })
    return texts,h,m


class S22cGoldSeparateFromActualInputContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text,cls.hidden,cls.meta=fake_geometry_for_tests_only()

    def test_256_source_raw_features_have_338_dims_without_gold(self):
        raw=build_features(self.text,self.hidden,self.meta)
        self.assertEqual((N,EXPECTED_COUNTS),(256,(448,976,32,192,192)))
        self.assertEqual(raw.raw_word_hidden320_and_raw18.shape[0],256)
        self.assertEqual(raw.raw_word_hidden320_and_raw18.shape[2],338)
        self.assertFalse(hasattr(raw,"event_status"))
        self.assertFalse(hasattr(raw,"predicate_argument_role"))
        self.assertNotIn("gold",inspect.signature(build_features).parameters)
        self.assertTrue(bool((raw.raw_word_hidden320_and_raw18[~raw.valid_word_mask]==0).all()))
        self.assertTrue(bool((((raw.raw_word_hidden320_and_raw18[...,320:]==0)|
                                  (raw.raw_word_hidden320_and_raw18[...,320:]==1))).all()))

    def test_new256_gold_counted_five_heads_not_forward_features(self):
        g=build_train_targets(build_features(self.text,self.hidden,self.meta))
        counts=tuple(int(z) for z in (
            (g.event_status>0).sum(),
            (g.predicate_argument_role>0).sum(),
            (g.argument_polarity==1).sum(),
            (g.coreference==1).sum(),
            (g.event_links>0).sum(),
        ))
        self.assertEqual(counts,(448,976,32,192,192))
        self.assertTrue(bool((g.predicate_argument_role.diagonal(dim1=1,dim2=2)==-1).all()))

    def test_conditional_direction_and_rejected_scope_separate_from_event_negation(self):
        graphs=build_s22_train256()
        labels=build_train_targets(build_features(self.text,self.hidden,self.meta))
        for fam in range(FAMILY_COUNT):
            first=fam*PATTERN_COUNT
            reversed_row=graphs[first+PATTERNS.index("conditional_result_first")]
            anchors=[e.predicate-1 for e in reversed_row.events]
            self.assertEqual(labels.event_links[first+PATTERNS.index("conditional_result_first"),
                                                anchors[1],anchors[0]].item(),
                             EVENT_RELATION.index("CONDITION_RESULT"))
            for pat in ("rejected_not_recipient_but","rejected_positive_first"):
                j=first+PATTERNS.index(pat)
                event_idx=graphs[j].events[0].predicate-1
                self.assertEqual(labels.event_status[j,event_idx].item(),
                                 STATUS.index("ASSERTED"))
                self.assertEqual(int((labels.argument_polarity[j]==1).sum()),1)
            j=first+PATTERNS.index("negated_male_past")
            event_idx=graphs[j].events[0].predicate-1
            self.assertEqual(labels.event_status[j,event_idx].item(),
                             STATUS.index("NEGATED"))
            self.assertEqual(int((labels.argument_polarity[j]==1).sum()),0)

    def test_never_accept_mismatched_plaintexts_or_duplicate_bpe(self):
        with self.assertRaises(ValueError):
            build_features(self.text[:-1]+("new fake",),self.hidden,self.meta)
        bad={"wordpiece_alignment":[dict(x) for x in self.meta["wordpiece_alignment"]],
             "true_train256_token_counts":self.meta["true_train256_token_counts"]}
        bad["wordpiece_alignment"][0]["word_token_positions"]=[
            list(x) for x in bad["wordpiece_alignment"][0]["word_token_positions"]]
        bad["wordpiece_alignment"][0]["word_token_positions"][1]=[0]
        with self.assertRaises(ValueError):
            build_features(self.text,self.hidden,bad)
        inputs=build_features(self.text,self.hidden,self.meta)
        with self.assertRaises(ValueError):
            build_train_targets(replace(inputs,raw_texts=inputs.raw_texts[::-1]))


if __name__=="__main__":
    unittest.main()
