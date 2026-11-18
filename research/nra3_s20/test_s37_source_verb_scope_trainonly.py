"""S37 source-only tests: no original43M evidence, model never sees gold."""
from __future__ import annotations
import inspect
import unittest
import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    FEATURES,raw_surface_grammar,
)
from research.nra3_s19.s19g_paid_semantic_graph import GraphLogits
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    build_s22_train256,splits,
)
from research.nra3_s20.s26_predicted_event_inversion_rescue import (
    PREV_NEG,LIKELY_DATIVE,
)
from research.nra3_s20.s35_real43m_local_negation_scope_trainonly import (
    apply_local_ne_dative_scope,
)
from research.nra3_s20.s37_real43m_source_verb_scope_trainonly import (
    PAST,FUTURE,apply_raw_source_verb_scope,
    assert_only_ineligible_rejected_masked,S37_RULE,
)


class S37SourceVerbOnlyContracts(unittest.TestCase):
    def test_no_correct_source_blocked_and_no_false_conjunction_event_source(self):
        b,w=2,5
        raw=torch.zeros(b,w,338,dtype=torch.float32)
        valid=torch.ones(b,w,dtype=torch.bool)
        raw[:,3,320+PREV_NEG]=1.
        raw[:,3,320+LIKELY_DATIVE]=1.
        raw[0,1,320+PAST]=1.
        raw[1,1,320+FUTURE]=1.
        out=GraphLogits(
           torch.zeros(b,w,6),torch.zeros(b,w,w,9),
           torch.zeros(b,w,w,3),torch.zeros(b,w,w,2),
           torch.zeros(b,w,w,5))
        out.argument_polarity[...,0]=1.
        for i in (1,2):
            out.argument_polarity[:,i,3,2]=15.
        old=out.argument_polarity.clone()
        s35=apply_local_ne_dative_scope(out,raw,valid)
        s37=apply_raw_source_verb_scope(s35,raw,valid)
        self.assertEqual(int(s37.argument_polarity[0,1,3].argmax()),2)
        self.assertEqual(int(s37.argument_polarity[1,1,3].argmax()),2)
        self.assertNotEqual(int(s37.argument_polarity[0,2,3].argmax()),2)
        self.assertNotEqual(int(s37.argument_polarity[1,2,3].argmax()),2)
        self.assertTrue(assert_only_ineligible_rejected_masked(s35,s37,raw,valid))
        self.assertTrue(torch.equal(old,out.argument_polarity))
        for head in ("event_status","role","coreference","event_relation"):
            self.assertTrue(torch.equal(getattr(s35,head),getattr(s37,head)))

    def test_all_32_original_synthetic_rejected_event_source_cues(self):
        # This is a *TRAIN-only annotation alignment* audit, never inference.
        rows=build_s22_train256()
        self.assertEqual(len(rows),256)
        found=0
        for lexical in range(8):
            for construction in (24,25,26,27):
                g=rows[lexical*32+construction]
                events={x.event_id:x.predicate for x in g.events}
                tokens=raw_surface_grammar(g.text)
                for argument in g.arguments:
                    if argument.polarity!="REJECTED":
                        continue
                    found+=1
                    source_idx=events[argument.event_id]-1
                    self.assertTrue(
                       bool(tokens[source_idx].features[PAST] or
                            tokens[source_idx].features[FUTURE]),
                       (lexical,construction,g.text,tokens[source_idx].form))
        self.assertEqual(found,32)

    def test_no_annotation_or_specific_connector_blacklist_in_scope(self):
        code=inspect.getsource(apply_raw_source_verb_scope)
        for forbidden in ('gold[','holdout','held_ids','construction','PATTERNS',
                          'source_annotated','== "но"','"но" in','group_id'):
            self.assertNotIn(forbidden,code)
        self.assertEqual(S37_RULE,
                         "source_has_raw18_past_or_finite_future_verb_shape")
        for fold in (32,33,34,35,36,15,24,30):
            axis,group,train,held=splits()[fold]
            self.assertEqual((axis,group,len(train),len(held)),
                             ("CONSTRUCTION",fold-8,248,8))
            self.assertFalse(set(train).intersection(held))


if __name__=="__main__":
    unittest.main()
