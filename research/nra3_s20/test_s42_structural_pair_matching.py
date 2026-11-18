"""S42 isolated greedy directed-pair structural consistency invariants.

Artificial logits here prove only decoder contract; the separate
GitHub Actions job runs physical S40 CPU TRAIN80 shadow for 18 folds.
"""
from __future__ import annotations

import inspect
import unittest
import torch
from research.nra3_s20.s39_shadow_nra2_negation_scope_adapter import GraphPrediction
from research.nra3_s20.s39_new80_trainonly_scope_graph_source import validate_source
from research.nra3_s20.s42_sparse_predicate_recipient_matching import (
    matching_links_from_s40,verify_readonly_matching,
    S42_LINK_FLOOR_LOGIT,S42_RULE,
)

class S42SyntheticTrainOnlyDirectedSparseGraph(unittest.TestCase):
    def test_strict_source_target_matching_on_ambiguous_directed_edges(self):
        b,w=2,5
        relation=torch.full((b,w,w),-4.0)
        structural=torch.zeros((b,w,w,12))
        events=torch.randn(b,w)
        quote=torch.randn(b,w)
        # Source 1 may reject recipient 3 or 4; source 2 may also
        # reject recipient 3. Structural candidates are equally legal.
        entries={(1,3):2.0,(1,4):1.5,(2,3):1.0,(2,4):1.4}
        for i,j in entries:
            structural[:,i,j,11]=1.
            relation[:,i,j]=entries[i,j]
        old=relation.clone()
        logits=GraphPrediction(events,quote,relation)
        out,mask=matching_links_from_s40(logits,structural)
        self.assertTrue(verify_readonly_matching(logits,out,mask,structural))
        self.assertEqual(int(mask.sum()),4)
        self.assertTrue(bool(mask[:,1,3].all()))
        self.assertTrue(bool(mask[:,2,4].all()))
        self.assertFalse(bool(mask[:,1,4].any()))
        self.assertFalse(bool(mask[:,2,3].any()))
        self.assertTrue(torch.equal(relation,old))
        self.assertTrue(torch.equal(out.event_logits,events))
        self.assertTrue(torch.equal(out.quote_logits,quote))

    def test_one_candidate_minimum_floor_only_with_no_model_weight_updates(self):
        b,w=1,3
        relation=torch.full((b,w,w),-20.)
        structural=torch.zeros(b,w,w,12)
        structural[0,1,2,11]=1.
        logits=GraphPrediction(torch.zeros(b,w),torch.zeros(b,w),relation)
        out,selected=matching_links_from_s40(logits,structural)
        self.assertTrue(verify_readonly_matching(logits,out,selected,structural))
        self.assertEqual(float(out.rejected_logits[0,1,2]),S42_LINK_FLOOR_LOGIT)
        self.assertTrue(bool((out.rejected_logits[~selected]==-1e6).all()))
        self.assertEqual(int(selected.sum()),1)

    def test_no_gold_held_or_named_construction_in_inference(self):
        src=inspect.getsource(matching_links_from_s40)
        for bad in ("gold[","train_labels","source_predicate_token_index0",
                    "source_lexical_family","PATTERNS[","held_ids",
                    "baseline_s41_report","passive_shape","group_id"):
            self.assertNotIn(bad,src)
        self.assertEqual(S42_RULE,
            "one_source_one_target_directed_structural_greedy_matching")
        self.assertEqual(S42_LINK_FLOOR_LOGIT,.25)
        self.assertEqual(validate_source()["TRAIN_rows"],80)

if __name__=="__main__":
    unittest.main()
