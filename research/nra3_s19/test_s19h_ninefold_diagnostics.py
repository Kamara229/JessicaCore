"""S19h LOOCV arithmetic and split tests, never independent evaluation."""
from __future__ import annotations

import unittest

import torch

from research.nra3_s19.s19h_true43m_train9_loo_diagnostics import (
    take,merge_counts,LOO_FOLDS,metric_from_train_scoring_alias
)
from research.nra3_s19.test_s19g_paid_event_graph import fake_geometry_only,empty_shadow
from research.nra3_s19.s19g_train9_real43m_graph import pair_targets


class TrainOnlyNineFoldContracts(unittest.TestCase):
    def test_train_and_held_indices_partition_nine_without_overlap(self):
        self.assertEqual(LOO_FOLDS,9)
        data,y=fake_geometry_only()
        x=data.raw_word_hidden320_and_raw18
        for held in range(9):
            train=tuple(i for i in range(9) if i!=held)
            self.assertEqual(len(train),8)
            self.assertNotIn(held,train)
            self.assertEqual(set(train)|{held},set(range(9)))
            self.assertEqual(take(x,train).shape[0],8)
            self.assertEqual(take(x,(held,)).shape[0],1)
            self.assertTrue(torch.equal(take(x,(held,))[0],x[held]))

    def test_gold_only_given_to_loss_and_not_forward(self):
        data,y=fake_geometry_only()
        model=empty_shadow()
        logits=model(data.raw_word_hidden320_and_raw18,
                     data.valid_word_mask)
        targets={
          "event_status_TRAIN_ONLY":y.event_status,
          "predicate_argument_role_TRAIN_ONLY":y.predicate_argument_role,
          "argument_polarity_TRAIN_ONLY":y.argument_polarity,
          "coreference_TRAIN_ONLY":y.coreference,
          "event_relation_TRAIN_ONLY":y.event_links,
        }
        result=metric_from_train_scoring_alias(
            logits,targets,data.valid_word_mask)
        self.assertEqual(set(result),{"event","directed_role","argument_polarity",
                                      "coreference","event_relation"})
        self.assertEqual(result["event"]["held_TRAIN_gold_nonNONE"],15)
        self.assertEqual(result["argument_polarity"]["held_TRAIN_gold_nonNONE"],40)
        self.assertEqual(int((pair_targets(targets)==2).sum()),1)

    def test_aggregation_uses_sum_of_counts_not_average_of_f1(self):
        def group(tp,fp,fn):
            return {k:{"TP":tp,"FP":fp,"FN":fn,
                       "held_TRAIN_gold_nonNONE":tp+fn}
                    for k in ("event","directed_role","argument_polarity",
                              "coreference","event_relation")}
        rows=[
          {"per_head_heldout_TRAIN_only":group(1,0,0)},
          {"per_head_heldout_TRAIN_only":group(0,2,1)}
        ]
        a=merge_counts(rows)
        self.assertEqual((a["event"]["TP"],a["event"]["FP"],
                          a["event"]["FN"]),(1,2,1))
        self.assertAlmostEqual(
            a["event"]["TRAIN_INTERNAL_LOO_micro_F1_NOT_independent"],2/5)


if __name__=="__main__":
    unittest.main()
