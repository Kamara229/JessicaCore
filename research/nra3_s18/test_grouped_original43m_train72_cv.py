"""S18a groups, no unbilled params or old independent-EVAL access."""
from __future__ import annotations
import hashlib
import unittest
import torch

from research.nra3_s9.new_ru_train72 import build_s9_train72,TEMPLATES
from research.nra3_s16.paid_directed_valence import PaidDirectedValence
from research.nra3_s16.real43m_train72_directed_valence import DirectedPairTrain
from research.nra3_s18.grouped_original43m_train72_cv import (
    controlled_group_folds,select_pairs,STEPS,LEXICAL_FOLDS,SYNTAX_FOLDS,
    score_rounded,PHYSICAL_BYTES,PARAMETERS
)


class OldTRAIN72StrictGroupFolds(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_each_axis_partitions_all72_original_TRAIN_exactly_once_without_overlap(self):
        rows=build_s9_train72()
        folds=controlled_group_folds(rows)
        self.assertEqual(len(folds),10)
        self.assertEqual(len(LEXICAL_FOLDS),4)
        self.assertEqual(len(SYNTAX_FOLDS),6)
        self.assertEqual(STEPS,150)
        for f in folds:
            train,held=set(f["train_rows"]),set(f["holdout_rows"])
            self.assertFalse(train & held)
            self.assertEqual(train | held,set(range(72)))
            self.assertGreater(len(held),0)
            if f["kind"]=="TRAIN_INTERNAL_LEXICAL":
                self.assertEqual(len(train),54)
                self.assertEqual(len(held),18)
                self.assertEqual({rows[i].lexical_family for i in held},
                                 set(f["heldout_groups"]))
                self.assertFalse({rows[i].lexical_family for i in held}
                            & {rows[i].lexical_family for i in train})
                self.assertEqual({rows[i].construction for i in held},set(TEMPLATES))
            else:
                self.assertEqual(len(train),60)
                self.assertEqual(len(held),12)
                self.assertEqual({rows[i].construction for i in held},
                                 set(f["heldout_groups"]))
                self.assertFalse({rows[i].construction for i in held}
                            & {rows[i].construction for i in train})
                self.assertEqual({rows[i].lexical_family for i in held},set(range(12)))
        for sub in (folds[:4],folds[4:]):
            hits=[i for f in sub for i in f["holdout_rows"]]
            self.assertEqual(len(hits),72)
            self.assertEqual(len(set(hits)),72)

    def test_toy_real_candidate_pair_slicing_is_only_source_id_not_gold(self):
        n=72
        candidates=DirectedPairTrain(
          torch.randn(n,128),torch.randn(n,128),
          torch.arange(n,dtype=torch.long)+5,
          torch.arange(n,dtype=torch.long),
          torch.arange(n,dtype=torch.long)+5,
          torch.tensor([i%5 for i in range(n)],dtype=torch.long),
          torch.tensor([i%2 for i in range(n)],dtype=torch.long),
          torch.arange(n,dtype=torch.long),72,57
        )
        for fold in controlled_group_folds():
            train=select_pairs(candidates,fold["train_rows"])
            held=select_pairs(candidates,fold["holdout_rows"])
            self.assertEqual(train.rows,len(fold["train_rows"]))
            self.assertEqual(held.rows,len(fold["holdout_rows"]))
            self.assertEqual(train.total_pairs+held.total_pairs,72)
            self.assertEqual(
                set(train.source_index.tolist())&set(held.source_index.tolist()),
                set())
            self.assertEqual(train.positive_links+held.positive_links,
                             int((candidates.target!=0).sum()))
        with self.assertRaises(ValueError):
            select_pairs(candidates,(0,0,1))
        with self.assertRaises(ValueError):
            select_pairs(candidates,(72,))

    def test_physically_rounded_S16b_2309_FP16_loss_and_metric_no_validation_update(self):
        m=PaidDirectedValence()
        N=12
        x=DirectedPairTrain(
            torch.randn(N,128),torch.randn(N,128),
            torch.arange(N)+11,torch.arange(N),
            torch.arange(N)+11,
            torch.arange(N,dtype=torch.long)%5,
            torch.arange(N,dtype=torch.long)%2,
            torch.zeros(N,dtype=torch.long),1,9)
        p=m.physical_bytes()
        before,physical=score_rounded(m,x)
        after,physical2=score_rounded(m,x)
        self.assertEqual(physical,p)
        self.assertEqual(physical2,p)
        self.assertEqual(PARAMETERS,2309)
        self.assertEqual(PHYSICAL_BYTES,4665)
        self.assertEqual(before,after)
        self.assertEqual(before["all_two_direction_wordpair_exact_role"]["gold_links"],9)
        self.assertEqual(hashlib.sha256(physical).hexdigest(),
                         hashlib.sha256(p).hexdigest())

if __name__=="__main__":
    unittest.main()
