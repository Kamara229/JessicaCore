"""S17c safeguards: only synthetic LOGIT testing, never true EVAL model."""
from __future__ import annotations
import unittest

import torch

from research.nra3_s16.paid_directed_valence import ROLES
from research.nra3_s17.new_ru24_directed_eval_prereg import (
    S17_EVAL24_COMPLETE_SHA,build_new_eval24,gold_links
)
from research.nra3_s17.one_time_true43m_new24_directed_eval import (
    S17B_FROZEN_ORIGINAL43M_CPU_HIDDEN_SHA,
    S17_GOLD_SHA,S16B_FROZEN_MODEL_SHA,
    calc_exact_directed_role_metrics,original_cpu_new24_source
)


def _simulate(pred_class):
    """Exclusively hand-created unit tensor; not Calibri/NRA predictions."""
    rows=build_new_eval24()
    samples=[]
    for row in rows:
        n=len(row.words)
        pairs=tuple((p,a) for p in range(1,n+1)
                    for a in range(1,n+1) if p!=a)
        gold={(p,a):ROLES.index(z) for p,a,z in gold_links(row)}
        fake=torch.full((len(pairs),len(ROLES)),-5.)
        for i,(p,a) in enumerate(pairs):
            fake[i,pred_class(gold.get((p,a),0),i,p,a)]=5.
        samples.append((row,pairs,fake))
    return samples


class OneTimeNewEval24SourceOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_strict_gold_sha_precommitted_and_real_hidden_source_fail_closed(self):
        self.assertEqual(S17_GOLD_SHA,S17_EVAL24_COMPLETE_SHA)
        self.assertEqual(S17_GOLD_SHA,
           "fca54736aa555d9c5b3665d1bb9a1fc1979ad7abdeb083af015d90dc05ed1a32")
        self.assertEqual(S16B_FROZEN_MODEL_SHA,
           "3fe4bbce718e53e4e812e94367ce25834f873cbed646479c59b1e3949d2ecf38")
        # Must be explicitly SHA-locked after S17b CI, not a dummy source.
        if not S17B_FROZEN_ORIGINAL43M_CPU_HIDDEN_SHA:
            with self.assertRaises(ValueError):
                original_cpu_new24_source(None,None)

    def test_synthetic_perfect_all_pair_role_predictions_one_f1(self):
        fake=_simulate(lambda gold,*_:gold)
        actual=calc_exact_directed_role_metrics(fake)
        result=actual["S17c_all24_frozen_directed_role_micro"]
        self.assertEqual(result["tp"],74)
        self.assertEqual(result["fp"],0)
        self.assertEqual(result["fn"],0)
        self.assertEqual(result["f1"],1.)
        self.assertEqual(len(actual["S17c_per_sentence_readonly_diagnostic"]),24)
        self.assertTrue(all(v["f1"]==1. for v in actual["S17c_per_role"].values()))

    def test_synthetic_always_NONE_has_zero_directed_link_F1(self):
        fake=_simulate(lambda gold,*_:0)
        result=calc_exact_directed_role_metrics(fake)["S17c_all24_frozen_directed_role_micro"]
        self.assertEqual(result["tp"],0)
        self.assertEqual(result["fp"],0)
        self.assertEqual(result["fn"],74)
        self.assertEqual(result["f1"],0.)

    def test_synthetic_wrong_positive_role_counts_FP_and_FN(self):
        def wrong(gold,*_):
            return 2 if gold==1 else 0
        fake=_simulate(wrong)
        result=calc_exact_directed_role_metrics(fake)["S17c_all24_frozen_directed_role_micro"]
        self.assertEqual(result["tp"],0)
        self.assertGreater(result["fp"],0)
        self.assertEqual(result["fn"],74)
        self.assertEqual(result["f1"],0.)

    def test_fail_closed_if_a_new_sentence_is_filtered_or_scored_twice(self):
        full=_simulate(lambda gold,*_:gold)
        with self.assertRaises(ValueError):
            calc_exact_directed_role_metrics(full[:-1])
        broken=list(full)
        row,pairs,logits=broken[0]
        broken[0]=(row,pairs[:-1],logits[:-1])
        with self.assertRaises(ValueError):
            calc_exact_directed_role_metrics(broken)


if __name__=="__main__":
    unittest.main()
