"""Tiny deterministic negative/positive selection fixtures; no real TEST leakage."""
import unittest
import torch
from v144.protocol import (GAIN_GRID,GAIN_VALUES,SELECT_WINDOW_IDS,
    CONFIRM_WINDOW_IDS,BASELINE,ORIGINAL_PAID,fixed_input_windows,label)
from v144.selection import choose,NLL_MIN_GAIN_PER_DOMAIN
from v144.operators import check_gain
from v142.protocol import DOMAINS


def sample(nll,kl,tokens=20):
    return {"tokens_scored":tokens,
            "sum_NLL":nll*tokens,
            "sum_teacher_NLL":2.*tokens,
            "sum_teacher_to_student_KL":kl*tokens,
            "exact_FP32_next_token_argmax_count":0}


class CounterfactualGainValidation(unittest.TestCase):
    def test_locked_grid_and_nonoverlap(self):
        self.assertEqual(len(GAIN_GRID),25)
        self.assertEqual(len(set(GAIN_GRID)),25)
        self.assertFalse(set(SELECT_WINDOW_IDS)&set(CONFIRM_WINDOW_IDS))
        self.assertEqual(len(fixed_input_windows(torch.arange(256))),4)
        self.assertEqual(BASELINE,label(0,0))
        self.assertEqual(ORIGINAL_PAID,label(1,1))
        with self.assertRaises(ValueError):
            fixed_input_windows(torch.arange(128))
        with self.assertRaises(ValueError):
            check_gain(-1)
        with self.assertRaises(ValueError):
            check_gain(.37)

    def test_no_robust_candidate_defaults_to_original_CB59(self):
        table={label(i,h):{d:[sample(12.,8.)] for d in DOMAINS}
               for i,h in GAIN_GRID}
        table[ORIGINAL_PAID]["wiki_valid"]=[sample(15.,10.)]
        table[ORIGINAL_PAID]["news_valid"]=[sample(9.,7.)]
        selected,report=choose(table)
        self.assertEqual(selected,BASELINE)
        self.assertTrue(report["no_robust_both_domain_candidate"])

    def test_both_domains_better_nll_and_kl_required(self):
        table={label(i,h):{d:[sample(12.,8.)] for d in DOMAINS}
               for i,h in GAIN_GRID}
        viable=label(.25,.5)
        for domain in DOMAINS:
            table[viable][domain]=[sample(11.8,7.9)]
        table[label(1.,0.)]["news_valid"]=[sample(11.,6.)]
        table[label(1.,0.)]["wiki_valid"]=[sample(11.,9.)] # KL loss
        selected,report=choose(table)
        self.assertEqual(selected,viable)
        self.assertFalse(report["no_robust_both_domain_candidate"])
        self.assertGreater(report["worst_domain_NLL_gain"],NLL_MIN_GAIN_PER_DOMAIN)

    def test_missing_arm_or_domain_is_fatal(self):
        table={label(i,h):{d:[sample(12.,8.)] for d in DOMAINS}
               for i,h in GAIN_GRID}
        table.pop(ORIGINAL_PAID)
        with self.assertRaises(ValueError):
            choose(table)
        table[ORIGINAL_PAID]={"wiki_valid":[sample(12.,8.)]}
        with self.assertRaises(KeyError):
            choose(table)


if __name__=="__main__":
    unittest.main()
