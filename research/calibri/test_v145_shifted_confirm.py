"""No fresh dataset used by tests: verify preregistration before new source fetch."""
import unittest
import torch
from v145.protocol import (SOURCE_DOMAINS,ARMS,SELECTED,BASELINE,PAID,
    TOKEN_COUNT,WINDOW,WINDOWS,verify_predeclared,windows,disjoint)
from v145.audit import assemble


def record(nll,kl):
    return {"tokens_scored":63,"sum_NLL":nll*63,
            "sum_teacher_NLL":3.*63,
            "sum_teacher_to_student_KL":kl*63,
            "exact_FP32_next_token_argmax_count":0}


class NoTestLeakAndNoRetuning(unittest.TestCase):
    def test_new_pinned_train_domains_disjoint_old_valid_fit(self):
        self.assertTrue(verify_predeclared())
        self.assertEqual(SOURCE_DOMAINS,("wiki_shift","news_shift"))
        self.assertEqual(SELECTED,(0.,.5))
        self.assertEqual(ARMS,("input_0_head_0","input_0_head_50","input_100_head_100"))
        self.assertEqual((WINDOW,TOKEN_COUNT,WINDOWS),(64,512,8))
        self.assertTrue(disjoint((19456,19968),(14336,14592)))
        self.assertTrue(disjoint((9100,9200),(8500,8600)))
        self.assertFalse(disjoint((14336,14592),(14400,14500)))
        with self.assertRaises(ValueError):
            windows(torch.arange(256))

    def test_contrasts_require_only_same_locked_three_arms(self):
        data={arm:{domain:[record(15.,11.) for _ in range(8)]
                    for domain in SOURCE_DOMAINS} for arm in ARMS}
        for domain in SOURCE_DOMAINS:
            data[ARMS[1]][domain]=[record(14.8,10.8) for _ in range(8)]
        result=assemble(data)
        self.assertTrue(result["v144_preselected_passes_both_shifted_domains"])
        self.assertFalse(result["full_v140_passes_both_shifted_domains"])
        self.assertEqual(result["new_train_split_text_heldout"]["wiki_shift"]
                         ["original_CB59"]["tokens_scored"],504)

    def test_domain_specific_regression_not_masked(self):
        data={arm:{domain:[record(15.,11.) for _ in range(8)]
                    for domain in SOURCE_DOMAINS} for arm in ARMS}
        data[ARMS[1]]["wiki_shift"]=[record(16.,10.) for _ in range(8)]
        data[ARMS[1]]["news_shift"]=[record(14.,10.) for _ in range(8)]
        result=assemble(data)
        self.assertFalse(result["v144_preselected_passes_both_shifted_domains"])
        data[ARMS[2]]["news_shift"]=[]
        with self.assertRaises(ValueError):
            assemble(data)


if __name__=="__main__":
    unittest.main()
