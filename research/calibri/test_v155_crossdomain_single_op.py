"""Only synthetic selection/holdout checks; don't touch actual new source."""
import json
import tempfile
import unittest
from pathlib import Path
from v154.protocol import MATRIX_KEYS,CB153_SHA
from v155.protocol import (ARM_NAMES,DOMAINS,WIKI_START,NEWS_START,NEWS_END,
                            TOKENS,check,precommit_selection)
from v155.selection import holdout_gate,frozen_arms
from v153.codec import Physical,Patch
import numpy as np


class PreRegisteredPaidOperatorValidation(unittest.TestCase):
    def test_fresh_source_disjoint_from_prior_153_valid(self):
        self.assertTrue(check())
        self.assertEqual(WIKI_START,45056)
        self.assertEqual((NEWS_START,NEWS_END),(13200,13300))
        self.assertEqual(TOKENS,512)
        self.assertEqual(ARM_NAMES,("CB59_original","CB153_all_seven",
                                    "v154_best_single","v154_worst_single"))
        self.assertEqual(DOMAINS,("wiki_new_single_op_valid","news_new_single_op_valid"))

    def test_frozen_selection_never_uses_new_valid(self):
        ranking=[{
            "original_operator":key,
            "single_operator_KL_vs_CB59":(i-3)*.2,
        } for i,key in enumerate(MATRIX_KEYS)]
        old={
            "stage":"v154_ORIGINAL_SOURCE_LAYER6_SEVEN_OPERATOR_ABLATIONS",
            "fully_serialized_CB153_SHA256":CB153_SHA,
            "complete_original_CB59_SHA256":
              "50eac24ba7d7a452c495f8adced2cdef3f9cc44e024217fc21d8f2cebb87413c",
            "diagnostic":{"per_original_operator_diagnostic":ranking},
        }
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/"locked_v154_old_diagnostic.json"
            p.write_text(json.dumps(old))
            chosen=precommit_selection(p)
            self.assertEqual(chosen["best"],MATRIX_KEYS[0])
            self.assertEqual(chosen["worst"],MATRIX_KEYS[-1])
            self.assertEqual(chosen["all_diagnostic_operators_in_selection"],7)
            old["diagnostic"]["per_original_operator_diagnostic"].pop()
            p.write_text(json.dumps(old))
            with self.assertRaises(ValueError):
                precommit_selection(p)

    def test_all_stored_patches_remain_paid_despite_single_operator_runtime(self):
        patches={
            name:Patch(name,(64,64),
                       np.zeros((32,2,8),dtype=np.float32),
                       np.zeros(64,dtype=np.uint8))
            for name in MATRIX_KEYS}
        physical=Physical(b"immutable complete",6,patches,11292174,CB153_SHA)
        chosen={"best":MATRIX_KEYS[0],"worst":MATRIX_KEYS[-1]}
        arms=frozen_arms(physical,chosen)
        self.assertEqual(tuple(arms),ARM_NAMES)
        self.assertEqual(len(arms[ARM_NAMES[0]].patches),0)
        self.assertEqual(len(arms[ARM_NAMES[1]].patches),7)
        self.assertEqual(len(arms[ARM_NAMES[2]].patches),1)
        self.assertEqual(len(arms[ARM_NAMES[3]].patches),1)
        self.assertEqual(physical.size,11292174)
        self.assertEqual(len(physical.patches),7)

    def test_unseen_domain_regression_prevents_false_pass(self):
        def metric(nll,kl):
            return {"mean_TRUE_next_token_NLL":nll,
                    "teacher_to_candidate_KL":kl,
                    "teacher_next_token_argmax_agreement_fraction":0}
        data={
            arm:{domain:metric(15.,10.) for domain in DOMAINS}
            for arm in ARM_NAMES}
        for domain in DOMAINS:
            data["v154_best_single"][domain]=metric(14.,9.)
            data["v154_worst_single"][domain]=metric(15.5,11.)
        changes,passed=holdout_gate(data)
        self.assertTrue(passed)
        self.assertEqual(changes[DOMAINS[0]]["single_best_minus_CB59_NLL"],-1)
        data["v154_best_single"][DOMAINS[1]]=metric(16.,8.)
        changes,passed=holdout_gate(data)
        self.assertFalse(passed)
        data.pop("CB59_original")
        with self.assertRaises(ValueError):
            holdout_gate(data)


if __name__=="__main__":
    unittest.main()
