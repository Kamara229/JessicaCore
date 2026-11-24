"""Z46 immutable source-only prereg, no new model load / DEV scoring."""
import unittest
from collections import Counter
from research.calibri_nra3 import s119q2i8z46_fresh28_onpolicy_bpe2_preregister as s
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import KINDS

class TestZ46Fresh28Prereg(unittest.TestCase):
    def test_new_internal_dev28_precommit_and_old_train(self):
        out=s.verify()
        self.assertEqual(out["NEW_28_INTERNAL_DEV"],28)
        self.assertEqual(out["previous_84_legal_TRAIN"],84)
        self.assertTrue(out["source_disjoint_all_known_prior_spent_DEV_and_train"])
        self.assertEqual(len(s.rows("TRAIN")),84)
        self.assertEqual(len(s.rows("DEV")),28)
        self.assertEqual(Counter(x["kind"] for x in s.rows("DEV")),
                         {k:4 for k in KINDS})
        self.assertEqual(out["brand_NEW_28_internal_DEV_SHA256"],
                         s.digest(s.rows("DEV")))
        self.assertEqual(out["previous_84_legal_TRAIN_SHA256"],
                         s.digest(s.rows("TRAIN")))
    def test_exact_ownprefix_training_config_locked(self):
        cfg=s.verify()["frozen_physical_config"]
        self.assertEqual((cfg["rank"],cfg["shadow_params"]), (32,30720))
        self.assertEqual((cfg["train_epochs"],cfg["train_lr"],
                          cfg["train_weight_decay"]), (8,.004,.04))
        self.assertEqual((cfg["topk_native"],cfg["native_vocab"]), (64,49152))
        self.assertEqual(cfg["wrong_label_offset"],17)
        self.assertTrue(cfg["only_second_native_BPE_rerank"])
        self.assertTrue(cfg["original_first_and_third_untouched"])
    def test_never_teacher_force_dev_or_inject_gold(self):
        cfg=s.verify()["frozen_physical_config"]
        self.assertTrue(cfg["DEV_target_blind_candidate_selection"])
        self.assertTrue(cfg["DEV_own_prefix_no_teacher_forcing"])
        self.assertTrue(cfg["DEV_optimizer_steps_zero"])
        self.assertTrue(cfg["per_prompt_original_KL_and_entropy_guard"])
        self.assertEqual(cfg["eval_own_native_positions"],3)
    def test_diversity_and_exact_first2_3_precommitted_before_physical(self):
        x=s.verify()["frozen_physical_config"]["thresholds_nonpromoting"]
        self.assertEqual(x["human_first2_min"],4)
        self.assertEqual(x["human_minus_each_control_first2_min"],3)
        self.assertEqual(x["human_first3_min"],1)
        self.assertEqual(x["distinct_second_original_fraction_min"],.75)
        self.assertEqual(x["crossprompt_entropy_change_vs_original_min_bits"],-.25)
        self.assertEqual(x["premature_EOS_first3_max"],0)
    def test_new_data_not_entered_previous_holdouts_or_training(self):
        old=s.z40.previous_spent_and_train_prompts()+s.z40.rows("DEV")
        oldtexts={" ".join(x["prompt"].strip().casefold().split()) for x in old}
        curr={" ".join(x["prompt"].strip().casefold().split()) for x in s.rows("DEV")}
        self.assertFalse(curr.intersection(oldtexts))
        self.assertEqual(len(curr),28)
    def test_only_explicit_splits_and_no_promotion(self):
        with self.assertRaises(ValueError):s.rows("GOLD")
        with self.assertRaises(ValueError):s.rows("EVAL128")
        x=s.verify()
        self.assertFalse(x["candidate_accepted"])
        self.assertEqual(x["output_gate"],0)
        self.assertFalse(x["independent_external_blind_holdout"])
        self.assertTrue(x["becomes_SPENT_after_first_physical_DEV_evaluation"])

if __name__=="__main__":
    unittest.main()
