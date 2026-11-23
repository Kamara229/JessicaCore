"""Z23 freeze 28 new RU DEV, 4 scales and strictly no-target native decoder contract."""
import unittest
from collections import Counter
import torch
from research.calibri_nra3.s119q2i8z23_new28dev_frozen_future_calibration_source import (
    rows,verify,digest,KINDS,SCALES,TOP_K,BASE_FUTURE_FP16_SHA,BASE_ARTIFACT_ID,
)
from research.calibri_nra3.s119q2i8z23_targetblind_native_calibration_contract import (
    VOCAB,HIDDEN,STATES,TinyFutureState,target_blind_calibrated_candidates,
    posthoc_human_gold,
)

class TestZ23NewHoldoutAndFrozenCalibration(unittest.TestCase):
    def test_new_unseen_source_SHA_and_fixed_scales(self):
        source=verify()
        self.assertEqual(source["new_unseen_synthetic_DEV"],28)
        self.assertEqual(source["new_DEV_SHA256"],digest(rows("DEV")))
        self.assertTrue(source["disjoint_all_prior_spent_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16_Z18_Z21"])
        self.assertEqual(Counter(x["kind"] for x in rows()),{k:4 for k in KINDS})
        self.assertEqual((SCALES,TOP_K,BASE_ARTIFACT_ID),((0.,2.,8.,16.),64,14841506942))
        self.assertEqual(len(BASE_FUTURE_FP16_SHA),4)
        for split in ("TRAIN","J7","EVAL128","GOLD","TEST",""):
            with self.subTest(split=split),self.assertRaises(ValueError):
                rows(split)
    def test_targetblind_all_scale_choices_identical_baseline_zero(self):
        native=torch.full((1,VOCAB),-25.)
        native[0,11]=10;native[0,12]=9;native[0,2]=8
        emb=torch.randn((VOCAB,HIDDEN),generator=torch.Generator().manual_seed(56))
        h=torch.randn((1,HIDDEN),generator=torch.Generator().manual_seed(99))
        states={x:TinyFutureState().eval() for x in STATES}
        for state in states.values():
            state.requires_grad_(False)
        d=target_blind_calibrated_candidates(native,h,emb,states)
        self.assertEqual(d["original_NATIVE_GREEDY_BPE2_ID"],11)
        self.assertEqual(d["original_TARGET_BLIND_TOP64_NATIVE_BPE_IDS"][:3],
                         [11,12,2])
        self.assertTrue(d["EOS_preselected_original_TOP64"])
        self.assertEqual(len(d["states"]),2)
        for v in d["states"].values():
            self.assertEqual(len(v["scale_choices"]),4)
            self.assertEqual(v["scale_choices"]["0"]["choice_native_BPE2_ID"],11)
            self.assertEqual(len(v["original_top64_future_cosines"]),64)
        scored=posthoc_human_gold(d,True,12,native)
        self.assertEqual(scored["original_native_logit_gap_greedy_minus_gold_if_correct_OWN_FIRST"],1.)
        self.assertTrue(scored["gold_BPE2_in_original_TOP64_after_correct_first"])
        self.assertFalse(scored["original_greedy_BPE2_correct"])
        self.assertEqual(scored["original_native49152_second_gold_rank_if_correct_OWN_FIRST"],2)
        self.assertTrue(scored["human_gold_only_POSTHOC_no_inference_or_training"])
    def test_wrong_own_first_is_unscorable_not_oracle(self):
        native=torch.full((1,VOCAB),-18.)
        native[0,11]=5
        emb=torch.randn((VOCAB,HIDDEN))
        h=torch.randn((1,HIDDEN))
        states={x:TinyFutureState().eval().requires_grad_(False) for x in STATES}
        d=target_blind_calibrated_candidates(native,h,emb,states)
        scored=posthoc_human_gold(d,False,11,native)
        self.assertFalse(scored["gold_BPE2_in_original_TOP64_after_correct_first"])
        self.assertIsNone(scored["original_native49152_second_gold_rank_if_correct_OWN_FIRST"])
        self.assertIsNone(scored["original_native_logit_gap_greedy_minus_gold_if_correct_OWN_FIRST"])
        for v in scored["per_frozen_trained_state"].values():
            self.assertFalse(any(v["second_gold_correct_at_fixed_scale"].values()))
    def test_unfrozen_weights_invalid_native_shape_and_gold_id(self):
        native=torch.zeros((1,VOCAB))
        emb=torch.ones((VOCAB,HIDDEN))
        h=torch.zeros((1,HIDDEN))
        states={x:TinyFutureState() for x in STATES}
        with self.assertRaises(ValueError):
            target_blind_calibrated_candidates(native,h,emb,states)
        for s in states.values():s.requires_grad_(False)
        with self.assertRaises(ValueError):
            target_blind_calibrated_candidates(torch.zeros((1,42)),h,emb,states)
        d=target_blind_calibrated_candidates(native,h,emb,states)
        with self.assertRaises(ValueError):
            posthoc_human_gold(d,True,VOCAB,native)
if __name__=="__main__":
    unittest.main()
