"""S34 research-only invariants: no QA leakage, exact S33 replay and old guard."""
import unittest
from pathlib import Path

from calibri2.s34.continued_protected_color_noun_fp32 import (
    S34_STEPS, S34_LR, S34_CURRICULUM,
    OLD_ANCHOR_BLEND, COLOR_WEIGHT, NOUN_WEIGHT, EOS_WEIGHT,
    EXPECTED_S33_ACCEPTED, EXPECTED_S33_NEW, EXPECTED_S33_OLD,
    FIRST_COLOR_MARGIN,
)
from calibri2.s33.one_step_lexical_noun_guard_fp32 import (
    TOTAL_STEPS as S33_STEPS, SOURCE_SHA as S33_SHA
)
from calibri2.s30.matched_dense_fp32_train import SOURCE_SHA as S30_SHA


class S34Contract(unittest.TestCase):
    def test_starting_weights_exact_and_predeclared_train(self):
        self.assertEqual(S33_STEPS, 128)
        self.assertEqual(S33_SHA, S30_SHA)
        self.assertEqual(S34_STEPS, 80)
        self.assertEqual(S34_LR, (0.00002, 0.000008, 0.000003))
        self.assertEqual(S34_CURRICULUM, ("new", "new", "old", "new"))
        self.assertEqual(OLD_ANCHOR_BLEND, 0.40)
        self.assertEqual((COLOR_WEIGHT, NOUN_WEIGHT, EOS_WEIGHT),
                         (4., 12., 2.))
        self.assertEqual(FIRST_COLOR_MARGIN, 1.5)
        self.assertEqual((EXPECTED_S33_ACCEPTED,
                          EXPECTED_S33_NEW,
                          EXPECTED_S33_OLD), (30, 4, 6))

    def test_no_frozen_QA_in_training_selection(self):
        path = Path(__file__).with_name(
            "continued_protected_color_noun_fp32.py")
        source = path.read_text()
        for banned in (
            "from calibri2.s29.", "import calibri2.s29.",
            "from calibri2.s26.frozen_new_qa",
            "from calibri2.s26.independent_qa",
            "S27_POSTHOC", "S28_READONLY"
        ):
            self.assertNotIn(banned, source)
        self.assertIn("protected.issubset(kept)", source)
        self.assertIn("model.load_state_dict(pre_weights)", source)
        self.assertIn("opt.load_state_dict(pre_opt)", source)
        self.assertIn("model.load_state_dict(weights)", source)
        self.assertIn("opt.load_state_dict(moments)", source)
        self.assertIn("if not replay[", source)
        self.assertIn("torch.save({", source)
        self.assertIn("dense_FP32_not_paid_0_7bpp", source)
        self.assertIn("only_TRAIN_used_in_update_and_selection", source)

    def test_original_guard_is_id_not_accuracy(self):
        from calibri2.s32.microguard_pcgrad_fp32 import guarded_old_ids
        old = [
            {"id": "a", "full_correct": True},
            {"id": "b", "full_correct": True},
            {"id": "c", "full_correct": False}
        ]
        now = [
            {"id": "a", "full_correct": False},
            {"id": "b", "full_correct": True},
            {"id": "c", "full_correct": True}
        ]
        self.assertEqual(len(guarded_old_ids(old)),
                         len(guarded_old_ids(now)))
        self.assertFalse(guarded_old_ids(old).issubset(guarded_old_ids(now)))


if __name__ == "__main__":
    unittest.main()
