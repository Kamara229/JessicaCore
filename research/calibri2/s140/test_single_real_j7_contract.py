"""S140 isolated structural safety contract; not real model evidence."""
from pathlib import Path
import unittest


class S140SingleJ7Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=Path(__file__).with_name(
            "train_single_real_j7_post_adam.py").read_text()

    def test_one_atomic_block_not_five_or_growth(self):
        self.assertIn("STEPS=2",self.source)
        self.assertIn("TRANSACTION_STEPS=2",self.source)
        self.assertIn("S140 single-transaction J7 invariant",self.source)
        self.assertNotIn("next_rank=authorize_growth(",self.source)
        self.assertIn("no_S136_growth_or_Jessica_promotion",self.source)

    def test_optimizer_aware_projection_and_full_physical_guard_intact(self):
        for required in (
            "correct_adam_stage2_displacement(",
            "atomic_train_block(",
            "persisted_cpu_guard",
            "strict_train_retention(",
            "mask_gradient_to_new_channels_only(expansion)",
            "verify_frozen_inherited(expansion,immutable_before)",
            "candidate_actual_CPU(",
            "S140_ONE_J7_ACTUAL_PHYSICAL_GUARD.json",
            "S140_ACTUAL_43M_ONE_J7_FP16.safetensors",
            "physical_FP16_reload_pass",
            "S140_REAL_43M_POST_ADAM_J7_VERIFIED",
        ):
            with self.subTest(required=required):
                self.assertIn(required,self.source)

    def test_rejected_update_must_preserve_all_inherited_answers(self):
        self.assertIn("final_old!=inherited_before or final_new!=inherited_new",self.source)
        self.assertIn("protected parent 456+128 responses lost",self.source)
        self.assertIn("S140 committed J7 transaction without new TRAIN gain",self.source)
        self.assertIn('bpp>.7 or actual_speed["ratio"]<.9',self.source)
        self.assertIn('"model_accepted":False',self.source)


if __name__=="__main__":
    unittest.main()
