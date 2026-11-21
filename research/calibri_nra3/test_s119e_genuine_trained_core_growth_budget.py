"""S119e deliberately synthetic accounting checks; real S119d parents via CI."""
import unittest

from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119e_genuine_trained_core_growth_budget import (
    LEGACY_ARCHITECTURE_CORE_PARAMS,
    UNTRAINED_OR_FIXED_ARCHITECTURE_GAP,
    minimum_new_genuine_params, signed_growth,
)


class S119EGenuineCoreExpansionRequirements(unittest.TestCase):
    PARENT = 3_780_223.108535898 + 79_830

    def sample(self,mode="new_trained_core",trained=2_000_000,bytes_add=100_000):
        return signed_growth(
            parent_charged_byte_equivalent=self.PARENT,
            grow_unique_trained_params=trained,
            new_packed_bytes=bytes_add,
            newly_functional_trained_params=trained,
            growth_mode=mode,
            counts_verified_by_real_loader=False,
        )

    def test_01_parent_rate_matches_real_S119d(self):
        r=self.sample(bytes_add=1,trained=1)
        self.assertAlmostEqual(
            r["projected_legacy_bpp"],
            8*(self.PARENT+1)/(LEGACY_ARCHITECTURE_CORE_PARAMS+1))
        self.assertFalse(r["model_accepted"])

    def test_02_new_genuinely_trained_core_raises_both_denominators(self):
        r=self.sample()
        self.assertEqual(r["legacy_denominator_new"],
                         LEGACY_ARCHITECTURE_CORE_PARAMS+2_000_000)
        self.assertEqual(r["unique_trained_denominator_new"],
                         36_380_480+2_000_000)

    def test_03_activate_existing_zero_channel_does_NOT_duplicate_legacy(self):
        r=self.sample(mode="activate_old_architecture_training")
        self.assertEqual(r["legacy_denominator_new"],LEGACY_ARCHITECTURE_CORE_PARAMS)
        self.assertEqual(r["unique_trained_denominator_new"],36_380_480+2_000_000)
        self.assertGreater(r["projected_legacy_bpp"],.713804174)

    def test_04_cannot_count_nonexistent_fixed_or_more_than_available(self):
        with self.assertRaises(GateFailure):
            self.sample(mode="activate_old_architecture_training",
                        trained=UNTRAINED_OR_FIXED_ARCHITECTURE_GAP+1)
        with self.assertRaises(GateFailure):
            self.sample(trained=0)

    def test_05_fake_zero_functionality_never_yields_bpp_pass(self):
        r=signed_growth(
            parent_charged_byte_equivalent=self.PARENT,
            grow_unique_trained_params=999_999_999,
            new_packed_bytes=1,
            newly_functional_trained_params=0,
            growth_mode="new_trained_core",
            counts_verified_by_real_loader=False)
        self.assertEqual(r["legacy_denominator_new"],LEGACY_ARCHITECTURE_CORE_PARAMS)
        self.assertFalse(r["projected_conservative_pass_rate_ONLY"])
        self.assertFalse(r["model_accepted"])

    def test_06_marginal_rate_at_or_above_target_can_never_rescue(self):
        for q in (.7,.701,.8,2.0):
            self.assertIsNone(minimum_new_genuine_params(
                parent_bytes=self.PARENT,
                reference_denominator=36_380_480,
                marginal_bits_per_new_trained_weight=q))

    def test_07_legacy_and_conservative_have_different_real_debt(self):
        a=minimum_new_genuine_params(
            parent_bytes=self.PARENT,
            reference_denominator=LEGACY_ARCHITECTURE_CORE_PARAMS,
            marginal_bits_per_new_trained_weight=.30)
        b=minimum_new_genuine_params(
            parent_bytes=self.PARENT,
            reference_denominator=36_380_480,
            marginal_bits_per_new_trained_weight=.30)
        self.assertGreater(b,a)
        self.assertGreater(b,13_000_000)
        self.assertLess(a,2_000_000)

    def test_08_additional_metadata_is_not_free(self):
        a=minimum_new_genuine_params(
            parent_bytes=self.PARENT,reference_denominator=36_380_480,
            marginal_bits_per_new_trained_weight=.30)
        b=minimum_new_genuine_params(
            parent_bytes=self.PARENT,reference_denominator=36_380_480,
            marginal_bits_per_new_trained_weight=.30,
            new_runtime_metadata_bytes=16_384)
        self.assertGreater(b,a)

    def test_09_training_and_cpu_and_j7_not_certified_by_accounting(self):
        r=self.sample(trained=16_000_000,bytes_add=50_000)
        self.assertTrue(r["projected_conservative_pass_rate_ONLY"])
        for field in ("complete_actual_loader_inventory_not_verified",
                      "future_real_616_replay_required",
                      "future_independent_quality_firstread_required",
                      "physical_source_growth_checkpoint_NOT_created_here"):
            self.assertTrue(r[field])
        self.assertFalse(r["model_accepted"])
        self.assertFalse(r["activation_allowed"])

    def test_10_invalid_negative_bytes_and_old_modes_rejected(self):
        with self.assertRaises(GateFailure):
            self.sample(bytes_add=-12)
        with self.assertRaises(GateFailure):
            self.sample(mode="dummy_large_untrained_fake_core")


if __name__=="__main__":
    unittest.main()
