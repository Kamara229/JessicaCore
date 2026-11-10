"""Deterministic capacity math only; no model or QA loading."""
import unittest
from calibri2.s126.progressive_capacity import (
    ROADMAP,REFERENCE_FP32_UNIQUE_PARAMS,PhysicalAdmission,
    validate_precommitted_geometry,Geometry,
)


class CapacityTest(unittest.TestCase):
    def test_exact_36m_counts_and_monotonic_135m(self):
        rows=validate_precommitted_geometry()
        self.assertEqual(rows[0]["unique_params"],36_380_480)
        self.assertEqual(rows[1]["unique_params"],39_822_400)
        self.assertEqual(rows[-1]["unique_params"],135_427_584)
        self.assertLess(abs(rows[-1]["unique_params"]-
                            REFERENCE_FP32_UNIQUE_PARAMS)/
                        REFERENCE_FP32_UNIQUE_PARAMS,0.01)
        self.assertTrue(all(r["theoretical_only"] for r in rows))

    def test_geometry_rejects_invalid_gqa(self):
        with self.assertRaises(ValueError):
            Geometry("invalid",320,1536,12,6,2).unique_parameters()

    def test_admission_fails_any_skill_loss_or_budget_or_speed(self):
        ok=PhysicalAdmission(456,0,0,3_000_000,36_380_480,0.95,True,1)
        self.assertTrue(ok.eligible_for_RESEARCH_checkpoint())
        self.assertFalse(PhysicalAdmission(456,1,0,3_000_000,
                         36_380_480,0.95,True,1).eligible_for_RESEARCH_checkpoint())
        self.assertFalse(PhysicalAdmission(456,0,1,3_000_000,
                         36_380_480,0.95,True,1).eligible_for_RESEARCH_checkpoint())
        self.assertFalse(PhysicalAdmission(456,0,0,4_000_000,
                         36_380_480,0.95,True,1).eligible_for_RESEARCH_checkpoint())
        self.assertFalse(PhysicalAdmission(456,0,0,3_000_000,
                         36_380_480,0.89,True,1).eligible_for_RESEARCH_checkpoint())
        self.assertFalse(PhysicalAdmission(128,0,0,3_000_000,
                         36_380_480,0.95,True,1).eligible_for_RESEARCH_checkpoint())
        self.assertFalse(PhysicalAdmission(456,0,0,3_000_000,
                         36_380_480,0.95,False,1).eligible_for_RESEARCH_checkpoint())


if __name__=="__main__":
    unittest.main()
