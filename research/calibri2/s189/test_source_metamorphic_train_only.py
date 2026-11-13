"""S189 standalone fail-closed contracts, pure Python source syntax; no model."""
import unittest
from calibri2.s189.source_metamorphic_train_only import run_source_metamorphic


class S189SourceSafety(unittest.TestCase):
    def test_source_only_16_template_metamorphic_coverage(self):
        r=run_source_metamorphic()
        s=r["groups"]
        for k in ("original_grammar_variants_correct",
                  "owner_value_swap_correct",
                  "injected_suffix_abstained",
                  "duplicate_owner_abstained",
                  "duplicate_value_abstained",
                  "different_colors_abstained",
                  "unsupported_paraphrase_abstained"):
            self.assertEqual(s[k],128)
        self.assertEqual(s["numeric_non_tie_correct"],72)
        self.assertEqual(s["numeric_ties_abstained"],9)
        self.assertEqual(s["numeric_unsafe_forms_abstained"],6)
        self.assertEqual(r["predeclared_synthetic_cases"],983)
        self.assertTrue(r["not_neural_generalization_and_not_model_accepted"])

if __name__=="__main__":
    unittest.main()
