"""S3G smoke tests for a controlled, non-blind research diagnostic split."""
import unittest
from collections import Counter
from research.nra2_bridge.stage3g_factorized_graph_controls import (
    build_rows, validate, GROUP_SIZES, oracle, FAMILIAR, NOVEL,
)


class TestFactorizedGraphControls(unittest.TestCase):
    def test_deterministic_provenance_and_unique_prompts(self):
        a, b = build_rows(), build_rows()
        self.assertTrue(validate(a))
        self.assertEqual([r.prompt_sha256 for r in a],
                         [r.prompt_sha256 for r in b])
        self.assertEqual(len(a), 128)

    def test_labels_structures_and_balanced_wordings(self):
        rows = build_rows()
        self.assertEqual(dict(Counter(r.group for r in rows)), GROUP_SIZES)
        for group, count in GROUP_SIZES.items():
            subset = [r for r in rows if r.group == group]
            self.assertEqual(sum(r.reachable for r in subset), count//2)
            template_set = NOVEL if group in ("CTRL_WORDING16", "CTRL_BOTH16") else FAMILIAR
            self.assertTrue(all(any(r.question.startswith(t.split("{edges}")[0])
                                    for t in template_set) for r in subset))
            for item in subset:
                self.assertEqual(oracle(item.edges, item.source, item.target),
                                 item.reachable)

    def test_controls_are_not_model_evaluation(self):
        # Model use is deliberately absent: no torch, no checkpoint, no logits.
        import research.nra2_bridge.stage3g_factorized_graph_controls as module
        self.assertFalse(hasattr(module, "load_model"))
        self.assertFalse(hasattr(module, "train"))
        self.assertFalse(hasattr(module, "score_model"))


if __name__ == "__main__":
    unittest.main()
