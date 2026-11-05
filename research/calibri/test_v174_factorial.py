"""Catch unknown source modules, omitted interactions and broken tied restoration."""
import unittest
from v174.protocol import MASKS, family, selected, freeze_cases


class ProtocolTests(unittest.TestCase):
    def test_unknown_tensor_fails_closed(self):
        for path in ("model.norm.weight", "model.layers.0.mlp.typo.weight", "unclassified.weight"):
            with self.assertRaises(ValueError):
                family(path)

    def test_full_factorial_including_rescue_interactions(self):
        paths = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
                 "model.layers.29.mlp.down_proj.weight"]
        assignments = {tuple(selected(mask, p) for p in paths) for mask in MASKS}
        self.assertEqual(len(assignments), 8)
        for mask in MASKS:
            self.assertEqual(selected(mask, paths[0]), selected(mask, "lm_head.weight"))

    def test_missing_or_duplicate_case_rejected(self):
        rows = [{"style": "direct", "question_id": i, "prompt_sha": str(i)} for i in range(96)]
        self.assertEqual(len(freeze_cases(rows)), 96)
        with self.assertRaises(ValueError):
            freeze_cases(rows[:-1])
        rows[-1] = rows[0]
        with self.assertRaises(ValueError):
            freeze_cases(rows)


class InterventionTests(unittest.TestCase):
    def test_exact_tied_roundtrip_across_all_interventions(self):
        import torch
        from v174.intervene import apply_mask
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(8, 8)
                self.lm_head = torch.nn.Linear(8, 8, bias=False)
                self.lm_head.weight = self.model.embed_tokens.weight
        model = Toy()
        name = "model.embed_tokens.weight"
        original = {name: model.model.embed_tokens.weight.detach().clone()}
        decoded = {name: torch.zeros_like(original[name])}
        for mask in (*MASKS, 0):
            apply_mask(model, original, decoded, mask)
            expected = decoded[name] if mask & 1 else original[name]
            self.assertTrue(torch.equal(model.lm_head.weight, expected))
            self.assertEqual(model.lm_head.weight.data_ptr(), model.model.embed_tokens.weight.data_ptr())


if __name__ == "__main__":
    unittest.main()
