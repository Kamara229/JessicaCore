"""S3BC: fail-closed data-only, role-order and heldout leak tests."""
import unittest

from research.nra2_bridge.stage3bc_contrastive_train512_prereg import (
    build_rows, commitment, validate, STYLES, SPLIT,
    DESTINATION_FIRST, SOURCE_FIRST,
)
from research.nra2_bridge.stage3aq_train_dev_futureeval_prereg import (
    build_rows as source_rows, TRAIN, DEV, EVAL,
)
from research.nra2_bridge.stage3au_train512_paraphrase_prereg import build_rows as s3au_rows
from research.nra2_bridge.stage3ay_structural_train512_prereg import build_rows as s3ay_rows
from research.nra2_bridge.stage3t_unseen_wording_prereg import query_char_offsets


class Stage3BCGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = build_rows()
        cls.parent = source_rows()
        cls.prereg = commitment(cls.rows)

    def test_exactly_512_unique_train_only(self):
        self.assertTrue(validate(self.rows))
        self.assertEqual({r.split for r in self.rows}, {SPLIT})
        self.assertEqual(len(self.rows), 512)
        self.assertEqual(len({r.prompt_sha256 for r in self.rows}), 512)
        self.assertEqual(len({r.graph_id for r in self.rows}), 64)

    def test_no_dev_future_eval_or_prior_train_overlap(self):
        old_sha = {r.prompt_sha256 for r in self.parent}
        old_sha.update(r.prompt_sha256 for r in s3au_rows())
        old_sha.update(r.prompt_sha256 for r in s3ay_rows())
        self.assertFalse({r.prompt_sha256 for r in self.rows} & old_sha)
        train_keys = {r.topology_key for r in self.parent if r.split == TRAIN}
        held_keys = {r.topology_key for r in self.parent if r.split in (DEV, EVAL)}
        self.assertEqual({r.topology_key for r in self.rows}, train_keys)
        self.assertFalse({r.topology_key for r in self.rows} & held_keys)

    def test_balanced_role_order_and_labels_per_grammar(self):
        self.assertEqual(len(STYLES), 8)
        self.assertEqual(DESTINATION_FIRST, (1, 3, 5, 7))
        self.assertEqual(SOURCE_FIRST, (0, 2, 4, 6))
        for i in range(8):
            group = [r for r in self.rows if r.query_style == i]
            self.assertEqual(len(group), 64)
            self.assertEqual(sum(r.reachable for r in group), 32)
            for r in group:
                order = query_char_offsets(r.question)["text_order_nodes"]
                self.assertEqual(order, [r.dst, r.src] if i%2 else [r.src, r.dst])

    def test_preregistered_no_model_and_protected_splits(self):
        p = self.prereg
        self.assertEqual(p["TRAIN512_count"], 512)
        self.assertEqual(p["S3AQ_parent_commitment_sha256"],
                         "c55799793263d5b4175fdfd62c293bcc2cb1794c73fa6ba856cf8b0a961f7839")
        self.assertEqual(p["S3BA_pinned_role_fp16_sha256"],
                         "4fedf391ee02b1d1c45c82c3690d212ba9a43e9d0da32446f5d32ac6c379bb09")
        self.assertEqual(len(p["commitment_sha256"]), 64)
        self.assertTrue(p["future_EVAL128_not_prefilled_scored_or_trained"])
        self.assertTrue(p["old_DEV128_already_scored_and_excluded_from_training"])
        self.assertTrue(p["no_new_CPU_prefill_or_model_weights_or_training_or_scores"])
        self.assertFalse(p["candidate_accepted"])
        self.assertEqual(p["output_gate"], 0.0)


if __name__ == "__main__":
    unittest.main()
