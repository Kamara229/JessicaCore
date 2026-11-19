"""S90 source/gold/CI group TRAIN-only contracts, zero real model weights."""
import ast
import unittest
from pathlib import Path

import torch

from research.nra3_s20.s89_sourceonly_event_mention_top1_ranker import (
    S89MainEventMentionTop1,PAID,
)
from research.nra3_s20.s83_NEW_DIVERSE72_source_ONLY_never_use_S79_spent import (
    source_only_new72,source_only_new72_sha256,TRAIN_GROUPS,DIAG_GROUPS,
)
from research.nra3_s20.s90_real43m_GT_S83_TRAIN56_top1_selector_only import (
    STAGE,S83_SHA,S84_SHA,SEED,EPOCHS,N,K,top1_metrics,
)


class TestS90OnlyNewS83GroupTRAINFocus(unittest.TestCase):
    def test_real_old_S84_SHA_precommitted_for_new_TRAIN72_only(self):
        self.assertEqual(N,72)
        self.assertEqual(K,8)
        self.assertEqual(S83_SHA,source_only_new72_sha256())
        self.assertEqual(S84_SHA,
            "25c2cf61a3b3c331cd80b3c5e979eeb1badca3c3d2c8fd5125b7fc0de3fe35c4")
        self.assertEqual(SEED,901119)
        self.assertEqual(EPOCHS,80)
        self.assertEqual(STAGE,
            "NRA2_S90_REAL43M_GT_S83_TRAIN56_S89_MENTION_TOP1_5725_FP16_SHADOW")

    def test_train56_and_synthetic_DIAG16_groups_fixed_no_shared_source(self):
        src=source_only_new72()
        tr=[r for r in src if r["group"] in TRAIN_GROUPS]
        diag=[r for r in src if r["group"] in DIAG_GROUPS]
        self.assertEqual(len(tr),56)
        self.assertEqual(len(diag),16)
        self.assertTrue(set(x["group"] for x in tr).isdisjoint(
            x["group"] for x in diag))
        self.assertEqual(len({x["text"] for x in src}),72)

    def test_only_new_5725_sourceonly_ranker_parameters_gradient(self):
        ranker=S89MainEventMentionTop1()
        self.assertEqual(sum(p.numel() for p in ranker.parameters()),5725)
        self.assertEqual(PAID,5725)
        self.assertFalse(ranker.activation_allowed)
        self.assertFalse(ranker.candidate_accepted)
        self.assertEqual(ranker.output_gate,0)

    def test_no_spent_S79_S86_imports_into_S90_or_ancestor_source(self):
        files=(
            "research/nra3_s20/s90_real43m_GT_S83_TRAIN56_top1_selector_only.py",
            "research/nra3_s20/s85_true43mGT_S82_event_scope_train_new72.py",
            "research/nra3_s20/s89_sourceonly_event_mention_top1_ranker.py",
        )
        for p in files:
            tree=ast.parse(Path(p).read_text(encoding="utf8"))
            imp=[z.module or "" for z in ast.walk(tree)
                 if isinstance(z,ast.ImportFrom)]
            self.assertFalse(any("s79_NEW_NOVEL48" in x or
                                 "s86_newSYNTHETIC48" in x or
                                 "s86_NOVEL48_GOLD" in x or
                                 "s88_S85_FROZEN_REAL43M" in x
                                 for x in imp),p)

    def test_S83_constructed_gold_only_after_source_scoring_in_main(self):
        tree=ast.parse(Path(
            "research/nra3_s20/s90_real43m_GT_S83_TRAIN56_top1_selector_only.py"
        ).read_text(encoding="utf8"))
        scorer=next(x for x in tree.body if isinstance(x,ast.FunctionDef)
                    and x.name=="source_only_old_S59_scores_BEFORE_gold")
        self.assertFalse(any(isinstance(x,ast.ImportFrom) and
                             "gold" in (x.module or "").lower()
                             for x in ast.walk(scorer)))
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef)
                  and x.name=="main")
        text=ast.get_source_segment(
            Path("research/nra3_s20/s90_real43m_GT_S83_TRAIN56_top1_selector_only.py"
                 ).read_text(encoding="utf8"),main)
        self.assertLess(text.index("source_only_old_S59_scores_BEFORE_gold(args)"),
                        text.index("open_constructed_targets_ONLY_after_source_forward(source)"))
        self.assertLess(text.index("open_constructed_targets_ONLY_after_source_forward(source)"),
                        text.index("fixed_S83_train_ONLY(source,gold)"))


if __name__=="__main__":
    unittest.main()
