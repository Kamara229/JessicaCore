"""TRAIN-only S122 64 symmetric pair curriculum synthetic invariants.

No QA, no model, no GPU, no user-specific data. Unit test is pure Python.
"""
from __future__ import annotations
import unittest
from collections import Counter
from calibri2.s122.independent_binding_train_curriculum import design_pairs,PAIR_COUNT


class TestTrainOnlyCounterfactualCurriculum(unittest.TestCase):
    def setUp(self):
        self.words=tuple(f"noun{i:02d}" for i in range(24))
        self.colors=("blue","green","red","yellow")
        self.names=(("A0","B0"),("A1","B1"),
                    ("A2","B2"),("A3","B3"))
        self.frames=tuple(
            f"TRAIN frame {i}: {{a}} is {{left}}; {{b}} is {{right}}"
            for i in range(4))

    def get(self):
        return design_pairs(self.words,self.colors,self.names,self.frames)

    def test_64_pairs_use_all_12_different_color_combinations(self):
        pairs=self.get()
        self.assertEqual(len(pairs),PAIR_COUNT)
        self.assertEqual(len(set(x["pair_id"] for x in pairs)),64)
        self.assertEqual(len(set((p["frame"],p["left_label"],p["right_label"])
                                 for p in pairs)),64)
        # The two sides of each counterfactual are both generated. A
        # reversed pair with the same frame would DUPLICATE examples.
        self.assertEqual(len(set((
            p["frame"],
            tuple(sorted((p["left_label"],p["right_label"])))
        ) for p in pairs)),64)
        color_pairs=Counter(
            (p["left_color"],p["right_color"]) for p in pairs)
        self.assertEqual(len(color_pairs),12)
        self.assertEqual(sum(color_pairs.values()),64)
        self.assertTrue(all(a!=b for a,b in color_pairs))
        self.assertTrue(all(p["left_label"]!=p["right_label"] for p in pairs))

    def test_four_frames_and_original_name_pairs_only(self):
        pairs=self.get()
        self.assertEqual({x["frame"] for x in pairs},set(self.frames))
        for x in pairs:
            position=self.frames.index(x["frame"])
            self.assertEqual((x["a"],x["b"]),self.names[position])
            self.assertIn(x["left_label"].split()[1],self.words)
            self.assertIn(x["right_label"].split()[1],self.words)

    def test_reproducible_without_randomness_or_hidden_QA(self):
        a=self.get();b=self.get()
        self.assertEqual(a,b)
        self.assertTrue(all(x["family"]=="S122_causal_cross_binding_TRAIN"
                            for x in a))

    def test_reject_unauthorized_vocab_or_frame_drift(self):
        bad_cases=[
            (self.words[:23],self.colors,self.names,self.frames),
            (self.words,self.colors[:3],self.names,self.frames),
            (self.words,self.colors,
             self.names[:3],self.frames),
            (self.words,self.colors,
             self.names,self.frames[:3]),
            (("noun0",)*24,self.colors,self.names,self.frames),
        ]
        for args in bad_cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    design_pairs(*args)


if __name__=="__main__":
    unittest.main(verbosity=2)
