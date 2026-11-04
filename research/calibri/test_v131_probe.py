import unittest
import torch
from v131.probe import rank
from v131.protocol import CHOSEN, MODES, examples


class TestFirstToken(unittest.TestCase):
    def test_rank_tiebreak(self):
        x = torch.tensor([2.0, 3.0, 3.0, 1.0])
        self.assertEqual(rank(x, 1), 1)
        self.assertEqual(rank(x, 2), 2)
        self.assertEqual(rank(x, 0), 3)
        self.assertEqual(rank(x, 3), 4)

    def test_counterfactuals_have_two_sides(self):
        prompts = list(examples())
        self.assertEqual(len(prompts), 2*len(CHOSEN))
        self.assertEqual(len(MODES), 7)
        for a, b in zip(prompts[::2], prompts[1::2]):
            self.assertEqual(a[0], b[0])
            self.assertEqual(a[4], b[5])
            self.assertEqual(a[5], b[4])
            self.assertNotEqual(a[2], a[3])


if __name__ == "__main__":
    unittest.main()
