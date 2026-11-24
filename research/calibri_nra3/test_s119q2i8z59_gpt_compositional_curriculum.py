import unittest
from research.calibri_nra3 import s119q2i8z59_gpt_compositional_curriculum as m
class TestZ59(unittest.TestCase):
    def test_manifest(self):
        x=m.verify()
        self.assertEqual((x["new_TRAIN"],x["paired_groups"],x["replay_TRAIN"],x["total_TRAIN"]),(56,28,42,98))
        self.assertEqual(x,m.verify())
        self.assertEqual(x["optimizer_steps"],0)
    def test_arithmetic(self):
        for (op,a,b),pair in zip(m.MATH,m.PAIRS["arithmetic_units"]):
            self.assertEqual(pair,m.arithmetic_pair(op,a,b))
        self.assertEqual(m.PAIRS["arithmetic_units"][0][0][1],"17")
        self.assertEqual(m.PAIRS["arithmetic_units"][1][0][1],"19")
        self.assertEqual(m.PAIRS["arithmetic_units"][2][0][1],"21")
        self.assertEqual(m.PAIRS["arithmetic_units"][3][1][1],"6")
if __name__=="__main__":unittest.main()
