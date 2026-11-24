import unittest
from research.calibri_nra3.s119q2i8z75_joint_after_head import ARMS,PINS,verify_z73
class TestZ75(unittest.TestCase):
    def test_fixed_parents(self):
        self.assertEqual(set(ARMS),set(PINS))
        self.assertEqual(len(set(PINS.values())),2)
        self.assertTrue(all(len(x)==64 for x in PINS.values()))
        with self.assertRaises((ValueError,KeyError,RuntimeError)):verify_z73({"stage":"invalid"})
if __name__=="__main__":unittest.main()
