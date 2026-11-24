import unittest
from research.calibri_nra3.s119q2i8z79_margin_train import ARMS,EPOCHS,PARENT_SHA,GT_SHA
from research.calibri_nra3.s119q2i8z78_pending_gold_margin import manifest
class TestZ79(unittest.TestCase):
    def test_preregistered(self):
        m=manifest()
        self.assertEqual(ARMS,tuple(m["arms"]))
        self.assertEqual(EPOCHS,m["epochs_per_arm"])
        self.assertEqual(PARENT_SHA,m["parent_joint_SHA256"])
        self.assertEqual(GT_SHA,m["GT12_SHA256"])
if __name__=="__main__":unittest.main()
