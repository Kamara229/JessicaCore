import unittest
from research.calibri_nra3.s119q2i8z104_terminal_transfer_prereg import rows, verify, score, summarize
class TestZ104(unittest.TestCase):
    def test_prereg(self):
        self.assertEqual(verify()["cases"],36)
        self.assertEqual(len(verify()["arms"]),4)
    def test_primary(self):
        r=rows()[0];w=r["completion"]
        self.assertTrue(score(r,w,eos=True,valid_utf8=True)["complete_correct"])
        self.assertFalse(score(r,w,eos=False,valid_utf8=True)["complete_correct"])
        self.assertFalse(score(r,w,eos=True,valid_utf8=False)["complete_correct"])
        self.assertFalse(score(r,w+".",eos=True,valid_utf8=True)["complete_correct"])
    def test_pairs(self):
        a=summarize([score(r,r["completion"],eos=True,valid_utf8=True) for r in rows()])
        self.assertEqual(a["overall"]["complete_correct"],36)
        self.assertEqual(a["known_pairs_both_correct"],12)
        self.assertEqual(a["by_group"]["other_target"]["n"],12)
        rec=[score(r,r["completion"],eos=i!=0,valid_utf8=True) for i,r in enumerate(rows())]
        self.assertEqual(summarize(rec)["known_pairs_both_correct"],11)
if __name__=="__main__":unittest.main()
