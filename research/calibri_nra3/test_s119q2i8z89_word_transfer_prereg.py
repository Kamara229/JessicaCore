import unittest
from research.calibri_nra3.s119q2i8z89_word_transfer_prereg import rows,verify,score,summarize
class TestZ89(unittest.TestCase):
    def test_manifest(self):
        self.assertEqual(verify()["cases"],24)
    def test_exact_score(self):
        r=rows()[0]
        self.assertTrue(score(r,r["completion"],eos=True,valid_utf8=True)["complete_correct"])
        self.assertFalse(score(r,r["completion"],eos=False,valid_utf8=True)["complete_correct"])
        self.assertFalse(score(r,r["completion"]+".",eos=True,valid_utf8=True)["complete_correct"])
        self.assertFalse(score(r,r["completion"],eos=True,valid_utf8=False)["complete_correct"])
    def test_group(self):
        data=[score(r,r["completion"],eos=True,valid_utf8=True) for r in rows()]
        self.assertEqual(summarize(data)["by_group"]["other_target"]["complete_correct"],12)
if __name__=="__main__":unittest.main()
