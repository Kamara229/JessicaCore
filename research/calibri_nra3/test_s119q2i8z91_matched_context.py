import unittest
from research.calibri_nra3.s119q2i8z91_matched_context import ARMS,prepare,verify
class TestZ91(unittest.TestCase):
    def test_context_only(self):
        a,b=[prepare()[x] for x in ARMS]
        self.assertEqual([r["completion"] for r in a],[r["completion"] for r in b])
        self.assertEqual(sum(r["prompt"]!=s["prompt"] for r,s in zip(a,b)),36)
        self.assertEqual(len({r["prompt"] for r in a if r["id"].startswith("S119Q2I8Z87_")}),12)
        self.assertEqual(len({r["prompt"] for r in b if r["id"].startswith("S119Q2I8Z87_")}),48)
        self.assertEqual(verify()["planned_updates_total"],1168)
if __name__=="__main__":unittest.main()
