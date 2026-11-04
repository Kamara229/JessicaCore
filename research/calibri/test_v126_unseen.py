"""Holdout gold references never supplied to independent host generation."""
import unittest
from v126.protocol import VALID,REVISION,SHA_V125
from v126.holdout import controls,text_key
class Tests(unittest.TestCase):
    def test_novel_real_world_reference_tasks(self):
        self.assertEqual(len(VALID),16)
        self.assertEqual(len(REVISION),40)
        self.assertEqual(len(SHA_V125),64)
        self.assertEqual(len({q for q,a in VALID}),16)
    def test_gold_candidates_only_for_grading(self):
        candidates={q:{"answer":"clearly bogus response"} for q,a in VALID}
        data=controls(candidates)
        self.assertEqual(len(data),48)
        for k in range(0,48,3):
            self.assertEqual([a[3] for a in data[k:k+3]],
              ["independent_gold_ORACLE_only","plausible_wrong","UNSEEN_GPT2_REAL_OUTPUT"])
            self.assertEqual(sum(a[2] for a in data[k:k+3]),1)
        self.assertEqual(text_key(" ALLo?"),"allo")
if __name__=="__main__":unittest.main()
