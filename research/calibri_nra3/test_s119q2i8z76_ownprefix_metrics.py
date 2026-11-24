import unittest
import torch
from research.calibri_nra3.s119q2i8z76_ownprefix_metrics import score,greedy,common_cap
class TestZ76(unittest.TestCase):
    def test_no_gold_prefix(self):
        seen=[]
        def step(p):
            seen.append(p);x=torch.zeros(1,5);x[0,3 if len(seen)==1 else 2]=1;return x
        self.assertEqual(greedy([0,1],step,4,vocab=5),[3,2])
        self.assertEqual(seen,[[0,1],[0,1,3]])
        self.assertEqual(common_cap([[0]*130,[0]*10]),29)
    def test_empty_and_repeat(self):
        s=score([2],[3,2],"","да",True,48)
        self.assertTrue(s["empty_text"])
        self.assertFalse(s["complete_correct"])
        self.assertFalse(s["nonempty_valid_UTF8"])
        self.assertTrue(score([3,4,5,6]*2,[3,2],"abc","да",True,8)["repeat4_block"])
if __name__=="__main__":unittest.main()
