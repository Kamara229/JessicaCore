import unittest
import torch
from research.calibri_nra3.s119q2i8z58_train42_audit import audit_row,summarize
class TestZ58(unittest.TestCase):
    def test_reference_prefix_and_premature_eos(self):
        calls=[]
        def step(p):
            calls.append(p);x=torch.zeros(1,8);x[0,2]=5.;return x
        r=audit_row([1,3],[4,5,2],step,vocab=8)
        self.assertEqual(calls,[[1,3],[1,3,4],[1,3,4,5]])
        self.assertEqual([x["premature_EOS_top1"] for x in r],[True,True,False])
        self.assertTrue(r[-1]["correct"])
        self.assertEqual(r[-1]["rank"],1)
    def test_summary_split(self):
        def step(p):
            x=torch.zeros(1,8);x[0,2]=5.;return x
        r=audit_row([1],[4,2],step,vocab=8)
        s=summarize([{"id":str(i),"kind":"test","new":i<28,"positions":r} for i in range(42)])
        self.assertEqual(s["new"]["n"],28)
        self.assertEqual(s["replay"]["n"],14)
        self.assertEqual(s["all"]["tokens"],84)
        self.assertEqual(s["all"]["premature_EOS_top1"],42)
if __name__=="__main__":unittest.main()
