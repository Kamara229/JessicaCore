import unittest
import torch
from research.calibri_nra3.s119q2i8z57_fullanswer_eval import greedy,summarize,ARMS,rows
class TestZ57(unittest.TestCase):
    def test_native_own_prefix(self):
        calls=[]
        def step(p):
            calls.append(p);x=torch.zeros(1,8);x[0,4 if len(p)==2 else 2]=3;return x
        self.assertEqual(greedy([1,3],step,vocab=8),[4,2])
        self.assertEqual(calls,[[1,3],[1,3,4]])
    def test_fp32_not_candidate_gate(self):
        def data(n):
            return [{"id":r["id"],"kind":r["kind"],"whole_answer_and_EOS":i<n,
                     "whole_text_match":i<n,"EOS":True,"valid_UTF8":True,"generated_ids":[2]}
                    for i,r in enumerate(rows())]
        a={ARMS[0]:data(1),ARMS[1]:data(4),ARMS[2]:data(28)}
        first=summarize(a);a[ARMS[2]]=data(0)
        self.assertEqual(first["internal_gate"],summarize(a)["internal_gate"])
        self.assertTrue(first["all_internal_gates_pass"])
        self.assertFalse(first["candidate_accepted"])
if __name__=="__main__":unittest.main()
