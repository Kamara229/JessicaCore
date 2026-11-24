import unittest
import torch
from research.calibri_nra3.s119q2i8z94_micro_overfit import metrics,own_metrics,STEPS
class TestZ94(unittest.TestCase):
    def test_metrics(self):
        x=torch.tensor([[0.,3.],[4.,0.]])
        self.assertEqual(metrics(x,torch.tensor([1,0]))["top1"],2)
        self.assertEqual(STEPS,120)
    def test_own_metrics(self):
        rows=[{"id":str(i),"complete_correct":False,"text_correct":True,"EOS":False,"valid_UTF8":True,"prefix_match":2,"generated_ids":[1,3]} for i in range(12)]
        self.assertEqual(own_metrics(rows)["complete_correct"],0)
        self.assertEqual(own_metrics(rows)["first_two_correct"],12)
if __name__=="__main__":unittest.main()
