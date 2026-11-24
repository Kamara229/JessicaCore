import unittest
import torch
from research.calibri_nra3.s119q2i8z83_first_two_train import early_margin,prefix_counts,ARMS
class TestZ83(unittest.TestCase):
    def test_margin(self):
        x=torch.tensor([[1.,2.,4.,0.]],requires_grad=True)
        y=early_margin(x,1,1);y.backward()
        self.assertLess(float(x.grad[0,1]),0.)
        self.assertGreater(float(x.grad[0,2]),0.)
        self.assertEqual(float(x.grad[0,0]),0.)
        self.assertEqual(float(early_margin(x,1,2)),0.)
    def test_prefix_counts(self):
        row={"positions":[{"target":1,"top1":1},{"target":2,"top1":2},{"target":3,"top1":9}]}
        self.assertEqual(prefix_counts([row]*42),{"first1":42,"first2":42,"first3":0})
        self.assertEqual(len(ARMS),2)
if __name__=="__main__":unittest.main()
