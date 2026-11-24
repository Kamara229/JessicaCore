import unittest
import torch
from research.calibri_nra3.s119q2i8z84_early_probe import predict,native_metrics
class TestZ84(unittest.TestCase):
    def test_fixed_classifier(self):
        x=torch.tensor([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
        y=torch.tensor([17,17,92,92])
        self.assertEqual(predict(x,y,x).tolist(),y.tolist())
        self.assertTrue(set(predict(x,y,torch.zeros(2,2)).tolist())<={17,92})
    def test_metrics(self):
        self.assertEqual(native_metrics([{"target":1,"top1":2,"rank":4},{"target":2,"top1":2,"rank":1}])["correct"],1)
if __name__=="__main__":unittest.main()
