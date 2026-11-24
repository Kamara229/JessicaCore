import unittest
import torch
from research.calibri_nra3.s119q2i8z71_representation_probe import fit_predict,stats,rank_stats
class TestZ71(unittest.TestCase):
    def test_probe_and_rank(self):
        x=torch.tensor([[-2.,0.],[-1.,0.],[1.,0.],[2.,0.]])
        y=torch.tensor([0,0,1,1])
        p=fit_predict(x,y,torch.tensor([[-3.,0.],[3.,0.]]))
        self.assertEqual(p.tolist(),[0,1])
        self.assertEqual(stats(torch.tensor([0,1]),p)["balanced_accuracy"],1.)
        self.assertAlmostEqual(rank_stats(x)["effective_rank"],1.)
if __name__=="__main__":unittest.main()
