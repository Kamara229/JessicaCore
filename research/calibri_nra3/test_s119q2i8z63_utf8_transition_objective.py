import unittest
import torch
from research.calibri_nra3.s119q2i8z63_utf8_transition_objective import advance,allowed_mask,invalid_mass_loss
class TestZ63(unittest.TestCase):
    def test_utf8_state_constraints(self):
        pieces=[b"a",b"\xd0",None,b"\x9d",b"\xd0\x9d",None,b"\xed",b"\xa0",b"\x80"]
        self.assertEqual(advance(b"",b"\xd0"),b"\xd0")
        self.assertEqual(advance(b"\xd0",b"\x9d"),b"")
        self.assertIsNone(advance(b"\xd0",b"a"))
        self.assertIsNone(advance(b"\xed",b"\xa0")) # surrogate forbidden
        self.assertEqual(allowed_mask(pieces,b"",2).tolist(),[True,True,True,False,True,False,True,False,False])
        self.assertEqual(allowed_mask(pieces,b"\xd0",2).tolist(),[False,False,False,True,False,False,False,True,True])
    def test_loss_gradient_and_extremes(self):
        m=torch.tensor([True,False,True,False])
        x=torch.tensor([[0.,1000.,0.,0.]],requires_grad=True)
        y=invalid_mass_loss(x,m)
        self.assertTrue(torch.isfinite(y));y.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertLess(float(x.grad[0,0]),0.)
        self.assertGreater(float(x.grad[0,1]),0.)
        x=torch.zeros(1,4,requires_grad=True)
        self.assertAlmostEqual(float(invalid_mass_loss(x,m)),.25*__import__("math").log(2),places=6)
if __name__=="__main__":unittest.main()
