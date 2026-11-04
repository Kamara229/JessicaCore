"""Pure fixed-size/gain-range proof before real 135M packed backprop tests."""
import unittest
from types import SimpleNamespace
import torch
from v122.steer import Installed
from v127.actuation import Actuator
from v127.protocol import ORACLE_CASES,STEPS,TRAIN_MODES,MAX_DELTA
class Test(unittest.TestCase):
    def test_six_numbers_manage_any_of_sixty_paid_operators(self):
        bridge=object.__new__(Installed)
        bridge.maps=[(SimpleNamespace(gain=None),i%2,(i//2+.5)/30.) for i in range(60)]
        a=Actuator(bridge)
        x=torch.zeros(6,requires_grad=True)
        output=a.oracle(x,"shared_six_oracle")
        self.assertEqual(tuple(output.shape),(60,))
        torch.testing.assert_close(output,torch.ones_like(output))
        out=a.oracle(torch.tensor([4.,0.,0.,-4.,0.,0.]),"shared_six_oracle")
        self.assertTrue((out>=.3).all() and (out<=1.7).all())
        loss=(a.oracle(x,"shared_six_oracle")*torch.linspace(1.,3.,60)).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad.abs().sum()),0)
        y=torch.zeros(60,requires_grad=True)
        controlled=a.oracle(y,"individual_sixty_oracle")
        controlled.sum().backward()
        self.assertEqual(len(y.grad),60)
        self.assertTrue(torch.isfinite(y.grad).all())
        a.neutral()
        self.assertTrue(all(float(m.gain)==1 for m,r,d in bridge.maps))
        with self.assertRaises(ValueError):a.oracle(torch.zeros(6),"individual_sixty_oracle")
    def test_frozen_oracle_setup(self):
        self.assertEqual(STEPS,12)
        self.assertEqual(len(ORACLE_CASES),6)
        self.assertEqual(len(TRAIN_MODES),2)
        self.assertEqual(MAX_DELTA,.7)
if __name__=="__main__":
    unittest.main()
