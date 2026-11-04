"""Invariant tests: same 10MB controller across host depths and widths."""
import unittest
import torch
import numpy as np
from torch import nn
from v122.policy import SharedGreenPolicy
from v122.steer import SteeringBlock
from v122.protocol import TRAIN,VALID,STEPS,MAX_CHECKPOINT_BYTES
class Tests(unittest.TestCase):
    def test_shared_model_byte_and_layer_invariance(self):
        torch.set_num_threads(2)
        policy=SharedGreenPolicy().eval()
        self.assertLess(sum(p.numel()*4 for p in policy.parameters()),MAX_CHECKPOINT_BYTES)
        with torch.inference_mode():
            c=policy("Find a correct answer.")
            self.assertEqual(tuple(c.shape),(6,))
            gates=[]
            for layer_count in (3,30,300,3000):
                values=[policy.gates(c,(k+.5)/layer_count,role)
                        for role in (0,1) for k in (0,layer_count//2,layer_count-1)]
                gates.append(torch.stack(values))
            for group in gates:
                torch.testing.assert_close(group,torch.ones_like(group),atol=1e-8,rtol=0)
        self.assertEqual(sum(p.numel() for p in policy.parameters()),
                         sum(p.numel() for p in SharedGreenPolicy().parameters()))
    def test_model_widths_and_steer_grad(self):
        for cols,rows in ((8,16),(32,12),(576,1536)):
            class Base(nn.Module):
                def forward(self,x):
                    return torch.zeros((*x.shape[:-1],rows),dtype=x.dtype)
            u=np.ones((rows,1),dtype=np.float16)
            v=np.ones((1,cols),dtype=np.float16)
            block=SteeringBlock(Base(),u,v)
            g=torch.tensor(1.,requires_grad=True)
            block.gain=g
            y=block(torch.ones(1,cols))
            self.assertEqual(tuple(y.shape),(1,rows))
            y.mean().backward()
            self.assertIsNotNone(g.grad)
            self.assertGreater(float(g.grad),0)
    def test_frozen_train_valid_task_sets(self):
        self.assertEqual(len(TRAIN),STEPS)
        self.assertEqual(len(VALID),12)
        self.assertFalse(set(TRAIN)&set(VALID))
        self.assertFalse({q for q,a in TRAIN}&{q for q,a in VALID})
if __name__=="__main__":
    unittest.main()
