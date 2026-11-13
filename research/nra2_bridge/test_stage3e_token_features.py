"""S3E all-token read-only CPU layer12 collector tests, no 43M load."""
import unittest
import torch
from torch import nn
from research.nra2_bridge.stage3e_token_features import FrozenAllTokenFeatures
from research.nra2_bridge.frozen_features import prompt_sha


class Toy12(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.Identity() for _ in range(12)])

    def forward(self,x):
        for layer in self.model.layers:
            x=layer(x)
        return x


class AllTokenReadOnlyTests(unittest.TestCase):
    def test_exact_sequence_zero_padding_and_unchanged_forward(self):
        torch.manual_seed(19)
        model=Toy12()
        collector=FrozenAllTokenFeatures().install(model)
        q=["graph alpha","graph beta"]
        a=torch.randn(1,5,320)
        b=torch.randn(1,7,320)
        with collector.prompt_scope(q[0]):
            self.assertIs(model(a),a)
            model(torch.randn(1,1,320))  # cache decoding must not change first prefill
        with collector.prompt_scope(q[1]):
            self.assertIs(model(b),b)
        f=collector.select([prompt_sha(x) for x in q])
        x,m,n=collector.select_tokens([prompt_sha(x) for x in q])
        self.assertEqual(tuple(x.shape),(2,7,320))
        self.assertEqual(n.tolist(),[5,7])
        self.assertEqual(m.sum(1).tolist(),[5,7])
        self.assertTrue(torch.equal(x[0,:5],a[0].half()))
        self.assertTrue(torch.equal(x[1,:7],b[0].half()))
        self.assertTrue(torch.count_nonzero(x[0,5:])==0)
        self.assertTrue(torch.equal(f,x[torch.arange(2),n.long()-1]))
        with self.assertRaises(AssertionError):
            collector.select_tokens([prompt_sha(q[0])])
        collector.uninstall()
        self.assertIs(model(a),a)

    def test_malformed_or_unlabeled_inputs_cannot_be_captured(self):
        model=Toy12()
        collector=FrozenAllTokenFeatures().install(model)
        model(torch.randn(1,4,320))
        self.assertFalse(collector.token_sequences)
        with collector.prompt_scope("bad"):
            with self.assertRaises(AssertionError):
                model(torch.randn(1,6,319))
        collector.uninstall()


if __name__=="__main__":
    unittest.main()
