"""Pure numeric and full restoration regressions for v87."""
import unittest
import torch
from torch import nn
from v87.metrics import score,mean
from v87.oracle import intervene,SCENARIOS
class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(8,3)
        self.lm_head=nn.Linear(3,8,bias=False)
class TestCausalAtlas(unittest.TestCase):
    def test_scenarios_have_no_duplicate_targets(self):
        self.assertEqual(SCENARIOS["sham_baseline"],())
        for paths in SCENARIOS.values():
            self.assertEqual(len(paths),len(set(paths)))
    def test_original_embedding_restored_after_exception(self):
        s,t=Dummy(),Dummy()
        before=s.model.embed_tokens
        with self.assertRaisesRegex(ValueError,"test rollback"):
            with intervene(s,t,("model.embed_tokens",)):
                self.assertIs(s.model.embed_tokens,t.model.embed_tokens)
                raise ValueError("test rollback")
        self.assertIs(s.model.embed_tokens,before)
    def test_original_same_logits_have_zero_kl(self):
        x=torch.tensor([[1.,0.,0.],[0.,2.,1.],[2.,0.,1.],[0.,0.,3.]])
        ids=torch.tensor([0,1,2,0])
        z=score(x,x.clone(),ids)
        self.assertAlmostEqual(z["full_vocab_KL"],0.,places=10)
        self.assertEqual(mean([z,z])["tokens"],6)
    def test_wrong_source_rejected(self):
        with self.assertRaises(AssertionError):
            score(torch.zeros(3,4),torch.zeros(2,4),torch.tensor([0,1,2]))
if __name__=="__main__":
    unittest.main()
