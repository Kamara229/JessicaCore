"""CPU-only S3B tests; no Calibri model load, no QA/TEST or LIVE."""
import unittest
import torch
import torch.nn.functional as F

from research.nra2_standalone.model import RelationalNRA
from research.nra2_bridge.stage3b_relational_adapter import (
    Hidden320ToRelationalEdges,NUMERIC_PAIRS,FAMILIES,
    partition_proxy_targets,sha_text,zero_gate_passthrough,frozen_proxy_forward,
)


class Stage3BTests(unittest.TestCase):
    def test_balanced_predeclared_train_holdout_split(self):
        numeric={sha_text(f"Answer Yes or No: does {a} exceed {b}?"):
                 "numeric_new_wording" for a,b in NUMERIC_PAIRS}
        owners={sha_text(f"{fam} synthetic source {j}"):fam
                for fam in FAMILIES[:3] for j in range(8)}
        grouped={**numeric,**owners}
        self.assertEqual(len(grouped),32)
        audit={"dataset_question_sha256_sorted":sorted(grouped)}
        s190={"hashed_per_prompt_audit":[{"prompt_sha256":k,"group":g}
                                        for k,g in grouped.items()]}
        tr,held,labels=partition_proxy_targets(audit,s190)
        self.assertEqual((len(tr),len(held)),(24,8))
        self.assertEqual(len(labels),32)
        self.assertFalse(set(tr)&set(held))
        self.assertEqual((sum(labels[k] for k in tr),sum(labels[k] for k in held)),(9,3))
        self.assertEqual((tr,held),partition_proxy_targets(audit,s190)[:2])

    def test_trained_edge_adapter_updates_without_pretrained_nra(self):
        torch.manual_seed(12)
        ad=Hidden320ToRelationalEdges().cpu()
        area=RelationalNRA(width=16,hops=5).cpu().eval().requires_grad_(False)
        snapshot={k:v.clone() for k,v in area.state_dict().items()}
        x=torch.randn(6,320)
        y=torch.tensor([0.,1.,0.,1.,0.,1.])
        edge,core=frozen_proxy_forward(ad,area,x)
        self.assertEqual(tuple(edge.shape),(6,))
        self.assertEqual(tuple(core.shape),(6,))
        loss=F.binary_cross_entropy_with_logits(edge,y)+0.25*F.binary_cross_entropy_with_logits(core,y)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in ad.parameters()))
        self.assertTrue(all(p.grad is None for p in area.parameters()))
        self.assertTrue(all(torch.equal(v,snapshot[k]) for k,v in area.state_dict().items()))

    def test_zero_gate_returns_same_tensor_exactly(self):
        original=torch.randn(3,11)
        result=zero_gate_passthrough(original,torch.tensor([0.,1.,-1.]))
        self.assertIs(result,original)
        with self.assertRaises(AssertionError):
            zero_gate_passthrough(original,torch.zeros(3),gate=.0001)
        with self.assertRaises(AssertionError):
            zero_gate_passthrough(original,torch.tensor([float("nan")]*3))

    def test_reject_wrong_hidden_shape_and_nonfinite(self):
        ad=Hidden320ToRelationalEdges()
        with self.assertRaises(AssertionError):
            ad(torch.zeros(3,319))
        with self.assertRaises(AssertionError):
            ad(torch.full((3,320),float("nan")))


if __name__=="__main__":
    unittest.main()
