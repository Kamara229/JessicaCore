"""S6 offline tests: 43 paid matrix QAT, no S5 source fabricated as proof."""
import hashlib
import unittest

import numpy as np
import torch

from direct_codebook_v59 import CB59
from research.nra3_s0.neural_area import NRA3Config,NRA3NeuralArea
from research.nra3_s1.physical_cb59 import encode_area
from research.nra3_s4.russian_ud_train_only import build_train,TRAIN12_CANONICAL_SHA256
from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord,CausalAlignment,
)
from research.nra3_s6.physical_joint_grammar_qat import (
    PhysicallyPaidNRA3QAT,load_pinned_real43m_train12,S5_SOURCE_SHA256,
)


class PhysicallyPaidQATUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(306606)

    def test_requires_real_pinned_S5_source_not_fake_tensor(self):
        with self.assertRaises(ValueError):
            load_pinned_real43m_train12("/tmp/nonexistent-fake-S5.safetensors",
                                       "/tmp/nonexistent-fake-S5.json")
        self.assertEqual(len(S5_SOURCE_SHA256),64)
        self.assertEqual(len(TRAIN12_CANONICAL_SHA256),64)

    def test_single_unified_NRA3_6m_source_all_43_operators_paid_QAT(self):
        cfg=NRA3Config()
        # This test is MOCK numerical source only, NOT a trained Russian model.
        untrained=NRA3NeuralArea(cfg).cpu().eval()
        original,_,_=encode_area(untrained)
        self.assertEqual(CB59(original).meta["original_unique_parameter_count"],
                         6_106_154)
        qat=PhysicallyPaidNRA3QAT(original,cfg)
        self.assertEqual(len(qat.names),43)
        self.assertEqual(len(list(qat.books.parameters())),86)
        self.assertTrue(all(not p.requires_grad for p in qat._host.parameters()))
        rows=build_train()
        row=rows[0]
        n=10
        aligned=CausalAlignment(
            row.sha256(),n,tuple(range(100,100+n)),
            tuple(AlignedWord(i+1,(i,),i) for i in range(len(row.words))),
            n-1,
        )
        x=torch.randn(n,320)
        optimizer=torch.optim.Adam(qat.books.parameters(),lr=.04)
        before=qat(x,row,aligned)
        self.assertTrue(torch.isfinite(before.total))
        optimizer.zero_grad(set_to_none=True)
        before.total.backward()
        grads={k:b.centroids.grad for k,b in zip(qat.names,qat.books)}
        self.assertTrue(any(k.startswith("grammar.") and v is not None
                            and bool(v.abs().sum()>0) for k,v in grads.items()))
        self.assertTrue(any(k.startswith("semantic.") and v is not None
                            and bool(v.abs().sum()>0) for k,v in grads.items()))
        self.assertIsNotNone(grads["upos_head.weight"])
        self.assertIsNotNone(grads["role_head.weight"])
        optimizer.step()
        physical=qat.save_full_paid_physical_cb59()
        self.assertEqual(len(original),len(physical))
        self.assertNotEqual(hashlib.sha256(original).hexdigest(),
                            hashlib.sha256(physical).hexdigest())
        b0,b1=CB59(original),CB59(physical)
        self.assertEqual(b0.meta,b1.meta)
        self.assertEqual(set(b0.raw),set(b1.raw))
        for key in b0.raw:
            self.assertTrue(np.array_equal(b0.raw[key],b1.raw[key]),key)
        for key in b0.handles:
            self.assertEqual(b0.handles[key].packed.tobytes(),
                             b1.handles[key].packed.tobytes(),key)
        status=qat.physical_change_counts(physical)
        self.assertGreater(status["grammar_matrix_changes"],0)
        self.assertGreater(status["semantic_matrix_changes"],0)
        self.assertTrue(status["upos_head_changed"])
        self.assertTrue(status["role_head_changed"])
        self.assertTrue(status["original_all_1d_source_bytes_unchanged"])
        self.assertFalse(qat.candidate_accepted)
        self.assertEqual(qat.output_gate,0.)
        with self.assertRaises(ValueError):
            qat(torch.randn(n,321),row,aligned)
        qat.output_gate=.1
        with self.assertRaises(RuntimeError):
            qat(x,row,aligned)


if __name__=="__main__":
    unittest.main()
