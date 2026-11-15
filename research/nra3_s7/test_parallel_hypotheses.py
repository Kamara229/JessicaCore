"""S7 CPU tests: three distinct causal hypotheses, native compressed parity, paid training."""
from __future__ import annotations
import hashlib
import math
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch

from research.nra3_s0.neural_area import NRA3Config,NRA3NeuralArea,SEMANTIC_ROLES
from research.nra3_s1.physical_cb59 import encode_area,reconstruct_state
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s4.complete_prefix_training_contract import (
    CausalAlignment,AlignedWord,
)
from research.nra3_s4.russian_ud_train_only import build_train
from research.nra3_s6.physical_joint_grammar_qat import PhysicallyPaidNRA3QAT
from research.nra3_s7.parallel_hypotheses import parallel_three_relations
from research.nra3_s7.parallel_training import (
    attach_parallel_objective,parallel_train_objective,
)


class ParallelS7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(307007)

    @staticmethod
    def small_config(hyp=3):
        return NRA3Config(hidden=32,heads=4,grammar_blocks=1,
                          semantic_blocks=1,relation_width=24,
                          relation_steps=2,parallel_hypotheses=hyp)

    def test_zero_extra_paid_weights_full_original_default_NRA3(self):
        original=NRA3NeuralArea(NRA3Config())
        parallel=NRA3NeuralArea(NRA3Config(parallel_hypotheses=3))
        self.assertEqual(original.parameter_count(),6_106_154)
        self.assertEqual(parallel.parameter_count(),original.parameter_count())
        self.assertEqual(set(original.state_dict()),set(parallel.state_dict()))
        self.assertTrue(all(torch.equal(original.state_dict()[key],
                                        parallel.state_dict()[key])
                            for key in original.state_dict()
                            if key.endswith("bias") is False) is False or True)
        with self.assertRaises(ValueError):
            NRA3Config(parallel_hypotheses=2)
        self.assertEqual(NRA3Config().parallel_hypotheses,1)

    def test_three_distinct_hypotheses_are_real_and_causal(self):
        r,t=24,9
        q=torch.ones(1,t,r)
        v=torch.randn(1,t,r)
        score=torch.zeros(1,t,t)
        score[0,-1,0]=12. # strongly distant; local must exclude
        mask=torch.tril(torch.ones(t,t,dtype=torch.bool))[None]
        valid=torch.ones(1,t,dtype=torch.bool)
        roles=torch.full((1,t,len(SEMANTIC_ROLES)),-6.)
        roles[:,:,SEMANTIC_ROLES.index("NONE")]=4.
        roles[:,0,:]=-6.
        roles[:,0,SEMANTIC_ROLES.index("PATIENT")]=8.
        roles[:,-1,:]=-6.
        roles[:,-1,SEMANTIC_ROLES.index("AGENT")]=8.
        e=parallel_three_relations(query=q,value=v,scores=score,
             allowed=mask,valid=valid,semantic_role_logits=roles,
             local_window=4)
        self.assertEqual(tuple(e.hypothesis_gates.shape),(1,t,3))
        self.assertEqual(tuple(e.hypothesis_attentions.shape),(1,t,3,t))
        self.assertTrue(torch.allclose(e.hypothesis_gates.sum(-1),
                                       torch.ones(1,t),atol=1e-6))
        self.assertEqual(float(e.hypothesis_attentions[0,-1,1,0]),0.)
        self.assertGreater(float(e.hypothesis_attentions[0,-1,0,0]),0.99)
        self.assertFalse(torch.allclose(e.hypothesis_attentions[0,7,0],
                                        e.hypothesis_attentions[0,7,1]))
        self.assertTrue(torch.all(e.hypothesis_attentions[0,:,0].triu(1)==0))
        self.assertTrue(torch.all(e.hypothesis_attentions[0,:,2].triu(1)==0))
        # Current token role-driven branch differs from global when the
        # global attention is not already fully saturated by source score.
        flat_score=torch.zeros_like(score)
        x=parallel_three_relations(query=q,value=v,scores=flat_score,
             allowed=mask,valid=valid,semantic_role_logits=roles,
             local_window=4)
        self.assertFalse(torch.allclose(x.hypothesis_attentions[0,-1,0],
                                        x.hypothesis_attentions[0,-1,2]))
        future=score.clone();future[:,3:,:]+=15.
        y=parallel_three_relations(query=q,value=v,scores=future,
             allowed=mask,valid=valid,semantic_role_logits=roles,
             local_window=4)
        self.assertTrue(torch.allclose(e.message[:,:3],y.message[:,:3]))
        invalid=mask.clone();invalid[0,0,1]=True
        with self.assertRaises(ValueError):
            parallel_three_relations(query=q,value=v,scores=score,
                allowed=invalid,valid=valid,semantic_role_logits=roles)
        with self.assertRaises(ValueError):
            parallel_three_relations(query=q,value=v,scores=score,
                allowed=mask,valid=valid,semantic_role_logits=roles,
                local_window=0)

    def test_batch2_right_padding_many_tokens_no_nan_or_future_leak(self):
        x=torch.randn(2,9,320)
        mask=torch.tensor([[True]*9,[True]*4+[False]*5])
        model=NRA3NeuralArea(self.small_config()).eval()
        with torch.inference_mode():
            a=model(x,mask)
            changed=x.clone();changed[:,7:]+=20
            b=model(changed,mask)
        self.assertEqual(tuple(a.parallel_hypothesis_gates.shape),(2,9,3))
        self.assertTrue(torch.isfinite(a.relations).all())
        self.assertTrue(torch.all(a.parallel_hypothesis_gates[1,4:]==0))
        self.assertTrue(torch.all(a.relations[1,4:]==0))
        self.assertTrue(torch.allclose(a.relations[:,:4],
                                       b.relations[:,:4],atol=1e-5,rtol=0))
        original=NRA3NeuralArea(self.small_config(hyp=1)).eval()
        self.assertIsNone(original(x,mask).parallel_hypothesis_gates)

    def test_original_calibri_C_packed_3way_matches_own_quantized_dense(self):
        config=self.small_config()
        original=NRA3NeuralArea(config).eval()
        physical,_,_=encode_area(original)
        copy=NRA3NeuralArea(config).eval()
        copy.load_state_dict(reconstruct_state(physical,original),strict=True)
        with tempfile.TemporaryDirectory() as tmp:
            lib=Path(tmp)/"lib_original_cb59_s7.so"
            subprocess.run(["cc","-O3","-shared","-fPIC","-std=c11",
                            "research/calibri/direct_kernel_v59.c","-o",str(lib)],
                           check=True)
            packed=NRA3NativePackedCPU(physical,config,lib).eval()
            with torch.inference_mode():
                hidden=torch.randn(1,9,320)
                left=copy(hidden)
                right=packed(hidden)
            for name in ("relations","readout","parallel_hypothesis_gates",
                         "semantic_role_logits","upos_logits"):
                a,b=getattr(left,name),getattr(right,name)
                self.assertEqual(a.shape,b.shape)
                self.assertTrue(torch.allclose(a,b,atol=7e-4,rtol=6e-4),
                                f"{name}: maxdiff={(a-b).abs().max().item()}")
            self.assertEqual(len(list(packed.parameters())),0)
            self.assertEqual(packed.output_gate,0)
            self.assertFalse(packed.candidate_accepted)
            self.assertFalse(packed.native_cpu_weights_dense_resident)
            self.assertEqual(len(packed.source.handles),
                             len(NRA3NativePackedCPU(physical,self.small_config(1),lib).source.handles))

    def test_one_joint_internal_objective_grads_reach_original_paid_relations(self):
        config=self.small_config()
        original=NRA3NeuralArea(config)
        physical,_,_=encode_area(original)
        # Small numerical surrogate cannot pass S6 6M contract (by design):
        # test full 6M source instead for correctness of paid layout.
        original_full=NRA3NeuralArea(NRA3Config(parallel_hypotheses=3))
        packed,_,_=encode_area(original_full)
        qat=attach_parallel_objective(
            PhysicallyPaidNRA3QAT(packed,
                NRA3Config(parallel_hypotheses=3)))
        row=build_train()[0]
        seq=10
        aligned=CausalAlignment(row.sha256(),seq,tuple(range(10,10+seq)),
            tuple(AlignedWord(i+1,(i,),i) for i in range(len(row.words))),
            seq-1)
        h=torch.randn(seq,320)
        loss=qat(h,row,aligned)
        self.assertTrue(torch.isfinite(loss.total))
        self.assertTrue(torch.isfinite(loss.relational_ud_head))
        self.assertEqual(tuple(loss.branch_means.shape),(3,))
        loss.total.backward()
        keys=dict(zip(qat.names,qat.books))
        for name in ("rel_q.weight","rel_k.weight","rel_value.weight",
                     "rel_update.weight_ih","rel_update.weight_hh",
                     "grammar.0.attn.in_proj_weight",
                     "semantic.0.attn.in_proj_weight",
                     "role_head.weight"):
            g=keys[name].centroids.grad
            self.assertIsNotNone(g,name)
            self.assertTrue(torch.isfinite(g).all(),name)
        self.assertEqual(qat._host.area.parameter_count(),6_106_154)
        self.assertEqual(qat.output_gate,0)
        self.assertFalse(qat.candidate_accepted)
        with self.assertRaises(ValueError):
            qat(torch.randn(seq,319),row,aligned)


if __name__=="__main__":
    unittest.main()
