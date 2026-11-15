"""S8 verified structural and numerical tests: no EVAL, no model promotion."""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

import torch
from torch.nn import functional as F

from research.nra3_s0.neural_area import (
    NRA3Config,NRA3NeuralArea,SEMANTIC_ROLES,
)
from research.nra3_s1.physical_cb59 import encode_area,reconstruct_state
from research.nra3_s2.native_cpu import NRA3NativePackedCPU
from research.nra3_s4.russian_ud_train_only import build_train
from research.nra3_s4.complete_prefix_training_contract import (
    AlignedWord,CausalAlignment,
)
from research.nra3_s6.physical_joint_grammar_qat import PhysicallyPaidNRA3QAT
from research.nra3_s7.parallel_hypotheses import parallel_three_relations
from research.nra3_s7.parallel_training import (
    attach_parallel_objective,parallel_train_objective,
)
from research.nra3_s8.selector_supervision import (
    BRANCHES,GLOBAL,LOCAL,ROLE,train_only_causal_branch_targets,
    locked_train12_weak_target_counts,
)


class S8CalibratedRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(308008)

    @staticmethod
    def config(version):
        return NRA3Config(hidden=32,heads=4,grammar_blocks=1,
                          semantic_blocks=1,relation_width=24,
                          relation_steps=2,parallel_hypotheses=3,
                          parallel_selector_version=version)

    def aligned(self,row,length=None):
        if length is None:
            length=len(row.words)+3
        return CausalAlignment(
            row.sha256(),length,tuple(range(length)),
            tuple(AlignedWord(i+1,(i+1,),i+1)
                  for i in range(len(row.words))),length-1)

    def test_three_label_classes_are_TRAIN_ONLY_and_no_future_teacher(self):
        tally=locked_train12_weak_target_counts()
        self.assertEqual(set(tally),set(BRANCHES))
        self.assertTrue(all(n>=2 for n in tally.values()))
        rows=build_train()
        for row in rows:
            targets=train_only_causal_branch_targets(row,self.aligned(row))
            self.assertEqual(len(targets),len(row.words))
            self.assertTrue(all(k in (GLOBAL,LOCAL,ROLE)
                                for k in targets.tolist()))
            # Every token whose gold head is to its RIGHT cannot use the
            # local-head hypothesis just because of an unavailable head.
            for j,w in enumerate(row.words):
                if w.head!=0 and w.head>j+1 and targets[j]==LOCAL:
                    self.fail("S8 teacher used a future head as a local source")
        row=rows[0]
        bad=self.aligned(row)
        bad=bad.__class__(bad.original_prompt_sha256,bad.tokenizer_token_count,
                          bad.prefill_token_ids,bad.word_alignment,
                          bad.complete_prefix_position,True)
        with self.assertRaises(ValueError):
            train_only_causal_branch_targets(row,bad)

    def test_old_S7_gate_mode_is_identical_and_S8_calibration_is_nonuniform(self):
        b,t,r=1,10,24
        q=torch.ones(b,t,r)*.1
        v=torch.randn(b,t,r)*.1
        scores=torch.randn(b,t,t)*.05
        scores[:,-1,0]=2.5
        allowed=torch.tril(torch.ones(t,t,dtype=torch.bool))[None]
        valid=torch.ones(b,t,dtype=torch.bool)
        roles=torch.zeros(b,t,len(SEMANTIC_ROLES))
        roles[:,0,SEMANTIC_ROLES.index("PATIENT")]=6
        roles[:,-1,SEMANTIC_ROLES.index("AGENT")]=6
        keyword=dict(query=q,value=v,scores=scores,allowed=allowed,
                     valid=valid,semantic_role_logits=roles,local_window=4)
        old=parallel_three_relations(**keyword)
        legacy=parallel_three_relations(**keyword,selector_version=0)
        new=parallel_three_relations(**keyword,selector_version=1)
        self.assertTrue(torch.equal(old.message,legacy.message))
        self.assertTrue(torch.equal(old.hypothesis_gates,legacy.hypothesis_gates))
        self.assertEqual(tuple(new.hypothesis_gates.shape),(b,t,3))
        self.assertTrue(torch.isfinite(new.message).all())
        self.assertTrue(torch.allclose(new.hypothesis_gates.sum(-1),
                                       torch.ones(b,t),atol=1e-6))
        self.assertGreater(
            float((new.hypothesis_gates-1/3).abs().max()),.05)
        self.assertGreater(
            float((new.hypothesis_gates-old.hypothesis_gates).abs().max()),.05)
        future=scores.clone()
        future[:,6:,:]+=14
        shifted=parallel_three_relations(**(keyword|{"scores":future}),
                                         selector_version=1)
        self.assertTrue(torch.allclose(new.message[:,:5],
                                       shifted.message[:,:5],atol=1e-6,rtol=0))
        with self.assertRaises(ValueError):
            parallel_three_relations(**keyword,selector_version=3)
        with self.assertRaises(ValueError):
            NRA3Config(parallel_hypotheses=1,parallel_selector_version=1)

    def test_S8b_causal_running_mean_selector_preserves_past_only(self):
        b,t,r=1,12,24
        torch.manual_seed(308018)
        q=torch.randn(b,t,r)
        v=torch.randn(b,t,r)
        scores=torch.randn(b,t,t)*.2
        allowed=torch.tril(torch.ones(t,t,dtype=torch.bool))[None]
        valid=torch.ones(b,t,dtype=torch.bool)
        roles=torch.randn(b,t,len(SEMANTIC_ROLES))
        kwargs=dict(query=q,value=v,scores=scores,allowed=allowed,
                    valid=valid,semantic_role_logits=roles,local_window=8,
                    selector_version=2)
        original=parallel_three_relations(**kwargs)
        changed_q=q.clone();changed_q[:,7:]+=100
        changed_v=v.clone();changed_v[:,7:]-=100
        changed_role=roles.clone();changed_role[:,7:]+=20
        later=parallel_three_relations(**(kwargs|{
            "query":changed_q,"value":changed_v,
            "semantic_role_logits":changed_role
        }))
        self.assertTrue(torch.equal(original.hypothesis_gates[:,:7],
                                    later.hypothesis_gates[:,:7]))
        self.assertTrue(torch.equal(original.message[:,:7],
                                    later.message[:,:7]))
        self.assertTrue(torch.allclose(original.hypothesis_gates.sum(-1),
                                       torch.ones(b,t),atol=1e-6))
        self.assertGreater(float((original.hypothesis_gates-1/3).abs().max()),.02)
        model=NRA3NeuralArea(self.config(2)).eval()
        legacy=NRA3NeuralArea(self.config(0)).eval()
        self.assertEqual(set(model.state_dict()),set(legacy.state_dict()))
        self.assertEqual(model.parameter_count(),legacy.parameter_count())

    def test_S8_paid_source_no_new_parameters_joint_router_gradient(self):
        cfg=NRA3Config(parallel_hypotheses=3,parallel_selector_version=1)
        original=NRA3NeuralArea(cfg)
        legacy=NRA3NeuralArea(NRA3Config(parallel_hypotheses=3))
        self.assertEqual(original.parameter_count(),6_106_154)
        self.assertEqual(set(original.state_dict()),set(legacy.state_dict()))
        source,_,_=encode_area(original)
        qat=attach_parallel_objective(PhysicallyPaidNRA3QAT(source,cfg))
        row=build_train()[0]
        al=self.aligned(row,12)
        hidden=torch.randn(12,320)
        output=qat(hidden,row,al)
        self.assertIsNotNone(output.selector_train_ce)
        self.assertIsNotNone(output.selector_gold_indices)
        self.assertEqual(tuple(output.branch_means.shape),(3,))
        self.assertTrue(torch.isfinite(output.total))
        output.total.backward()
        names=dict(zip(qat.names,qat.books))
        for n in ("rel_q.weight","rel_k.weight","rel_value.weight",
                  "rel_update.weight_ih","role_head.weight",
                  "grammar.0.attn.in_proj_weight"):
            self.assertIsNotNone(names[n].centroids.grad,n)
            self.assertTrue(torch.isfinite(names[n].centroids.grad).all(),n)
        self.assertTrue(bool(names["rel_q.weight"].centroids.grad.abs().sum()>0))
        self.assertFalse(qat.candidate_accepted)
        self.assertEqual(qat.output_gate,0)
        with self.assertRaises(ValueError):
            qat(torch.randn(12,322),row,al)

    def test_original_native_5bit_C_CPU_S8_matches_same_quantized_dense(self):
        cfg=self.config(1)
        parent=NRA3NeuralArea(cfg).eval()
        source,_,_=encode_area(parent)
        dense=NRA3NeuralArea(cfg).eval()
        dense.load_state_dict(reconstruct_state(source,parent),strict=True)
        with tempfile.TemporaryDirectory() as td:
            lib=Path(td)/"lib_calibri_S8_native_cb59.so"
            subprocess.run(["cc","-O3","-std=c11","-fPIC","-shared",
                            "research/calibri/direct_kernel_v59.c",
                            "-o",str(lib)],check=True)
            packed=NRA3NativePackedCPU(source,cfg,lib).eval()
            row=build_train()[0]
            al=self.aligned(row,12)
            x=torch.randn(1,12,320)
            with torch.inference_mode():
                a=dense(x);b=packed(x)
                aa=parallel_train_objective(dense,x,row,al)
                bb=parallel_train_objective(packed,x,row,al)
            for name in ("relations","readout","parallel_hypothesis_gates"):
                self.assertTrue(torch.allclose(getattr(a,name),getattr(b,name),
                                               atol=8e-4,rtol=8e-4),name)
            self.assertTrue(torch.allclose(aa.total,bb.total,
                                           atol=8e-4,rtol=8e-4))
            self.assertTrue(torch.equal(aa.selector_gold_indices,
                                        bb.selector_gold_indices))
            self.assertFalse(packed.native_cpu_weights_dense_resident)
            self.assertEqual(len(list(packed.parameters())),0)
            self.assertEqual(packed.output_gate,0)
            self.assertFalse(packed.candidate_accepted)


if __name__=="__main__":
    unittest.main()
