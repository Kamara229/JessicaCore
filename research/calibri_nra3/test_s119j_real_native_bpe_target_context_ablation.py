"""Synthetic safeguards; actual native token work in SHA-locked CI step."""
import unittest
import torch
from torch import nn
from research.calibri_nra3.s119a_hard_four_goals import GateFailure
from research.calibri_nra3.s119j_real_native_bpe_target_context_ablation import (
    build_frozen_train_lexicon,label_to_index,collect_targets,
    frozen_unigram_baseline,source_only_objective,logits_at_actual_native_bpe,
    original_source_valid,
)
from research.calibri_nra3.s119i_real12_contextual_structured_operator_ablation import (
    UnifiedCalibriWithStructuredInterface,
)

class S119JRealOriginalNativeBPEContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(119123)
        torch.set_num_threads(2)

    @staticmethod
    def fake_h256():
        # Strict synthetic SOURCE schema; not proof of actual S114.
        widths=20
        h=torch.randn(256,12,widths,320,dtype=torch.float16)
        mask=torch.ones(256,widths,dtype=torch.bool)
        ids=torch.arange(256*widths,dtype=torch.int32).reshape(256,widths)%191+1
        return h,mask,ids

    def test_01_train_native_vocab_only_no_gold_and_no_oov_mapping(self):
        h,mask,ids=self.fake_h256()
        original_source_valid(h,mask,ids)
        vocabulary,prototypes,counts=build_frozen_train_lexicon(h,mask,ids)
        self.assertGreater(vocabulary.numel(),100)
        self.assertEqual(prototypes.shape,(vocabulary.numel(),320))
        self.assertTrue(torch.isfinite(prototypes).all())
        self.assertEqual(label_to_index(vocabulary,999999),None)
        i=label_to_index(vocabulary,int(vocabulary[0]))
        self.assertIsNotNone(i)
        self.assertEqual(int(vocabulary[i]),int(vocabulary[0]))

    def test_02_disjoint_training_vs_probe_cohort(self):
        h,mask,ids=self.fake_h256()
        vocabulary,_,_=build_frozen_train_lexicon(h,mask,ids)
        train,tc=collect_targets(mask,ids,vocabulary,range(0,64))
        probe,pc=collect_targets(mask,ids,vocabulary,range(128,160))
        self.assertFalse(set(x[0] for x in train).intersection(x[0] for x in probe))
        self.assertEqual(tc["sampled_next_token_positions"],
                         tc["within_TRAIN_lexicon_positions"]+
                         tc["unknown_next_token_positions_NOT_EVALUATED"])
        self.assertEqual(pc["sampled_next_token_positions"],
                         pc["within_TRAIN_lexicon_positions"]+
                         pc["unknown_next_token_positions_NOT_EVALUATED"])

    def test_03_training_only_vocab_does_not_depend_on_probe_text(self):
        h,mask,ids=self.fake_h256()
        a,b,c=build_frozen_train_lexicon(h,mask,ids)
        h[128:]+=100.
        ids[128:]=99999
        d,e,f=build_frozen_train_lexicon(h,mask,ids)
        self.assertTrue(torch.equal(a,d))
        self.assertTrue(torch.equal(b,e))
        self.assertTrue(torch.equal(c,f))

    def test_04_matched_context_local_exact_identity_at_zero_fuse(self):
        m=UnifiedCalibriWithStructuredInterface().eval()
        h=torch.randn(1,12,12,320)
        valid=torch.ones(1,12,dtype=torch.bool)
        prototypes=torch.randn(37,320)
        with torch.inference_mode():
            a,*_=logits_at_actual_native_bpe(
                m,h,valid,8,prototypes,contextual=True)
            b,*_=logits_at_actual_native_bpe(
                m,h,valid,8,prototypes,contextual=False)
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(m.candidate_accepted)

    def test_05_real_native_token_label_cross_entropy_reaches_graph_and_core(self):
        m=UnifiedCalibriWithStructuredInterface()
        for p in m.nra3_neural_weights.parameters():
            p.requires_grad_(False)
        h=torch.randn(1,12,12,320)
        valid=torch.ones(1,12,dtype=torch.bool)
        prototypes=torch.randn(25,320)
        loss,parts=source_only_objective(
            m,h,valid,6,4,prototypes,contextual=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(parts["native_BPE_candidate_crossentropy"],0)
        loss.backward()
        for v in (m.next_state_projector.weight.grad,
                  m.structured_interface.fuse.weight.grad,
                  m.core_growth.up.grad,m.reasoning_role_gate.grad,
                  m.planning_action_area.action_selector.weight.grad):
            self.assertIsNotNone(v)
            self.assertTrue(torch.isfinite(v).all())
            self.assertGreater(float(v.abs().sum()),0)

    def test_06_must_refuse_next_token_as_model_input(self):
        m=UnifiedCalibriWithStructuredInterface()
        h=torch.randn(1,12,7,320)
        valid=torch.ones(1,7,dtype=torch.bool)
        prototypes=torch.randn(25,320)
        with self.assertRaises(GateFailure):
            logits_at_actual_native_bpe(m,h,valid,7,prototypes,contextual=True)

    def test_07_label_unigram_is_train_only(self):
        h,mask,ids=self.fake_h256()
        vocab,_,counts=build_frozen_train_lexicon(h,mask,ids)
        train,_=collect_targets(mask,ids,vocab,range(0,64))
        u=frozen_unigram_baseline(vocab,counts,train)
        self.assertEqual(u.shape,vocab.shape)
        self.assertTrue(torch.isfinite(u).all())
        self.assertAlmostEqual(float(u.exp().sum()),1,places=4)

if __name__=="__main__":
    unittest.main()
