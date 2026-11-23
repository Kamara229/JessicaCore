"""Z21 source and future BPE2..4 native predictor prereg verification (no original43M forward)."""
import unittest
from collections import Counter
import inspect
import torch
from research.calibri_nra3.s119q2i8z21_fresh28train14dev_future_state_source import (
    KINDS, rows, verify,digest,TOP_K,LATENT_DIM,
    PREDICTED_FUTURE_POSITIONS,TRAIN_EPOCHS,
)
from research.calibri_nra3.s119q2i8z21_future_state_lowrank16_matched_contract import (
    HIDDEN,VOCAB,STATES,TinyFutureState,deterministic_derangement_indices,
    target_next3_original_embeddings,train_matched_aligned_vs_permuted,
)

class TestZ21SourceAndTargetFreeFutureState(unittest.TestCase):
    def test_01_source_precommit_distinct_prior_seven_kinds(self):
        x=verify()
        self.assertEqual((x["TRAIN_count"],x["DEV_count"]),(28,14))
        self.assertEqual(x["TRAIN_SHA256"],digest(rows("TRAIN")))
        self.assertEqual(x["DEV_SHA256"],digest(rows("DEV")))
        self.assertTrue(x["disjoint_old_E_G_K_R_T_W_X_Z2_Z9_Z12_Z14_Z16_Z18"])
        self.assertEqual(Counter(r["kind"] for r in rows("TRAIN")),{k:4 for k in KINDS})
        self.assertEqual(Counter(r["kind"] for r in rows("DEV")),{k:2 for k in KINDS})
        self.assertEqual((TOP_K,LATENT_DIM,PREDICTED_FUTURE_POSITIONS,TRAIN_EPOCHS),
                         (64,16,(1,2,3),30))
        for x in ("GOLD","EVAL128","J7","TEST","TRAIN+DEV"):
            with self.assertRaises(ValueError): rows(x)

    def test_02_10240_params_and_initial_state_valid(self):
        p=TinyFutureState()
        self.assertEqual(sum(x.numel() for x in p.parameters()),10240)
        self.assertEqual((p.output_gate,p.candidate_accepted),(0,False))
        h=torch.arange(HIDDEN,dtype=torch.float32).unsqueeze(0)
        state=p(h)
        self.assertEqual(state.shape,(1,HIDDEN))
        self.assertTrue(torch.isfinite(state).all())
        self.assertGreater(float(state.abs().sum()),0.)
        self.assertEqual(p.train_loss(h,state.detach(),split="TRAIN").ndim,0)
        with self.assertRaises(ValueError):
            p.train_loss(h,state.detach(),split="DEV")
    def test_03_target_bpe2_bpe3_bpe4_exact_original_embedding(self):
        emb=torch.zeros((VOCAB,HIDDEN))
        emb[31,0]=1
        emb[32,1]=1
        emb[33,2]=1
        t=target_next3_original_embeddings(emb,[9,31,32,33,2])
        self.assertEqual(t.shape,(1,HIDDEN))
        self.assertAlmostEqual(float(t[0,0]),3**-.5,places=5)
        self.assertAlmostEqual(float(t[0,1]),3**-.5,places=5)
        self.assertAlmostEqual(float(t[0,2]),3**-.5,places=5)
        with self.assertRaises(ValueError):
            target_next3_original_embeddings(emb,[9,31,32])
    def test_04_top64_native_candidates_selected_before_future_state(self):
        p=TinyFutureState()
        logits=torch.full((1,VOCAB),-10.,requires_grad=True)
        with torch.no_grad():
            logits[0,321]=20
            logits[0,322]=19
            logits[0,2]=18
        emb=torch.zeros((VOCAB,HIDDEN),requires_grad=False)
        with torch.no_grad():
            emb[321,0]=1;emb[322,1]=1;emb[2,2]=1
        h=torch.randn((1,HIDDEN),requires_grad=True)
        ids,scores,state=p.predict_future_candidates(logits,h,emb)
        self.assertEqual(ids.shape,(64,))
        self.assertEqual(ids[:3].tolist(),[321,322,2])
        self.assertEqual(scores.shape,(64,))
        self.assertFalse(state.requires_grad)
        self.assertFalse(bool(torch.isnan(scores).any()))
        self.assertIn(2,ids.tolist())
        self.assertLessEqual(float((scores-logits.detach()[0,ids]).abs().max()),2.00001)
        self.assertIsNone(h.grad)
        self.assertIsNone(logits.grad)
    def test_05_shuffled_human_train_labels_deranged_and_locked(self):
        d=deterministic_derangement_indices()
        self.assertEqual(d,deterministic_derangement_indices())
        self.assertEqual(len(d),28)
        self.assertEqual(set(d.values()),set(range(28)))
        self.assertTrue(all(x!=y for x,y in d.items()))
        code=inspect.getsource(train_matched_aligned_vs_permuted)
        self.assertIn('name==STATES[0] else derange[i]',code)
        self.assertIn('if r["own_first_correct"]',code)
        self.assertIn('target_future_embedding',code)
        self.assertIn('original43M_gradient_updates',code)
        self.assertNotIn('DEV',code.split('def train_matched_aligned_vs_permuted',1)[-1].split('return models,summary')[0].replace('no DEV',''))
if __name__=="__main__":
    unittest.main()
