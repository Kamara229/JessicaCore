"""S43 sealed NEW128 TRAIN-only multi-event labels and learned-source tests.

These unit checks do NOT prove original Calibri43M+Green Tunnel quality.
No protected S38 adversarial control sentences loaded or scored.
"""
from __future__ import annotations
import inspect
import unittest
import torch

from research.nra3_s20.s43_new128_multi_event_trainonly_source import (
    STAGE,build_train128,validate_source,train_source_sha,train_partitions,
    PATTERNS,
)
from research.nra3_s20.s43_joint_learned_predicate_source_adapter import (
    JointLearnedPredicateSourceS43,
    S43_TRAIN_EVENT_SOURCE_COEFF,
    S43_TRAIN_LINK_SOURCE_CONSISTENCY_COEFF,
)
from research.nra3_s20.s43_train128_joint_predicate_source_24fold import (
    batch_raw_plus_train_labels,train_one,
    S43_TRAIN_LR,S43_TRAIN_STEPS,S43_SEED,
)

class S43New128SourceAndTrainablePredicate(unittest.TestCase):
    def test_exact_new128_multievent_source_is_separate_from_old_train(self):
        summary=validate_source()
        self.assertEqual(summary["stage"],STAGE)
        self.assertEqual(summary["count"],128)
        self.assertEqual(summary["TRAIN_two_rejected_edges_per_sentence"],32)
        self.assertEqual(summary["TRAIN_zero_rejected_edges_per_sentence"],16)
        self.assertEqual(len(PATTERNS),16)
        self.assertEqual(summary["source_sha256"],train_source_sha())
        self.assertEqual(len(train_partitions()),24)
        self.assertTrue(summary["disjoint_original_TRAIN256_and_S39_TRAIN80"])
        self.assertTrue(summary["S38_reserved_not_imported_or_used"])
        for axis,group,tr,held in train_partitions():
            self.assertTrue(not set(tr)&set(held))
            self.assertEqual(len(tr),(112 if axis=="LEXICAL" else 120))
            self.assertEqual(len(held),(16 if axis=="LEXICAL" else 8))

    def test_two_rejected_recipients_share_one_source_without_one_to_one_cap(self):
        rows=build_train128()
        for family in range(8):
            # construction 6 is the counterexample to S42 greedy matching.
            row=rows[family*16+6]
            self.assertEqual(len(row.event_sources),1)
            self.assertEqual(len(row.rejected_edges),2)
            self.assertEqual(row.rejected_edges[0][0],row.rejected_edges[1][0])
            self.assertNotEqual(row.rejected_edges[0][1],row.rejected_edges[1][1])
        p,q,pair,structure,valid,legal,ev,qy,links=batch_raw_plus_train_labels(
            [rows[6],rows[7],rows[8],rows[9]])
        self.assertEqual(int(links.sum()),8)
        self.assertTrue(bool(legal.any()))
        self.assertEqual(ev.shape[:2],p.shape[:2])

    def test_trainable_source_head_and_link_head_have_real_gradients(self):
        rows=build_train128()[0:18]
        x,q,p,s,valid,legal,ev,quotes,links=batch_raw_plus_train_labels(rows)
        torch.manual_seed(431118)
        model=JointLearnedPredicateSourceS43()
        out=model(x,q,p,s)
        self.assertEqual(out.event_logits.shape,ev.shape)
        self.assertEqual(out.quote_logits.shape,quotes.shape)
        self.assertEqual(out.rejected_logits.shape,links.shape)
        # Differentiable link/event jointly, no label as model input.
        loss=(out.event_logits[valid].square().mean()+
              out.quote_logits[valid].square().mean()+
              out.rejected_logits[legal].square().mean())
        loss.backward()
        self.assertTrue(all(t.grad is not None and
                            torch.isfinite(t.grad).all()
                            for t in model.parameters()))
        self.assertTrue(any(k.startswith("source_head") for k in model.state_dict()))
        self.assertTrue(any(k.startswith("edge_head") for k in model.state_dict()))
        self.assertEqual((model.output_gate,model.candidate_accepted,
                          model.activation_allowed),(0,False,False))

    def test_source_head_is_neural_not_heuristic_pseudo_gold(self):
        import research.nra3_s20.s43_joint_learned_predicate_source_adapter as adapter
        import research.nra3_s20.s43_train128_joint_predicate_source_24fold as trainer
        forward=inspect.getsource(adapter.JointLearnedPredicateSourceS43.forward)
        self.assertIn("self.source_head(",forward)
        self.assertIn("self.edge_head(",forward)
        self.assertNotIn("S42",forward)
        for banned in ("gold[","row.event_sources","source_predicate_token_index0",
                       "family_id","group_id","held_ids","TRAIN_gold"):
            self.assertNotIn(banned,forward)
        code=inspect.getsource(trainer)
        self.assertNotIn("from research.nra3_s20.s38_",code)
        self.assertNotIn("matching_links_from_s40(",code)
        self.assertEqual((S43_TRAIN_STEPS,S43_TRAIN_LR),(120,.012))
        self.assertEqual(S43_TRAIN_EVENT_SOURCE_COEFF,2.)
        self.assertEqual(S43_TRAIN_LINK_SOURCE_CONSISTENCY_COEFF,.65)
        self.assertGreater(S43_SEED,0)

if __name__=="__main__":
    unittest.main()
