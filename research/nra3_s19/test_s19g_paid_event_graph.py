"""S19g source/synthetic-only tests, NO independent EVAL or original CPU call."""
from __future__ import annotations

import unittest

import torch

from research.nra3_s19.paid_grammar_parser import CompactWordGrammar
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import (
    PaidSemanticEventGraph, semantic_parameter_count,full_parameter_count,
    STATUS,ROLE,ARG_STATUS,COREF_STATUS,EVENT_RELATION,
)
from research.nra3_s19.s19g_train9_real43m_graph import (
    fixed_multitask_loss,pair_targets,task_train_reuse_report,
)
from research.nra3_s19.s19f_real43m_train9_graph_targets import (
    assemble_gold_blind_338, make_train9_targets,
)
from research.nra3_s19.s19e_original43m_raw_alignment import s19_train_texts
from research.nra3_s19.graph_schema import raw_model_input


def fake_geometry_only():
    texts=s19_train_texts()
    counts=[len(raw_model_input(s).tokens) for s in texts]
    maxcount=max(counts)
    h=torch.zeros(9,maxcount,320,dtype=torch.float16)
    r={"wordpiece_alignment":[],"true_train9_token_counts":counts}
    for i,n in enumerate(counts):
        h[i,:n]=torch.randn(n,320).half()
        r["wordpiece_alignment"].append({
            "word_last_token_positions":list(range(n)),
            "word_token_positions":[[j] for j in range(n)]
        })
    input_data=assemble_gold_blind_338(texts,h,r)
    return input_data,make_train9_targets(input_data)


def empty_shadow():
    return PaidSemanticEventGraph(
        PaidContextualEventScope(
            CompactWordGrammar().eval().requires_grad_(False)).eval().requires_grad_(False))


class S19gFiveTypedGraphHeadContracts(unittest.TestCase):
    def test_5_heads_predict_all_words_and_pairs_without_oracle_predicate_masks(self):
        model=empty_shadow()
        data,y=fake_geometry_only()
        logits=model(data.raw_word_hidden320_and_raw18,data.valid_word_mask)
        b,w=data.valid_word_mask.shape
        self.assertEqual(logits.event_status.shape,(9,w,len(STATUS)))
        self.assertEqual(logits.role.shape,(9,w,w,len(ROLE)))
        self.assertEqual(logits.argument_polarity.shape,(9,w,w,len(ARG_STATUS)))
        self.assertEqual(logits.coreference.shape,(9,w,w,len(COREF_STATUS)))
        self.assertEqual(logits.event_relation.shape,(9,w,w,len(EVENT_RELATION)))
        self.assertTrue(bool(torch.isfinite(logits.event_status).all()))
        self.assertTrue(bool((logits.role[0,0,0]==-1e4).all()))
        self.assertEqual(model.output_gate,0)
        self.assertFalse(model.candidate_accepted)

    def test_all_physical_parent_and_new_semantic_weights_paid_and_sealed(self):
        model=empty_shadow()
        self.assertEqual(semantic_parameter_count(model),3049)
        self.assertEqual(full_parameter_count(model),12381)
        blob=model.physical_bytes()
        other=PaidSemanticEventGraph.from_physical(blob)
        self.assertEqual(model.head_physical(),other.head_physical())
        self.assertFalse(any(p.requires_grad for p in other.parameters()))
        self.assertEqual(other.output_gate,0)
        self.assertFalse(other.activation_allowed)

    def test_joint_loss_supervises_all_five_heads_no_frozen_parent_gradient(self):
        model=empty_shadow()
        data,y=fake_geometry_only()
        target={
          "event_status_TRAIN_ONLY":y.event_status,
          "predicate_argument_role_TRAIN_ONLY":y.predicate_argument_role,
          "argument_polarity_TRAIN_ONLY":y.argument_polarity,
          "coreference_TRAIN_ONLY":y.coreference,
          "event_relation_TRAIN_ONLY":y.event_links,
        }
        x,v=data.raw_word_hidden320_and_raw18,data.valid_word_mask
        loss=fixed_multitask_loss(model(x,v),target,v)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.parent.parameters()))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.semantic_params()))
        self.assertEqual(int((pair_targets(target)==2).sum()),1)

    def test_original_train9_supervises_distinct_event_truth_vs_role_rejection(self):
        data,y=fake_geometry_only()
        toks=raw_model_input(data.raw_texts[7]).tokens
        p={t.form:t.index-1 for t in toks}
        e=p["отправила"]
        self.assertEqual(y.event_status[7,e].item(),STATUS.index("ASSERTED"))
        self.assertEqual(y.predicate_argument_role[7,e,p["юристу"]].item(),ROLE.index("RECIPIENT"))
        self.assertEqual(y.predicate_argument_role[7,e,p["клиенту"]].item(),ROLE.index("RECIPIENT"))
        target={
          "event_status_TRAIN_ONLY":y.event_status,
          "predicate_argument_role_TRAIN_ONLY":y.predicate_argument_role,
          "argument_polarity_TRAIN_ONLY":y.argument_polarity,
          "coreference_TRAIN_ONLY":y.coreference,
          "event_relation_TRAIN_ONLY":y.event_links,
        }
        t=pair_targets(target)
        self.assertEqual(t[7,e,p["юристу"]].item(),ARG_STATUS.index("POSITIVE"))
        self.assertEqual(t[7,e,p["клиенту"]].item(),ARG_STATUS.index("REJECTED"))
        self.assertEqual(int((t==2).sum()),1)
        self.assertEqual(int((y.coreference==1).sum()),5)
        self.assertEqual(int((y.event_links>0).sum()),6)

    def test_corrupt_paid_payload_and_activation_do_not_load(self):
        model=empty_shadow()
        physical=model.physical_bytes()
        with self.assertRaises(ValueError):
            PaidSemanticEventGraph.from_physical(physical[:-10])
        with self.assertRaises(ValueError):
            PaidSemanticEventGraph.from_physical(b"FALSERIC"+physical[8:])
        model.output_gate=1
        with self.assertRaises(ValueError):
            model.physical_bytes()
        x,y=fake_geometry_only()
        with self.assertRaises(ValueError):
            model(x.raw_word_hidden320_and_raw18,x.valid_word_mask)

    def test_train_resubstitution_report_explicit_not_independent_metric(self):
        data,y=fake_geometry_only()
        target={
          "event_status_TRAIN_ONLY":y.event_status,
          "predicate_argument_role_TRAIN_ONLY":y.predicate_argument_role,
          "argument_polarity_TRAIN_ONLY":y.argument_polarity,
          "coreference_TRAIN_ONLY":y.coreference,
          "event_relation_TRAIN_ONLY":y.event_links,
        }
        o=empty_shadow()(data.raw_word_hidden320_and_raw18,data.valid_word_mask)
        metrics=task_train_reuse_report(o,target,data.valid_word_mask)
        self.assertEqual(set(metrics),
                         {"event","directed_role","argument_polarity",
                          "coreference","event_relation"})
        for task in metrics.values():
            self.assertTrue(0<=task["TRAIN_RESUBSTITUTION_micro_F1_NOT_independent"]<=1)
        self.assertEqual(metrics["event"]["TRAIN_ONLY_supervised_labels_nonNONE"],15)
        self.assertEqual(metrics["directed_role"]["TRAIN_ONLY_supervised_labels_nonNONE"],40)
        self.assertEqual(metrics["argument_polarity"]["TRAIN_ONLY_supervised_labels_nonNONE"],40)
        self.assertEqual(metrics["coreference"]["TRAIN_ONLY_supervised_labels_nonNONE"],5)
        self.assertEqual(metrics["event_relation"]["TRAIN_ONLY_supervised_labels_nonNONE"],6)


if __name__=="__main__":
    unittest.main()
