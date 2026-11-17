"""S20d source-only grouped CV arithmetic and frozen-parent protection."""
from __future__ import annotations

import unittest
import torch

from research.nra3_s20.s20_new96_synthetic_graph_train_source import (
    train_only_split_contract,build_s20_synthetic_train96,
)
from research.nra3_s20.s20d_train96_grouped20_true43m_graph import (
    aggregate,new_graph_model,source_only_counts,
    PAID_PARENT,PAID_GRAPH,PAID_TOTAL,STEPS,
)
from research.nra3_s19.paid_grammar_parser import CompactWordGrammar
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import PaidSemanticEventGraph
from research.nra3_s20.test_s20c_graph_source_separation import (
    fake_geometry_for_tests_only,
)
from research.nra3_s20.s20c_real43m_new96_graph_target_join import (
    build_features,build_train_targets,
)


class S20dTwentyFoldTrainOnlyContracts(unittest.TestCase):
    def test_exact_twenty_source_train_folds_with_rare_labels_present(self):
        rows=build_s20_synthetic_train96()
        groups=train_only_split_contract()
        self.assertEqual(len(groups),20)
        self.assertEqual(len(rows),96)
        for typ,group,tr,held in groups:
            self.assertTrue(set(tr).isdisjoint(held))
            self.assertEqual(set(tr)|set(held),set(range(96)))
            self.assertEqual(len(tr)+len(held),96)
            self.assertEqual(len(held),(12 if typ=="LEXICAL" else 8))
            seen={e.factuality for i in tr for e in rows[i].events}
            self.assertEqual(seen,{"ASSERTED","NEGATED","FUTURE_UNREALIZED","CONDITIONAL"})
            self.assertGreater(sum(a.polarity=="REJECTED"
                for i in tr for a in rows[i].arguments),0)

    def test_every_paid_parameter_and_decoder_gate_remain_frozen_in_source(self):
        frozen=PaidContextualEventScope(
            CompactWordGrammar().eval().requires_grad_(False)).eval().requires_grad_(False)
        model=new_graph_model(frozen,320096)
        self.assertEqual((PAID_PARENT,PAID_GRAPH,PAID_TOTAL),(9332,3049,12381))
        self.assertEqual(STEPS,240)
        self.assertTrue(all(not p.requires_grad for p in model.parent.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.semantic_params()))
        self.assertEqual(model.output_gate,0)
        self.assertFalse(model.candidate_accepted)
        self.assertFalse(model.activation_allowed)

    def test_score_five_heads_only_on_96_synthetic_geometry_not_real_cpu(self):
        text,h,meta=fake_geometry_for_tests_only()
        features=build_features(text,h,meta)
        labels=build_train_targets(features)
        gold={
            "event_status_TRAIN_ONLY":labels.event_status,
            "predicate_argument_role_TRAIN_ONLY":labels.predicate_argument_role,
            "argument_polarity_TRAIN_ONLY":labels.argument_polarity,
            "coreference_TRAIN_ONLY":labels.coreference,
            "event_relation_TRAIN_ONLY":labels.event_links,
        }
        model=new_graph_model(
            PaidContextualEventScope(
                CompactWordGrammar().eval().requires_grad_(False)).eval().requires_grad_(False),
            320096)
        with torch.inference_mode():
            logits=model(features.raw_word_hidden320_and_raw18,features.valid_word_mask)
            metrics=source_only_counts(logits,gold,features.valid_word_mask)
        self.assertEqual(set(metrics),{
            "event","directed_role","argument_polarity","coreference","event_relation"})
        self.assertEqual(metrics["event"]["TRAIN_INTERNAL_held_positive_support"],160)
        self.assertEqual(metrics["directed_role"]["TRAIN_INTERNAL_held_positive_support"],432)
        self.assertEqual(metrics["argument_polarity"]["TRAIN_INTERNAL_held_positive_support"],432)
        self.assertEqual(metrics["coreference"]["TRAIN_INTERNAL_held_positive_support"],64)
        self.assertEqual(metrics["event_relation"]["TRAIN_INTERNAL_held_positive_support"],64)

    def test_micro_average_counts_not_mean_of_fold_f1(self):
        def fold(axis,tp,fp,fn):
            return {"axis":axis,
                    "held_TRAIN_metrics":{
                        k:{"TP":tp,"FP":fp,"FN":fn,
                           "TRAIN_INTERNAL_held_positive_support":tp+fn}
                        for k in ("event","directed_role","argument_polarity",
                                  "coreference","event_relation")}}
        rows=[fold("LEXICAL",1,0,0) for _ in range(7)]+[
            fold("LEXICAL",0,4,2)]
        rows += [fold("CONSTRUCTION",2,2,1) for _ in range(12)]
        lex=aggregate(rows,"LEXICAL")["event"]
        syn=aggregate(rows,"CONSTRUCTION")["event"]
        self.assertEqual((lex["TP"],lex["FP"],lex["FN"]),(7,4,2))
        self.assertAlmostEqual(
            lex["TRAIN_INTERNAL_grouped_micro_F1_NOT_independent"],14/(14+4+2))
        self.assertEqual((syn["TP"],syn["FP"],syn["FN"]),(24,24,12))


if __name__=="__main__":
    unittest.main()
