"""S22d source-only grouped CV arithmetic and frozen-parent protection."""
from __future__ import annotations

import unittest
import torch

from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    splits,build_s22_train256,
)
from research.nra3_s20.s22d_train256_grouped40_true43m_graph import (
    aggregate,new_graph_model,source_only_counts,rejected_specific_counts,
    PAID_PARENT,PAID_GRAPH,PAID_TOTAL,STEPS,
)
from research.nra3_s19.paid_grammar_parser import CompactWordGrammar
from research.nra3_s19.s19d_paid_contextual_scope import PaidContextualEventScope
from research.nra3_s19.s19g_paid_semantic_graph import PaidSemanticEventGraph
from research.nra3_s20.test_s22c_real43m_new256_source_separation import (
    fake_geometry_for_tests_only,
)
from research.nra3_s20.s22c_real43m_new256_graph_target_join import (
    build_features,build_train_targets,
)


class S22dFortyFoldTrainOnlyContracts(unittest.TestCase):
    def test_exact_forty_source_train_folds_with_rare_labels_present(self):
        rows=build_s22_train256()
        groups=splits()
        self.assertEqual(len(groups),40)
        self.assertEqual(len(rows),256)
        for typ,group,tr,held in groups:
            self.assertTrue(set(tr).isdisjoint(held))
            self.assertEqual(set(tr)|set(held),set(range(256)))
            self.assertEqual(len(tr)+len(held),256)
            self.assertEqual(len(held),(32 if typ=="LEXICAL" else 8))
            seen={e.factuality for i in tr for e in rows[i].events}
            self.assertEqual(seen,{"ASSERTED","NEGATED","FUTURE_UNREALIZED","CONDITIONAL"})
            self.assertGreater(sum(a.polarity=="REJECTED"
                for i in tr for a in rows[i].arguments),0)

    def test_every_paid_parameter_and_decoder_gate_remain_frozen_in_source(self):
        frozen=PaidContextualEventScope(
            CompactWordGrammar().eval().requires_grad_(False)).eval().requires_grad_(False)
        model=new_graph_model(frozen,320256)
        self.assertEqual((PAID_PARENT,PAID_GRAPH,PAID_TOTAL),(9332,3049,12381))
        self.assertEqual(STEPS,240)
        self.assertTrue(all(not p.requires_grad for p in model.parent.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.semantic_params()))
        self.assertEqual(model.output_gate,0)
        self.assertFalse(model.candidate_accepted)
        self.assertFalse(model.activation_allowed)

    def test_score_five_heads_only_on_256_synthetic_geometry_not_real_cpu(self):
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
            320256)
        with torch.inference_mode():
            logits=model(features.raw_word_hidden320_and_raw18,features.valid_word_mask)
            metrics=source_only_counts(logits,gold,features.valid_word_mask)
        self.assertEqual(set(metrics),{
            "event","directed_role","argument_polarity","coreference","event_relation"})
        self.assertEqual(metrics["event"]["TRAIN_INTERNAL_held_positive_support"],448)
        self.assertEqual(metrics["directed_role"]["TRAIN_INTERNAL_held_positive_support"],976)
        self.assertEqual(metrics["argument_polarity"]["TRAIN_INTERNAL_held_positive_support"],976)
        self.assertEqual(metrics["coreference"]["TRAIN_INTERNAL_held_positive_support"],192)
        self.assertEqual(metrics["event_relation"]["TRAIN_INTERNAL_held_positive_support"],192)

    def test_rejected_specific_one_vs_rest_is_separate_from_combined_polarity(self):
        class Output:
            pass
        out=Output()
        # B=1, W=3, three explicit non-diagonal pair cells inspected.
        valid=torch.tensor([[True,True,True]])
        gold=torch.full((1,3,3),-1,dtype=torch.long)
        # Actual REJECTED on 0->1, correct; another on 1->2, missed.
        gold[0,0,1]=1
        gold[0,1,2]=1
        # Non-role negative on 2->0 and remaining eligible arcs.
        logits=torch.zeros(1,3,3,3)
        logits[:,:,:,0]=1.
        logits[0,0,1,2]=3.
        logits[0,1,2,1]=3.
        logits[0,2,0,2]=3.
        out.argument_polarity=logits
        result=rejected_specific_counts(
            out,{"argument_polarity_TRAIN_ONLY":gold},valid)
        self.assertEqual((result["TP"],result["FP"],result["FN"]),(1,1,1))
        self.assertEqual(result["held_REJECTED_positive_support"],2)
        self.assertAlmostEqual(result["precision"],.5)
        self.assertAlmostEqual(result["recall"],.5)

    def test_micro_average_counts_not_mean_of_fold_f1(self):
        def fold(axis,tp,fp,fn):
            return {"axis":axis,
                    "held_TRAIN_metrics":{
                        k:{"TP":tp,"FP":fp,"FN":fn,
                           "TRAIN_INTERNAL_held_positive_support":tp+fn}
                        for k in ("event","directed_role","argument_polarity",
                                  "coreference","event_relation")},
                    "REJECTED_specific_held_TRAIN":{
                        "TP":tp,"FP":fp,"FN":fn,
                        "held_REJECTED_positive_support":tp+fn}}
        rows=[fold("LEXICAL",1,0,0) for _ in range(7)]+[
            fold("LEXICAL",0,4,2)]
        rows += [fold("CONSTRUCTION",2,2,1) for _ in range(32)]
        lex=aggregate(rows,"LEXICAL")["event"]
        syn=aggregate(rows,"CONSTRUCTION")["event"]
        self.assertEqual((lex["TP"],lex["FP"],lex["FN"]),(7,4,2))
        self.assertAlmostEqual(
            lex["TRAIN_INTERNAL_grouped_micro_F1_NOT_independent"],14/(14+4+2))
        self.assertEqual((syn["TP"],syn["FP"],syn["FN"]),(64,64,32))
        rare=aggregate(rows,"LEXICAL")["rejected_specific"]
        self.assertEqual((rare["TP"],rare["FP"],rare["FN"]),(7,4,2))
        self.assertAlmostEqual(rare["recall"],7/9)


if __name__=="__main__":
    unittest.main()
