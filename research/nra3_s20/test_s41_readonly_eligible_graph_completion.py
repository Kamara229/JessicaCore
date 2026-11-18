"""S41 read-only structural eligibility completion contracts (no real43M CPU).

A rule-informed synthetic TRAIN development test, NOT independent EVAL.
"""
from __future__ import annotations
import inspect
import unittest
import torch

from research.nra3_s20.s39_new80_trainonly_scope_graph_source import (
    build_train80,validate_source,partitions,
)
from research.nra3_s20.s40_compositional_predicate_negation_graph import (
    CompositionalNRA2S40,plain_scope_features,
)
from research.nra3_s20.s41_typed_graph_edge_completion import (
    CompositionalNRA2S41,structurally_complete_REJECTED,
    S41_ELIGIBLE_LOWER_LOGIT,S41_RULE,
)
from research.nra3_s20.s41_physical_paired_train80_edge_completion_18fold import (
    build_batch,TRAIN_STEPS,TRAIN_LR,
)

class S41ReadOnlyS40GraphCompletionContracts(unittest.TestCase):
    def test_all_original_positive_train_edges_have_universal_structural_eligibility(self):
        source=validate_source()
        self.assertEqual(source["TRAIN_rows"],80)
        self.assertEqual(source["positive"],56)
        self.assertEqual(source["negative"],24)
        rows=build_train80()
        yes=no=0
        for row in rows:
            structural=plain_scope_features(row.text)
            eligible=bool(structural[row.source_predicate_token_index0,
                                    row.potential_recipient_token_index0,11].item())
            if row.rejected_recipient:
                yes+=eligible
            else:
                no+=eligible
        self.assertEqual((yes,no),(56,0))
        self.assertEqual(len(partitions()),18)

    def test_model_weights_and_protected_heads_identical_in_rule_stage(self):
        rows=build_train80()[:10]
        x,q,p,structure,valid,legal,ev,quotes,links=build_batch(rows)
        torch.manual_seed(411118)
        learned=CompositionalNRA2S40().eval()
        initial={key:value.detach().clone() for key,value in learned.state_dict().items()}
        with torch.inference_mode():
            raw=learned(x,q,p,structure)
            refined=structurally_complete_REJECTED(raw,structure)
        eligible=(structure[...,11]>0.5)
        self.assertTrue(bool(eligible.any()))
        self.assertTrue(torch.equal(raw.event_logits,refined.event_logits))
        self.assertTrue(torch.equal(raw.quote_logits,refined.quote_logits))
        self.assertTrue(torch.equal(raw.rejected_logits[~eligible],
                                    refined.rejected_logits[~eligible]))
        self.assertTrue(bool((refined.rejected_logits[eligible]>=S41_ELIGIBLE_LOWER_LOGIT).all()))
        self.assertTrue(all(torch.equal(v,learned.state_dict()[k])
                            for k,v in initial.items()))
        self.assertEqual((S41_ELIGIBLE_LOWER_LOGIT,S41_RULE),
                         (.25,"generic_directed_grammatical_eligible_edge_floor_for_all_predicate_types"))

    def test_no_new_learned_parameters_and_inactive_shadow(self):
        parent=CompositionalNRA2S40()
        child=CompositionalNRA2S41()
        self.assertEqual(sum(t.numel() for t in parent.parameters()),
                         sum(t.numel() for t in child.parameters()))
        self.assertEqual(child.output_gate,0)
        self.assertFalse(child.candidate_accepted)
        self.assertFalse(child.activation_allowed)
        self.assertEqual((TRAIN_STEPS,TRAIN_LR),(96,.018))

    def test_inference_no_gold_construction_blacklist_or_S38_tuning(self):
        import research.nra3_s20.s41_physical_paired_train80_edge_completion_18fold as trainer
        model_source=inspect.getsource(structurally_complete_REJECTED)
        for bad in ("gold[","held_ids","PATTERNS[","group_id","source_predicate_token_index0",
                    "quoted_target","S38","s38_","passive","infinitive"):
            self.assertNotIn(bad,model_source)
        train_source=inspect.getsource(trainer)
        self.assertNotIn("from research.nra3_s20.s38_",train_source)
        self.assertIn("metrics40=evaluate_after_training",train_source)
        self.assertIn("postprocess=True",train_source)
        self.assertIn("model_weights_digest(model)==before_sha",train_source)


if __name__=="__main__":
    unittest.main()
