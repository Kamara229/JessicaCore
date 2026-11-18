"""S36 source-only contract; fake scores are NOT original43M CPU test evidence."""
from __future__ import annotations

import inspect
import unittest

import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import (
    FEATURES,raw_surface_grammar,
)
from research.nra3_s19.s19g_paid_semantic_graph import GraphLogits
from research.nra3_s20.s22_new256_counterfactual_train_graph_source import (
    build_s22_train256,splits,
)
from research.nra3_s20.s36_genuine43m_fold34_fp_pair_audit import (
    describe_fp_pairs,FOLD,HELD_CONSTRUCTION,
)


class S36FrozenModelTrainingOnlyPairAudit(unittest.TestCase):
    def _fake_eight_held_examples(self):
        data=build_s22_train256()
        held=tuple(26+32*i for i in range(8))
        rows=[data[i] for i in held]
        n=max(len(row.tokens) for row in rows)
        x=torch.zeros(8,n,338)
        valid=torch.zeros(8,n,dtype=torch.bool)
        gold={"argument_polarity_TRAIN_ONLY":
              torch.full((8,n,n),-1,dtype=torch.long)}
        roles=torch.zeros(8,n,n,9)
        events=torch.zeros(8,n,6)
        polarity=torch.zeros(8,n,n,3)
        polarity[...,0]=1.
        for k,row in enumerate(rows):
            toks=raw_surface_grammar(row.text)
            valid[k,:len(toks)]=True
            for j,tok in enumerate(toks):
                x[k,j,320:]=torch.tensor(tok.features,dtype=torch.float32)
            src=next(e.predicate-1 for e in row.events)
            targ=next(a.token-1 for a in row.arguments if a.polarity=="REJECTED")
            gold["argument_polarity_TRAIN_ONLY"][k,src,targ]=1
            polarity[k,src,targ,2]=12.
            if k<7:
                # An explicitly wrong source points at the same true target.
                wrong=next(i for i in range(len(toks)) if i not in (src,targ))
                polarity[k,wrong,targ,2]=12.
        out=GraphLogits(
            events,roles,polarity,torch.zeros(8,n,n,2),
            torch.zeros(8,n,n,5))
        return out,x,valid,gold,held,data

    def test_offline_classification_seven_non_event_false_sources_to_real_rejected(self):
        out,x,valid,gold,held,data=self._fake_eight_held_examples()
        rows,metrics=describe_fp_pairs(out,x,valid,gold,held,data)
        self.assertEqual(len(rows),8)
        self.assertEqual(metrics["TP"],8)
        self.assertEqual(metrics["FP"],7)
        self.assertEqual(metrics["FN"],0)
        self.assertEqual(
            metrics["FP_wrong_non_event_source"],7)
        self.assertEqual(sum(len(z["false_positive_pairs"]) for z in rows),7)
        self.assertEqual(rows[7]["false_positive_pairs"],[])

    def test_strict_real_fold_and_no_gold_in_inference(self):
        self.assertEqual((FOLD,HELD_CONSTRUCTION),(34,26))
        axis,group,tr,held=splits()[FOLD]
        self.assertEqual((axis,group,len(tr),len(held)),
                         ("CONSTRUCTION",26,248,8))
        self.assertTrue(set(tr).isdisjoint(held))
        import research.nra3_s20.s36_genuine43m_fold34_fp_pair_audit as mod
        source=inspect.getsource(mod.main)
        self.assertIn("model=new_graph_model(",source)
        self.assertIn("candidate(held_x,held_v)",source)
        self.assertIn("apply_local_ne_dative_scope(pred,held_x,held_v)",source)
        self.assertIn("describe_fp_pairs(",source)
        # Literal network forward signature is raw, valid, no gold.
        self.assertNotIn("candidate(held_x,held_v,held_g)",source)
        self.assertNotIn("apply_local_ne_dative_scope(pred,held_x,held_v,held_g)",source)
        self.assertIn("no_new_model_candidate_or_decision_rule_selected",source)


if __name__=="__main__":
    unittest.main()
