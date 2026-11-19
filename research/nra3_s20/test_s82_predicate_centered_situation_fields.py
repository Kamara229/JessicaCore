"""S82 source-only event-local clause field, safety and eight experts.

Synthetic original 320D tensor STAND-IN for architecture unit test, NOT
physical 43M evidence and NOT independent generalization measurement.
"""
from __future__ import annotations
import ast
import unittest
from pathlib import Path

import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s82_predicate_centered_situation_fields import (
    S82PredicateCenteredMiniAreas,source_only_event_fields,
    STAGE,WIDTH,FIELDS,AREAS,
)


def prepare(texts,locators,width=None):
    words=[raw_surface_grammar(t) for t in texts]
    if width is None:width=max(len(w) for w in words)
    b=len(texts)
    valid=torch.zeros((b,width),dtype=torch.bool)
    unsafe=torch.zeros_like(valid)
    idx=torch.zeros((b,1),dtype=torch.long)
    allowed=torch.ones((b,1),dtype=torch.bool)
    for i,(tokens,look) in enumerate(zip(words,locators)):
        valid[i,:len(tokens)]=True
        target=[j for j,t in enumerate(tokens)
                if t.form.casefold()==look.casefold()]
        if not target:
            raise ValueError("test event missing")
        idx[i,0]=target[-1]
    return valid,unsafe,idx,allowed


class TestS82PredicateCenteredFields(unittest.TestCase):
    def test_eight_independent_experts_small_budget_gate_off(self):
        m=S82PredicateCenteredMiniAreas()
        b=m.budget()
        self.assertEqual(len(AREAS),8)
        self.assertEqual(WIDTH,len(FIELDS))
        self.assertEqual(WIDTH,32)
        self.assertEqual(b["new_neural_parameters"],8264)
        self.assertEqual(b["standalone_FP16_bytes"],16528)
        self.assertTrue(b["within_10000_new_params_budget"])
        self.assertEqual(len({id(z.net[0].weight) for z in m.areas}),8)
        self.assertFalse(m.is_trained)
        self.assertFalse(m.activation_allowed)
        self.assertFalse(m.candidate_accepted)
        self.assertEqual(m.output_gate,0)

    def test_quote_and_negation_are_distinct_for_same_action(self):
        texts=(
            "Курьер передал Марии конверт.",
            "Отчёт гласит: «Курьер передал Марии конверт».",
            "Курьер не передал Марии конверт.",
            "Сотрудник предположил, будто курьер передал Марии конверт.",
        )
        valid,unsafe,ids,allowed=prepare(texts,("передал",)*len(texts))
        v=source_only_event_fields(texts,valid,unsafe,ids,allowed)
        f={q:i for i,q in enumerate(FIELDS)}
        self.assertEqual(v.shape,(4,1,32))
        self.assertEqual(v[0,0,f["inside_literal_quote"]].item(),0)
        self.assertEqual(v[1,0,f["inside_literal_quote"]].item(),1)
        self.assertEqual(v[2,0,f["left_immediate_negator"]].item(),1)
        self.assertEqual(v[0,0,f["left_immediate_negator"]].item(),0)
        self.assertEqual(v[3,0,f["dream_counterfactual_same_clause"]].item(),1)

    def test_negation_in_other_clause_does_not_negate_current_event(self):
        text="Мастер заменил деталь, однако механик не отключил питание."
        valid,unsafe,ids,allowed=prepare((text,text),("заменил","отключил"))
        v=source_only_event_fields((text,text),valid,unsafe,ids,allowed)
        f={q:i for i,q in enumerate(FIELDS)}
        self.assertEqual(v[0,0,f["left_immediate_negator"]].item(),0)
        self.assertEqual(v[0,0,f["right_near_negator"]].item(),0)
        self.assertEqual(v[1,0,f["left_immediate_negator"]].item(),1)
        self.assertEqual(v[1,0,f["negator_away_outside_clause"]].item(),0)

    def test_append_irrelevant_future_clause_no_field_or_head_contamination(self):
        text="Курьер передал Марии конверт."
        added=text+" Бухгалтер не подписала акт и сказала о задержке."
        valid1,unsafe1,idx1,ok1=prepare((text,),("передал",))
        valid2,unsafe2,idx2,ok2=prepare((added,),("передал",))
        features1=source_only_event_fields((text,),valid1,unsafe1,idx1,ok1)
        features2=source_only_event_fields((added,),valid2,unsafe2,idx2,ok2)
        self.assertTrue(torch.equal(features1,features2))
        torch.manual_seed(82)
        m=S82PredicateCenteredMiniAreas().eval()
        h1=torch.randn(1,valid1.shape[1],320)
        h2=torch.randn(1,valid2.shape[1],320)
        # A frozen prefix representation matches; original real43M
        # hidden-state causal property is tested separately, NOT here.
        h2[:,:h1.shape[1]]=h1
        with torch.inference_mode():
            a=m(h1,valid1,unsafe1,idx1,ok1,(text,))
            b=m(h2,valid2,unsafe2,idx2,ok2,(added,))
        self.assertTrue(torch.equal(a,b))

    def test_native_original_bpe_unsafe_event_fails_closed(self):
        text="Техник выдал заключение."
        valid,unsafe,idx,allowed=prepare((text,),("выдал",))
        unsafe[0,idx[0,0]]=True
        with self.assertRaisesRegex(ValueError,"unsafe original BPE candidate"):
            source_only_event_fields((text,),valid,unsafe,idx,allowed)

    def test_source_only_forward_gradient_through_eight_heads_only(self):
        text="Документы подписаны руководителем после проверки."
        valid,unsafe,idx,allowed=prepare((text,),("подписаны",))
        torch.manual_seed(83)
        m=S82PredicateCenteredMiniAreas()
        h=torch.randn(1,valid.shape[1],320)
        y=m(h,valid,unsafe,idx,allowed,(text,))
        self.assertEqual(y.shape,(1,1,8))
        self.assertTrue(torch.isfinite(y).all())
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in m.parameters()))
        self.assertIsNone(h.grad)

    def test_source_features_never_import_gold_train_eval_or_s81_scores(self):
        p=Path("research/nra3_s20/s82_predicate_centered_situation_fields.py")
        t=ast.parse(p.read_text())
        imports=[]
        for z in t.body:
            if isinstance(z,ast.Import):
                imports.extend(t.name for t in z.names)
            elif isinstance(z,ast.ImportFrom):
                imports.append(z.module or "")
        self.assertFalse(any("s79" in z.lower() or "s81" in z.lower()
                             or "gold" in z.lower()
                             or "eval" in z.lower()
                             for z in imports))
        self.assertIn("research.nra3_s18.s18c3_raw_russian_grammar_cues",
                      imports)


if __name__=="__main__":
    unittest.main()
