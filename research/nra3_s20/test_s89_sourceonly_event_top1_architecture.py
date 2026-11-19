"""S89 architectural synthetic-only contracts; NOT trained/real43M scores."""
import ast
import unittest
from pathlib import Path

import torch

from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s89_sourceonly_event_mention_top1_ranker import (
    S89MainEventMentionTop1,PAID,STAGE,
)


def source(texts,verbs):
    words=[raw_surface_grammar(t) for t in texts]
    width=max(len(x) for x in words)
    b=len(texts);k=len(verbs[0])
    valid=torch.zeros((b,width),dtype=torch.bool)
    unsafe=torch.zeros_like(valid)
    cand=torch.zeros((b,k),dtype=torch.long)
    allowed=torch.ones((b,k),dtype=torch.bool)
    scores=torch.randn(b,width)
    for i,(raw,targets) in enumerate(zip(words,verbs)):
        valid[i,:len(raw)]=True
        for j,verb in enumerate(targets):
            loc=[p for p,x in enumerate(raw) if x.form.casefold()==verb]
            if len(loc)!=1:
                raise ValueError(f"synthetic event {verb} missing or ambiguous")
            cand[i,j]=loc[0]
    return valid,unsafe,cand,allowed,scores


class TestS89IndependentMentionTop1(unittest.TestCase):
    def test_5725_new_params_zero_activation_and_eight_old_scores_not_mutated(self):
        m=S89MainEventMentionTop1()
        b=m.budget()
        self.assertEqual(b["new_trainable_parameters"],PAID)
        self.assertEqual(PAID,5725)
        self.assertEqual(b["new_FP16_weight_bytes"],11450)
        self.assertTrue(b["original_S59_parent_and_asserted_S65_unchanged"])
        self.assertTrue(b["not_event_factuality_or_recipient_detection"])
        self.assertFalse(b["candidate_accepted"])
        self.assertFalse(b["activation_allowed"])
        self.assertEqual(b["output_gate"],0)
        self.assertFalse(m.is_trained)
        self.assertEqual(STAGE,"NRA2_S89_SOURCE_ONLY_MENTION_TOP1_EVENT_SELECTOR_SHADOW")

    def test_only_new_weights_receive_gradients_not_old_H8_or_S59(self):
        texts=(
            "Курьер не передал письмо, но водитель доставил посылку.",
            "На стенде написали: «Механик отключил прибор» и инженер сверил показания.",
        )
        keys=(("передал","доставил"),("отключил","сверил"))
        valid,unsafe,idx,allowed,score=source(texts,keys)
        torch.manual_seed(89)
        h=torch.randn(2,valid.shape[1],320)
        initial_h=h.clone();initial_score=score.clone()
        m=S89MainEventMentionTop1()
        out=m(h,valid,unsafe,idx,allowed,score,texts)
        self.assertEqual(out.shape,(2,2))
        out.square().mean().backward()
        self.assertTrue(all(z.grad is not None and torch.isfinite(z.grad).all()
                            for z in m.parameters()))
        self.assertIsNone(h.grad)
        self.assertIsNone(score.grad)
        self.assertTrue(torch.equal(initial_h,h))
        self.assertTrue(torch.equal(initial_score,score))

    def test_invalid_unsafe_original_BPE_candidate_fails_closed(self):
        texts=("Курьер передал конверт.",)
        valid,unsafe,idx,allowed,scores=source(texts,(("передал",),))
        unsafe[0,idx[0,0]]=True
        with self.assertRaisesRegex(ValueError,"valid safe real BPE"):
            S89MainEventMentionTop1()(
                torch.zeros(1,valid.shape[1],320),
                valid,unsafe,idx,allowed,scores,texts)

    def test_all_invalid_row_rejected_and_logits_masked(self):
        texts=("Техник запустил станок.",)
        valid,unsafe,idx,allowed,scores=source(texts,(("запустил",),))
        allowed[0,0]=False
        with self.assertRaisesRegex(ValueError,"valid safe real BPE"):
            S89MainEventMentionTop1()(
                torch.zeros(1,valid.shape[1],320),
                valid,unsafe,idx,allowed,scores,texts)
        allowed[0,0]=True
        idx=torch.cat((idx,torch.zeros_like(idx)),dim=1)
        allowed=torch.tensor([[True,False]])
        out=S89MainEventMentionTop1()(
            torch.zeros(1,valid.shape[1],320),
            valid,unsafe,idx,allowed,scores,texts)
        self.assertEqual(out.shape,(1,2))
        self.assertEqual(out[0,1].item(),-1.e5)

    def test_future_unrelated_sentence_does_not_change_earlier_focus(self):
        text="Курьер передал письмо."
        more=text+" Бухгалтер не подписала акт и сообщила о задержке."
        v1,u1,i1,a1,s1=source((text,),(("передал",),))
        v2,u2,i2,a2,s2=source((more,),(("передал",),))
        self.assertTrue(torch.equal(i1,i2))
        torch.manual_seed(891)
        h1=torch.randn(1,v1.shape[1],320)
        h2=torch.randn(1,v2.shape[1],320)
        h2[:,:v1.shape[1],:]=h1
        s2[:,:v1.shape[1]]=s1
        model=S89MainEventMentionTop1().eval()
        with torch.inference_mode():
            a=model(h1,v1,u1,i1,a1,s1,(text,))
            b=model(h2,v2,u2,i2,a2,s2,(more,))
        self.assertTrue(torch.equal(a,b))

    def test_gate_mutation_cannot_activate_model(self):
        text="Эксперт заверил акт."
        valid,unsafe,idx,allowed,scores=source((text,),(("заверил",),))
        model=S89MainEventMentionTop1()
        model.output_gate=1
        with self.assertRaisesRegex(ValueError,"cannot activate"):
            model(torch.zeros(1,valid.shape[1],320),
                  valid,unsafe,idx,allowed,scores,(text,))

    def test_source_module_never_imports_S86_S79_or_gold(self):
        path=Path("research/nra3_s20/s89_sourceonly_event_mention_top1_ranker.py")
        t=ast.parse(path.read_text(encoding="utf8"))
        imports=[z.module or "" for z in t.body if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("gold" in z.lower() or
                             "s86" in z.lower() or
                             "s79" in z.lower() or
                             "s88" in z.lower() for z in imports))
        self.assertTrue(any("s82_predicate_centered" in z for z in imports))


if __name__=="__main__":
    unittest.main()
