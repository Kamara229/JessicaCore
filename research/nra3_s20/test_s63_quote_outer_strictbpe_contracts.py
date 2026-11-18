"""Six S63 source-only quote, ORIGINAL-BPE-unsafe abstention, FP16 contracts.

Synthetic tensors only: NO original 616 J7 or population EVAL claim.
"""
import unittest
import torch
from research.nra3_s18.s18c3_raw_russian_grammar_cues import raw_surface_grammar
from research.nra3_s20.s62_joint_event_target_beam2166 import (
    Beam,S62JointEventRecipient,K,PAIR_D,SOURCE_D,
)
from research.nra3_s20.s63_quote_outer_strictbpe_residual471 import (
    raw_quote_spans,quote_overlay,S63QuoteOuterSafe,FP16,Q,JOINT_D,
)

TEXTS=(
 "Редактор повторил «Мастеру выдали письмо», затем отправил директору акт.",
 "Редактор прочитал “Курьеру передали пакет” и отправил секретарю журнал.",
)

def case():
    n=len(TEXTS)
    words=tuple(raw_surface_grammar(t) for t in TEXTS)
    w=max(len(z) for z in words)
    src=torch.zeros((n,K),dtype=torch.long)
    reporter=("повторил","прочитал")
    transferred=("отправил","отправил")
    recipients=("директору","секретарю")
    safe=torch.zeros((n,K,w),dtype=torch.bool)
    target=torch.full((n,K,w+1),-1e9)
    polar=torch.zeros((n,K,w))
    unsafe=torch.zeros((n,w),dtype=torch.bool)
    enabled=torch.zeros((n,K),dtype=torch.bool)
    enabled[:,:2]=True
    base_event=torch.zeros((n,K))
    for i,tokens in enumerate(words):
        find=lambda s: next(j for j,z in enumerate(tokens) if z.form==s)
        src[i,0]=find(reporter[i])
        src[i,1]=find(transferred[i])
        dest=find(recipients[i])
        safe[i,1,dest]=True
        target[i,0,w]=0.
        target[i,1,dest]=3.
        target[i,1,w]=-2.
        polar[i,1,dest]=2.
    features=torch.zeros((n,K,w,PAIR_D),dtype=torch.float32)
    beam=Beam(src,enabled,base_event,torch.zeros((n,K,SOURCE_D)),features,
              target,polar,safe,w)
    return beam,unsafe

class TestS63QuoteScope(unittest.TestCase):
    def test_all_four_quote_delimiters_balanced(self):
        for text in ("«А»","“А”","„А“","‘А’",'"А"'):
            spans,bad=raw_quote_spans(text)
            self.assertEqual(len(spans),1)
            self.assertFalse(bad)
        self.assertTrue(raw_quote_spans("«Нет")[1])

    def test_raw_quote_outer_event_source_only(self):
        beam,unsafe=case()
        x=quote_overlay(beam,TEXTS,unsafe)
        self.assertEqual(x.source_quote.shape,(2,K,Q))
        self.assertEqual(x.pair_quote.shape,(2,K,beam.word_count,JOINT_D))
        self.assertEqual(float(x.source_quote[0,0,4]),1.) # reporter verb
        self.assertEqual(float(x.source_quote[0,1,6]),1.) # outer transfer
        self.assertEqual(float(x.source_quote[0,0,8]),1.) # before quote later transfer
        self.assertFalse(bool(x.unsafe_row.any()))

    def test_strict_native_unsafe_abstains_silent_without_oracle(self):
        beam,unsafe=case()
        unsafe[1,1]=True
        x=quote_overlay(beam,TEXTS,unsafe)
        old=S62JointEventRecipient().eval().requires_grad_(False)
        m=S63QuoteOuterSafe().eval().requires_grad_(False)
        r=m.decode(x,old)
        self.assertEqual(int(r["directed_recipient"].sum()),1)
        self.assertTrue(bool(r["directed_recipient"][0].any()))
        self.assertFalse(bool(r["directed_recipient"][1].any()))
        self.assertEqual(r["native_ambiguous_abstentions"],1)
        self.assertEqual(r["output_gate"],0)
        self.assertFalse(r["streaming_causality_certified"])

    def test_quote_scope_can_change_global_event(self):
        beam,unsafe=case()
        beam.event_base[:,0]=5.
        x=quote_overlay(beam,TEXTS,unsafe)
        old=S62JointEventRecipient().eval().requires_grad_(False)
        m=S63QuoteOuterSafe()
        with torch.no_grad():
            m.event[6]=10. # source after closing quote transfer
        r=m.eval().requires_grad_(False).decode(x,old)
        self.assertTrue(bool(r["directed_recipient"][0].any()))
        self.assertEqual(int(r["chosen_event"][0]),int(beam.sources[0,1]))

    def test_physical_FP16_471_exact_and_no_activation(self):
        m=S63QuoteOuterSafe()
        with torch.no_grad():
            m.event[6]=1.25
            m.pair[2]=-.125
        z=m.fp16()
        self.assertEqual(tuple(z.shape),(471,))
        self.assertEqual(FP16,471)
        frozen=S63QuoteOuterSafe.from_fp16(z)
        self.assertTrue(torch.equal(frozen.fp16(),z))
        self.assertTrue(all(not p.requires_grad for p in frozen.parameters()))
        beam,unsafe=case()
        frozen.candidate_accepted=True
        with self.assertRaisesRegex(ValueError,"fail closed"):
            frozen.decode(quote_overlay(beam,TEXTS,unsafe),
                          S62JointEventRecipient().eval().requires_grad_(False))

    def test_all_quote_residual_heads_get_grad(self):
        beam,unsafe=case()
        x=quote_overlay(beam,TEXTS,unsafe)
        m=S63QuoteOuterSafe()
        old=S62JointEventRecipient().eval().requires_grad_(False)
        y,p=m(x,old)
        w=beam.word_count
        tgt=torch.tensor([int(torch.where(beam.target_safe[i,1])[0][0])
                          for i in range(2)])
        ix=w+1+tgt
        loss=torch.nn.functional.cross_entropy(
            y.reshape(2,-1),ix)
        loss=loss+torch.nn.functional.binary_cross_entropy_with_logits(
            p[torch.arange(2),1,tgt],torch.tensor([1.,0.]))
        loss.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all()
                            for v in m.parameters()))
if __name__=="__main__":unittest.main()
