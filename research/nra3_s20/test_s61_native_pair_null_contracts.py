"""S61 six safe source-only synthetic contracts, no new HELD import."""
import unittest
import torch
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import EventFirstInputs
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    S59NativeEventSemantic,semantic_features,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    source_local_scope,
)
from research.nra3_s20.s61_native_pair_null_polarity_2752 import (
    S61NativePairNull,PAID,PAIR_D,NULL_D,
)

def get_case():
    raws=("Оператор не передал отчёт","Оператор передал новый отчёт")
    b,w=2,4
    original=torch.zeros((b,w,339),dtype=torch.float32)
    original[:,2,0]=3.
    original[:,3,4]=2.
    events=torch.zeros((b,w,41));events[:,2,0]=1.
    pair=torch.zeros((b,w,w,54));pair[:,2,3,0]=1.
    can=torch.zeros((b,w),dtype=torch.bool);can[:,2]=True
    valid=torch.zeros((b,w,w),dtype=torch.bool);valid[:,2,3]=True
    event=EventFirstInputs(events,pair,can,valid)
    sem=semantic_features(original,event)
    x=source_local_scope(sem,raws)
    e=torch.zeros((b,w));e[:,2]=3.
    t=torch.zeros((b,w));t[:,2]=2.
    r=torch.zeros((b,w,w));r[:,2,3]=3.
    n=torch.zeros_like(r);n[:,2,3]=1.
    old59=S59NativeEventSemantic().eval().requires_grad_(False)
    return x,(e,t,r,n),old59

class TestS61PairNull(unittest.TestCase):
    def test_physical_reloaded_weights_exact(self):
        m=S61NativePairNull()
        with torch.no_grad():
            m.pair[12]=.125
            m.null_bias[:]=.5
            m.polar[7]=-.01
        z=m.fp16()
        self.assertEqual(PAID,2752)
        self.assertEqual(PAIR_D,1028)
        self.assertEqual(NULL_D,693)
        self.assertEqual(z.shape,(PAID,))
        r=S61NativePairNull.from_fp16(z)
        self.assertTrue(torch.equal(r.fp16(),z))
        self.assertTrue(all(not p.requires_grad for p in r.parameters()))

    def test_one_valid_pair_and_original_models_immutable(self):
        x,base,old59=get_case()
        saved=[z.clone() for z in base]
        m=S61NativePairNull().eval().requires_grad_(False)
        output=m.decode(x,base,old59)
        self.assertEqual(int(output['directed_recipient'].sum()),2)
        self.assertTrue(bool(output['directed_recipient'][0,2,3]))
        self.assertTrue(bool(output['directed_rejected'][1,2,3]))
        self.assertEqual(output['output_gate'],0)
        self.assertFalse(output['candidate_accepted'])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(base,saved)))

    def test_no_valid_target_forces_none(self):
        x,base,old59=get_case()
        mask=x.base.event.valid_pair.clone()
        mask[:]=False
        from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import EventFirstInputs
        from research.nra3_s20.s59_native320_contrastive_event_polarity import SemanticInputs
        from research.nra3_s20.s60_native43m_local_triclass_model import S60Input
        ev=x.base.event
        replace=EventFirstInputs(ev.event,ev.pair,ev.candidate_event,mask)
        sem=SemanticInputs(replace,x.base.semantic)
        no=S60Input(sem,x.scope,x.fused)
        out=S61NativePairNull().eval().requires_grad_(False).decode(no,base,old59)
        self.assertEqual(int(out['directed_recipient'].sum()),0)
        self.assertEqual(out['no_target_abstentions'],2)

    def test_learned_null_can_suppress_all_edges(self):
        x,base,old59=get_case()
        m=S61NativePairNull()
        with torch.no_grad():
            m.null_bias.fill_(100.)
        m.eval().requires_grad_(False)
        out=m.decode(x,base,old59)
        self.assertEqual(out['no_target_abstentions'],2)
        self.assertEqual(int(out['directed_rejected'].sum()),0)

    def test_joint_gradient_pair_null_polarity(self):
        x,base,old59=get_case()
        m=S61NativePairNull()
        source=torch.tensor([2,2],dtype=torch.long)
        scores,polar,mask=m.score_sources(x,base,old59,source)
        loss=torch.nn.functional.cross_entropy(scores,torch.tensor([3,4]))
        loss=loss+torch.nn.functional.binary_cross_entropy_with_logits(
            polar[:,3],torch.tensor([1.,0.]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in m.parameters()))

    def test_cannot_activate_or_mutate_original_input(self):
        x,base,old59=get_case()
        oldcopy=x.fused.clone()
        model=S61NativePairNull()
        model.activation_allowed=True
        with self.assertRaisesRegex(ValueError,'gate fail-closed'):
            model.decode(x,base,old59)
        self.assertTrue(torch.equal(x.fused,oldcopy))

if __name__=="__main__":unittest.main()
