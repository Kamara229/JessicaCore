"""Six model-free synthetic S60 structure tests, no new gold or EVAL claims."""
import unittest
import torch
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import EventFirstInputs
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    SemanticInputs,S59NativeEventSemantic,semantic_features,
)
from research.nra3_s20.s60_native43m_local_triclass_model import (
    source_local_scope,S60Joint3,P,SCOPE_D,
)

def case():
    b,w=2,4
    raw=("Оператор не передал ключ","Оператор передал новый ключ")
    x=torch.zeros((b,w,339),dtype=torch.float32)
    x[:,2,13]=2.
    x[:,3,39]=3.
    ev=torch.zeros((b,w,41),dtype=torch.float32)
    ev[:,2,4]=1.
    pair=torch.zeros((b,w,w,54),dtype=torch.float32)
    can=torch.zeros((b,w),dtype=torch.bool)
    can[:,2]=True
    mask=torch.zeros((b,w,w),dtype=torch.bool)
    mask[:,2,3]=True
    base=EventFirstInputs(ev,pair,can,mask)
    sem=semantic_features(x,base)
    fused=source_local_scope(sem,raw)
    e=torch.zeros((b,w));e[:,2]=3.
    t=torch.zeros((b,w));t[:,2]=2.
    r=torch.zeros((b,w,w));r[:,2,3]=4.
    n=torch.zeros_like(r);n[:,2,3]=1.
    old=S59NativeEventSemantic().eval().requires_grad_(False)
    return fused,(e,t,r,n),old

class TestS60Joint3(unittest.TestCase):
    def test_local_negation_signal_is_source_only(self):
        x,old,prev=case()
        self.assertEqual(x.fused.shape,(2,4,693))
        self.assertEqual(x.scope.shape,(2,4,SCOPE_D))
        self.assertEqual(float(x.scope[0,2,0]),1.)
        self.assertEqual(float(x.scope[1,2,0]),0.)
        self.assertFalse(x.fused.requires_grad)

    def test_direction_and_at_most_one_pair(self):
        x,old,prev=case()
        out=S60Joint3().eval().requires_grad_(False).decode(x,old,prev)
        self.assertLessEqual(int(out['directed_recipient'].sum()),2)
        self.assertFalse(bool((out['directed_recipient']&
                               ~x.base.event.valid_pair).any()))
        self.assertFalse(bool((out['directed_rejected']&
                               ~out['directed_recipient']).any()))
        self.assertEqual(out['output_gate'],0)
        self.assertFalse(out['activation_allowed'])

    def test_fp16_is_physically_reloaded_identical(self):
        obj=S60Joint3()
        with torch.no_grad():
            obj.event[3]=.125
            obj.classes[2,12]=-.5
        p=obj.fp16()
        self.assertEqual(tuple(p.shape),(P,))
        self.assertTrue(torch.equal(p,S60Joint3.from_fp16(p).fp16()))

    def test_all_joint_heads_receive_grad(self):
        x,old,prev=case()
        model=S60Joint3()
        ev,cl,_=model(x,old,prev)
        loss=ev[:,2].square().mean()+cl[:,2].square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and
                            torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_gate_cannot_activate(self):
        x,old,prev=case()
        obj=S60Joint3()
        obj.candidate_accepted=True
        with self.assertRaisesRegex(ValueError,'gate must remain zero'):
            obj.decode(x,old,prev)

    def test_incompatible_native_denied(self):
        x,old,prev=case()
        altered=x.base.semantic.detach().requires_grad_(True)
        with self.assertRaises(ValueError):
            source_local_scope(SemanticInputs(x.base.event,altered),
                               ("Оператор не передал ключ",
                                "Оператор передал новый ключ"))

if __name__=="__main__":unittest.main()
