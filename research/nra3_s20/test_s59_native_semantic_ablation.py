"""S59 source-only research ABI tests (synthetic tensors, NOT J7 or EVAL)."""
import unittest
import torch
from research.nra3_s20.s57_eventfirst_structured_semantic_decoder import EventFirstInputs
from research.nra3_s20.s59_native320_contrastive_event_polarity import (
    SemanticInputs,S59NativeEventSemantic,semantic_features,D,P,
)

class S59OriginalNativeSemanticSynthetic(unittest.TestCase):
    def setup_case(self):
        b,w=2,4
        native=torch.zeros((b,w,339),dtype=torch.float32)
        native[:,1,2]=3.
        native[:,2,3]=4.
        ev=torch.zeros((b,w,41),dtype=torch.float32)
        ev[:,1,0]=1.
        pair=torch.zeros((b,w,w,54),dtype=torch.float32)
        candidate=torch.zeros((b,w),dtype=torch.bool)
        candidate[:,1]=True
        mask=torch.zeros((b,w,w),dtype=torch.bool)
        mask[:,1,2]=True
        x=semantic_features(native,EventFirstInputs(ev,pair,candidate,mask))
        el=torch.tensor([[0.,3.,0.,0.],[0.,3.,0.,0.]])
        tl=torch.tensor([[0.,4.,0.,0.],[0.,-3.,0.,0.]])
        rec=torch.zeros((b,w,w));rec[:,1,2]=5.
        rej=torch.zeros_like(rec);rej[:,1,2]=2.
        return native,x,(el,tl,rec,rej)

    def test_feature_uses_original_native_not_only_event41(self):
        native,x,base=self.setup_case()
        self.assertEqual(x.semantic.shape,(2,4,D))
        self.assertFalse(x.semantic.requires_grad)
        self.assertTrue(float(x.semantic[0,1,:320].norm())>3.9)
        self.assertEqual(x.semantic[0,1,-41:].tolist(),x.event.event[0,1].tolist())

    def test_original_readout_unchanged_and_zero_gate_abstain(self):
        _,x,old=self.setup_case()
        saved=[z.clone() for z in old]
        model=S59NativeEventSemantic().eval().requires_grad_(False)
        ret=model.decode(x,old)
        self.assertEqual(int(ret['directed_recipient'].sum()),1)
        self.assertEqual(int(ret['directed_rejected'].sum()),1)
        self.assertTrue(bool(ret['directed_recipient'][0,1,2]))
        self.assertFalse(bool(ret['directed_recipient'][1].any()))
        self.assertEqual(ret['abstain_nontransfer'],1)
        self.assertEqual(ret['output_gate'],0)
        self.assertFalse(ret['candidate_accepted'])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(old,saved)))

    def test_physical_fp16_exact_roundtrip(self):
        obj=S59NativeEventSemantic()
        with torch.no_grad():
            obj.event[:4]=torch.tensor([.125,-.25,.4,0.01])
            obj.transfer_bias.fill_(1.25)
        physical=obj.fp16()
        self.assertEqual(physical.shape,(P,))
        self.assertEqual(physical.dtype,torch.float16)
        loaded=S59NativeEventSemantic.from_fp16(physical)
        self.assertTrue(torch.equal(loaded.fp16(),physical))
        self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))

    def test_all_three_heads_get_gradient(self):
        _,x,old=self.setup_case()
        obj=S59NativeEventSemantic()
        e,t,r,n=obj(x,old)
        loss=e[:,1].square().mean()+t[:,1].square().mean()
        loss=loss+(n[:,1,2]+x.semantic[:,1]@obj.reject+obj.reject_bias).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in obj.parameters()))

    def test_zero_gate_prohibits_activation(self):
        _,x,old=self.setup_case()
        obj=S59NativeEventSemantic()
        obj.candidate_accepted=True
        with self.assertRaisesRegex(ValueError,'fail-closed'):
            obj.decode(x,old)

    def test_native_features_reject_grad_or_bad_ABI(self):
        n,x,_=self.setup_case()
        n.requires_grad_(True)
        with self.assertRaises(ValueError):
            semantic_features(n,x.event)
        n=torch.zeros(2,4,338)
        with self.assertRaises(ValueError):
            semantic_features(n,x.event)

if __name__=="__main__":unittest.main()
