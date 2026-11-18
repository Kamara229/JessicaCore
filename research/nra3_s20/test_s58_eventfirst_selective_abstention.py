"""Synthetic contract tests for S58. No Calibri/J7/HELD quality claims."""
import unittest
import torch
from research.nra3_s20.s58_eventfirst_selective_abstention import (
    GuardInput, selective_abstain,
)


def fixture():
    e=torch.tensor([[3.,2.,0.,0.],[3.,2.,0.,0.],[3.,2.,0.,0.]])
    t=torch.zeros((3,4)); t[:,1]=2.
    rec=torch.zeros((3,4,4)); rec[:,1,2]=4.
    rej=torch.zeros_like(rec); rej[:,1,2]=3.
    ev=torch.zeros((3,4),dtype=torch.bool); ev[:,0:3]=True
    verb=torch.zeros_like(ev); verb[:,1]=True
    pairs=torch.zeros((3,4,4),dtype=torch.bool); pairs[:,1,2]=True
    same=pairs.clone()
    neg=torch.zeros_like(ev); neg[0,1]=True
    return GuardInput(e,t,rec,rej,ev,pairs,verb,same,neg)


class S58SyntheticContract(unittest.TestCase):
    def test_high_scoring_noun_cannot_override_verb_candidate(self):
        y=selective_abstain(fixture())
        self.assertEqual(y["chosen_event_minus1_on_abstain"].tolist(),[1,1,1])
        self.assertEqual(int(y["directed_recipient"].sum()),3)
        self.assertEqual(int(y["directed_rejected"].sum()),1)
        self.assertEqual(y["reasons"]["unlocalized_rejection_suppressed"],2)
        self.assertEqual((y["output_gate"],y["activation_allowed"]),(0,False))

    def test_missing_verb_abstains(self):
        x=fixture();x.surface_verb[0,:]=False
        y=selective_abstain(x)
        self.assertEqual(int(y["chosen_event_minus1_on_abstain"][0]),-1)
        self.assertFalse(bool(y["directed_recipient"][0].any()))

    def test_nontransfer_abstains(self):
        x=fixture();x.transfer[0,1]=-1
        y=selective_abstain(x)
        self.assertEqual(y["reasons"]["not_transfer"],1)
        self.assertFalse(bool(y["directed_recipient"][0].any()))

    def test_cross_clause_or_unsafe_target_abstains(self):
        x=fixture()
        x.same_clause[1,1,2]=False
        x.pair_safe[2,1,2]=False
        y=selective_abstain(x)
        self.assertEqual(y["reasons"]["no_same_clause_safe_target"],2)
        self.assertEqual(int(y["directed_recipient"].sum()),1)

    def test_event_ineligible_source_cannot_emit(self):
        x=fixture();x.event_safe[0,1]=False
        y=selective_abstain(x)
        self.assertFalse(bool(y["directed_recipient"][0].any()))

    def test_nonfinite_or_gradient_logit_fails_closed(self):
        x=fixture();x.event[0,0]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            selective_abstain(x)
        x=fixture();x.event.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,"trainable"):
            selective_abstain(x)


if __name__=="__main__":
    unittest.main()
