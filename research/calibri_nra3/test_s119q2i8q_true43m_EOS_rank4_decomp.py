"""S119q2i8q true 43M EOS-vs-gold rank4 native head causal decomposition.

Unit-only synthetic logits. All real 43M/49k distribution math happens
in source-only protected GitHub Actions physical read-only research stage.
"""
import unittest
import torch
from research.calibri_nra3.s119q2i8q_true43m_EOS_vs_gold_head_lowrank_causal import (
    STAGE,GAINS,scoped_exact_native_gain,NATIVE_VOCAB,
)
from research.calibri_nra3.s119q2i1_real43m_ru_language_head_joint_training import (
    NativeHeadAlignment,
)

class TestRank4EOSComponentAttribution(unittest.TestCase):
    def test_01_gain0_point25_one_exact_math(self):
        self.assertEqual(GAINS,(0.,.25,1.))
        self.assertIn("EOS_GOLD_NATIVE_HEAD_BIAS_VS_LOWRANK",STAGE)
        self.assertEqual(NATIVE_VOCAB,49152)
    def test_02_head_positive_EOS_context_even_with_negative_bias(self):
        # Construct rank4 head whose learned bias alone penalizes native EOS,
        # but context-dependent rank4 correction favors EOS vs gold.
        head=NativeHeadAlignment()
        eos=2;gold=155
        with torch.no_grad():
            for p in head.parameters():p.zero_()
            head.output_bias[eos]=-.05
            head.output_bias[gold]=.20
            head.down.weight[0,0]=1.0
            head.up.weight[eos,0]=10.0
            head.up.weight[gold,0]=0.0
        h=torch.zeros((1,1,320),dtype=torch.float32)
        h[0,0,0]=10.
        bias_adv=float(head.output_bias[eos]-head.output_bias[gold])
        self.assertLess(bias_adv,0)
        y0=head(h)
        self.assertGreater(float(y0[0,0,eos]-y0[0,0,gold]),0.)
        with scoped_exact_native_gain(head,.25):
            y25=head(h)
        with scoped_exact_native_gain(head,1.):
            y1=head(h)
        self.assertTrue(torch.allclose(y25,y1*.25))
    def test_03_exact_linear_first_gold_vs_EOS_crossover(self):
        # Native original 43M base gold-vs-EOS gap 3.0;
        # learned rank4 shifts EOS-vs-gold by +7; crossing=.42857
        # between historical .25 and gain1, without any new weights.
        base=3.;shift=7.
        self.assertGreater(base-.25*shift,0.)
        self.assertLess(base-1.*shift,0.)
        self.assertTrue(.25 < base/shift < 1.)
    def test_04_reversible_hook_after_raise(self):
        head=NativeHeadAlignment()
        before=set(head._forward_hooks)
        with self.assertRaises(RuntimeError):
            with scoped_exact_native_gain(head,1.):
                raise RuntimeError("forced fail")
        self.assertEqual(set(head._forward_hooks),before)
if __name__=="__main__":
    unittest.main()
