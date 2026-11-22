"""S119q2i8p native rank4 49152 output_bias vs EOS physical FP16 contracts."""
import unittest
import numpy as np
from research.calibri_nra3.s119q2i8p_fp16_native_head_eos_bias import (
    head_report,real_head_only,PREFIX,VOCAB,RANK,HIDDEN,EOS,
)

def sample(head_bias=.5):
    bias=np.zeros(VOCAB,dtype=np.float16)
    bias[EOS]=head_bias
    up=np.zeros((VOCAB,RANK),dtype=np.float16)
    up[EOS,:]=.25
    down=np.ones((RANK,HIDDEN),dtype=np.float16)*.01
    return {
        PREFIX+"language_head.output_bias":bias,
        PREFIX+"language_head.up.weight":up,
        PREFIX+"language_head.down.weight":down,
    }
class TestPhysicalHeadBias(unittest.TestCase):
    def test_01_true_saved_full_native_eos_bias_rank(self):
        m=head_report(sample())
        self.assertEqual(m["bias_EOS_rank_among_49152_descending"],1)
        self.assertEqual(m["bias_EOS_value_fp16_cast_fp32"],.5)
        self.assertEqual(m["bias_EOS_effect_gain_0p25"],.125)
        self.assertEqual(m["bias_EOS_effect_gain_1"],.5)
        self.assertEqual(m["bias_EOS_effect_extra_gain1_minus_gain0p25"],.375)
    def test_02_zero_readout_does_not_imply_eos(self):
        m=head_report(sample(0))
        self.assertTrue(m["output_bias_global_native_zero_output_bias"])
        self.assertEqual(m["bias_EOS_effect_extra_gain1_minus_gain0p25"],0.)
    def test_03_fp16_only_exact_rank4_and_49152(self):
        bad=sample()
        bad[PREFIX+"language_head.up.weight"]=bad[
            PREFIX+"language_head.up.weight"].astype(np.float32)
        with self.assertRaises(ValueError):real_head_only(bad)
        bad=sample()
        del bad[PREFIX+"language_head.output_bias"]
        with self.assertRaises(ValueError):real_head_only(bad)
    def test_04_wrong_weight_namespace_rejected(self):
        d={k.replace(PREFIX,"model.live.") : v for k,v in sample().items()}
        with self.assertRaises(ValueError):real_head_only(d)
if __name__=="__main__":
    unittest.main()
