"""S119q2i8s physical FP16 head-only artifacts and causal pairing contracts."""
import unittest
import numpy as np
from research.calibri_nra3.s119q2i8s_head_fp16_eos_paired_forensics import (
    STAGE,VOCAB,EOS,PREFIX,readout,head_numeric_state,paired_by_source,
)
def fake():
    d={
        PREFIX+"language_head.output_bias":np.zeros((VOCAB,),dtype=np.float16),
        PREFIX+"language_head.up.weight":np.zeros((VOCAB,4),dtype=np.float16),
        PREFIX+"language_head.down.weight":np.zeros((4,320),dtype=np.float16),
    }
    d[PREFIX+"language_head.output_bias"][EOS]=np.float16(-.05)
    return d
class TestRealNativeHeadPair(unittest.TestCase):
    def test_01_native_vocab_EOS_rank4(self):
        self.assertIn("EOS_MARGIN",STAGE)
        d=fake()
        r=head_numeric_state(d)
        self.assertEqual(r["native_vocab"],49152)
        self.assertEqual(r["current_real_EOS_ID"],2)
        self.assertAlmostEqual(r["real_saved_output_bias_EOS"],-.05,places=3)
    def test_02_fails_if_head_missing_or_non_fp16(self):
        d=fake();d.pop(PREFIX+"language_head.output_bias")
        with self.assertRaises(ValueError):readout(d)
        d=fake();d[PREFIX+"language_head.up.weight"]=np.zeros((VOCAB,4),dtype=np.float32)
        with self.assertRaises(ValueError):readout(d)
    def test_03_pairs_full_original_native_self_ids(self):
        def x(i,ids):
            return {
                "source_id":str(i),
                "source_internal_DEV_NOT_independent_blind":True,
                "generated_own_native_BPE_ids":ids,
                "true_native_full_EOS_correct":False,
                "first_gold_BPE_correct":False
            }
        a={"NEW_DEV":{"OWN_samples":[x(0,[2]),x(1,[44,2])]}}
        b={"NEW_DEV":{"OWN_samples":[x(0,[2]),x(1,[55,2])]}}
        v=paired_by_source(a,b,"NEW_DEV",2)
        self.assertEqual(v["identical_full_native_BPE_sequences"],1)
        self.assertEqual(v["different_full_native_BPE_sequences"],1)
        self.assertEqual(v["first_differing_native_position_1based"],{"1":1})
    def test_04_rejects_duplicate_sources(self):
        def x():return {
            "source_id":"a","source_internal_DEV_NOT_independent_blind":True,
            "generated_own_native_BPE_ids":[2],
            "true_native_full_EOS_correct":False,"first_gold_BPE_correct":False}
        a={"NEW_DEV":{"OWN_samples":[x(),x()]}}
        with self.assertRaises(ValueError):paired_by_source(a,a,"NEW_DEV",2)
if __name__=="__main__":
    unittest.main()
