"""v182 precommitted diagnostic invariants, no model files or private TEST."""
import unittest
from types import SimpleNamespace

import numpy as np
from direct_codebook_v59 import unpack5
from v181.codec import quantize
from v182.diagnose import decode_handle,distance,ordered_checkpoints,summarize_cases
from v182 import protocol as p

class FirstDivergenceRegressions(unittest.TestCase):
    def test_pinned_source_and_no_hidden_test_cases(self):
        self.assertEqual(p.FROZEN_CASE_IDS,(0,16,64,80))
        self.assertEqual(p.MODEL_SOURCE_BYTES,84679302)
        self.assertAlmostEqual(p.V1_TOTAL_BPP,5.0361251586,places=7)
        self.assertEqual(len(p.ARMS),6)
        self.assertEqual(len(ordered_checkpoints()),33)
        self.assertEqual(ordered_checkpoints()[0],"embedding")
        self.assertEqual(ordered_checkpoints()[-1],"next_token_logits")

    def test_exact_source_decoding_not_fp32_oracle_copy(self):
        rng=np.random.default_rng(182)
        original=rng.standard_normal(size=(17,64)).astype(np.float32)
        scale=np.sqrt(np.mean(original.astype(np.float64)**2,axis=1)).astype("<f2")
        parts,stats=quantize(original,scale,1)
        h=SimpleNamespace(group=1,shape=original.shape,
            packed=np.frombuffer(parts["packed"],dtype=np.uint8),
            row_scales=np.frombuffer(parts["row_scales"],dtype="<f2").astype(np.float32),
            centroids=np.frombuffer(parts["centroids"],dtype="<f2").astype(np.float32).reshape(32,1))
        direct=decode_handle(h)
        ids=unpack5(parts["packed"],17*64)
        expected=h.centroids[ids].reshape(original.shape)*h.row_scales[:,None]
        np.testing.assert_array_equal(direct,expected)
        self.assertGreater(np.linalg.norm(direct-original),0)
        self.assertTrue(np.isfinite(stats["relative_weight_RMSE"]))

    def test_relative_L2_and_similarity(self):
        ref=np.array([1.,2.,3.],dtype=np.float32)
        eq=distance(ref,ref)
        self.assertEqual(eq["relative_L2_to_reference"],0.)
        self.assertAlmostEqual(eq["cosine"],1,places=10)
        small=distance(ref,ref+np.array([0.,0.,.5]))
        self.assertGreater(small["relative_L2_to_reference"],.05)
        with self.assertRaises(ValueError):
            distance(ref,ref[:2])

    def test_summary_flags_real_native_dense_mismatch(self):
        cases=[]
        for idx in p.FROZEN_CASE_IDS:
            case={"first_over_5pct_by_checkpoint":{k:"embedding"
                  for k in p.ARMS if k!="FP32"},
                  "distances_to_FP32":{
                      k:{"next_token_logits":{"relative_L2_to_reference":0.1}}
                      for k in p.ARMS if k!="FP32"},
                  "first_token_matches_fp32":{k:False for k in p.ARMS if k!="FP32"},
                  "first_token_ids":{"V1_DENSE":123,"V1_NATIVE":124},
                  "native_vs_same_decoded_V1":{k:{"relative_L2_to_reference":0.01}
                                                 for k in ordered_checkpoints()}}
            cases.append(case)
        outcome=summarize_cases(cases)
        self.assertTrue(outcome["native_vs_same_V1_decoded"]["max_error_exceeds_precommitted_warning"])
        self.assertFalse(outcome["native_vs_same_V1_decoded"]["same_first_token_all_cases"])
        self.assertEqual(outcome["V1_DENSE"]["first_crossing_each_case"],["embedding"]*4)

if __name__=="__main__":
    unittest.main()
