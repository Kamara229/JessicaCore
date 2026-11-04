import unittest
import numpy as np
from v137.protocol import select,groups,physical_budget,FIT_GROUPS,VALID_GROUPS
from v137.fit import lloyd,product_centers,score
from direct_codebook_v59 import _nearest_integer_128


class FixedRateDictionaryTests(unittest.TestCase):
    def test_disjoint_source_train_validation(self):
        fit,valid=select(500_000)
        self.assertEqual((len(fit),len(valid)),(FIT_GROUPS,VALID_GROUPS))
        self.assertEqual(len(np.intersect1d(fit,valid)),0)
        a,b=select(500_000)
        self.assertTrue(np.array_equal(fit,a))
        self.assertTrue(np.array_equal(valid,b))

    def test_same_literal_code_and_centroid_bits(self):
        r=physical_budget(11_229_303,(49152,576))
        self.assertEqual(r["same_codebook_entries"],32)
        self.assertEqual(r["same_vector_dimension"],8)
        self.assertEqual(r["same_centroid_size_fp16_bytes"],512)
        self.assertEqual(r["incremental_centroid_training_bytes_IF_repacked"],0)
        self.assertEqual(r["embedding_packed_codes_fixed_bytes"],2_211_840)

    def test_fit_and_product_are_finite_32x8(self):
        rng=np.random.default_rng(137)
        source=rng.normal(size=(4096,8)).astype(np.float32)
        init=rng.normal(size=(32,8)).astype(np.float16).astype(np.float32)
        conventional=lloyd(source,init,passes=3)
        weighted=lloyd(source,init,energy=1+rng.uniform(size=len(source)),passes=3)
        product=product_centers(source,init,passes=3)
        for item in (conventional,weighted,product):
            self.assertEqual(item.shape,(32,8))
            self.assertTrue(np.isfinite(item).all())
            self.assertTrue(np.array_equal(item,item.astype(np.float16).astype(np.float32)))
        self.assertTrue(np.array_equal(conventional,lloyd(source,init,passes=3)))
        result=score(source,source,np.ones(len(source)),conventional)
        self.assertTrue(np.isfinite(result["relative_weight_RMSE"]))

    def test_sample_group_layout_and_source_code_index_bounds(self):
        weight=np.arange(64,dtype=np.float32).reshape(4,16)
        actual,normalized,s=groups(weight,np.ones(4,dtype=np.float32),[0,2,7])
        self.assertEqual(actual.shape,(3,8))
        self.assertTrue(np.array_equal(actual[1],weight[1,:8]))
        self.assertTrue(np.array_equal(actual[2],weight[3,8:]))
        centers=np.zeros((32,8),dtype=np.float32)
        with self.assertRaises(ValueError):
            score(actual,normalized,s,centers,source_codes=[0,31,33])


if __name__=="__main__":
    unittest.main()
