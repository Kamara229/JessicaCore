import unittest
import numpy as np
from v139.protocol import (budget,RESIDUAL_BITPLANE_BYTES,
    RESIDUAL_CODEBOOK_BYTES,RESIDUAL_BITS)
from v139.refinement import train,score,assign


class OneBitTiedResidualSource(unittest.TestCase):
    def test_physically_paid_all_model_rate_under_0p7(self):
        physical=budget(11_229_303)
        self.assertEqual(RESIDUAL_BITPLANE_BYTES,442_368)
        self.assertEqual(RESIDUAL_BITS,3_538_944)
        self.assertEqual(RESIDUAL_CODEBOOK_BYTES,1024)
        self.assertLessEqual(physical["all_in_bpp_conservative"],0.7)
        self.assertGreater(physical["remaining_headroom_bytes"],0)
        self.assertTrue(physical["does_not_include_v119_paid_body_or_head_adapters"])
        with self.assertRaises(ValueError):
            budget(11_770_063)

    def test_fixed_lattice_two_subcenters_uses_one_bit(self):
        rng=np.random.default_rng(139)
        main=np.zeros((32,8),dtype=np.float32)
        # 32 fixed base symbols, two opposite residual modes per symbol.
        codes=np.repeat(np.arange(32),24)
        modes=np.tile([0,1],len(codes)//2)
        vec=np.zeros((len(codes),8),dtype=np.float32)
        vec[:,0]=np.where(modes==0,-1.0,+1.0)
        vec+=rng.normal(0,.025,size=vec.shape).astype(np.float32)
        extra=train(vec,codes,main,steps=5)
        self.assertEqual(extra.shape,(32,2,8))
        self.assertTrue(np.all(np.isfinite(extra)))
        self.assertTrue(np.array_equal(extra,extra.astype("<f2").astype(np.float32)))
        scales=np.ones(len(vec),dtype=np.float32)
        plain=score(vec,vec,scales,codes,main)
        improved=score(vec,vec,scales,codes,main,extra)
        self.assertLess(improved["relative_weight_RMSE"],plain["relative_weight_RMSE"]*.15)
        self.assertEqual(sum(improved["refinement_onebit_codes_USED"]),len(vec))
        self.assertTrue(np.array_equal(extra,train(vec,codes,main,steps=5)))

    def test_validation_fails_for_impossible_code(self):
        r=np.zeros((3,8),dtype=np.float32)
        centers=np.zeros((32,8),dtype=np.float32)
        with self.assertRaises(ValueError):
            score(r,r,np.ones(3),[0,1,32],centers)
        with self.assertRaises(ValueError):
            train(r,np.zeros(4,dtype=int),centers)
        with self.assertRaises(ValueError):
            assign(r,np.zeros((3,8),dtype=np.float32))


if __name__=="__main__":
    unittest.main()
