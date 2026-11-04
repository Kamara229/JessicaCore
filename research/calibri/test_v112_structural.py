"""v112 physical-byte invariants and no-full-matrix direct decode tests."""
import hashlib
import unittest
import numpy as np
from v112.archive import pack5,unpack5_region,Archive,create_archive
from v112.toy import make_toy
from v112.pipeline import weight_error

class TestStructural(unittest.TestCase):
    def test_bitstream_and_offsets(self):
        rng=np.random.default_rng(112)
        for n in (1,2,3,7,8,9,31,65,129):
            x=rng.integers(0,32,n,dtype=np.uint8)
            raw=pack5(x)
            self.assertEqual(len(raw),(5*n+7)//8)
            np.testing.assert_array_equal(unpack5_region(raw,0,n),x)
            if n>4:
                np.testing.assert_array_equal(unpack5_region(raw,2,n-3),x[2:n-1])
        with self.assertRaises(ValueError):
            pack5(np.array([32],dtype=np.uint8))
    def test_same_physical_cost_and_streaming(self):
        source=make_toy(layers=4,rows=32,cols=32)
        sizes=set()
        for mode in ("global","role","layer_control"):
            b=create_archive(source,mode)
            a=Archive(b)
            sizes.add(len(b))
            self.assertEqual(set(source),set(a.records))
            self.assertTrue(np.isfinite(weight_error(source,a)))
            self.assertEqual(sum(w.size for w in source.values()),a.summary()["selected_elements"])
            for name in list(source)[:3]:
                x=np.linspace(-1,1,source[name].shape[1],dtype=np.float32)[None,:]
                np.testing.assert_allclose(a.direct(name,x,chunk=5),
                                           x@a.decode(name).T,rtol=1e-4,atol=2e-5)
            bad=bytearray(b);bad[-45]^=1
            with self.assertRaises(ValueError):
                Archive(bytes(bad))
        self.assertEqual(len(sizes),1)
    def test_structural_roles_present_and_negative_control(self):
        structured=make_toy(structured=True,layers=4,rows=32,cols=32)
        unstructured=make_toy(structured=False,layers=4,rows=32,cols=32)
        self.assertEqual(set(structured),set(unstructured))
        self.assertFalse(all(np.array_equal(structured[k],unstructured[k]) for k in structured))

if __name__=="__main__":
    unittest.main()
