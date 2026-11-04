"""Physical-byte safety for TRAIN-only same-size correction."""
import unittest
import numpy as np
from v112.toy import make_toy
from v112.archive import Archive,create_archive
from v113.train import gain_for_row,recalibrate

class Tests(unittest.TestCase):
    def test_projection(self):
        a=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
        np.testing.assert_allclose(gain_for_row(3*a,a),[2.,2.])
    def test_only_fp16_scales_can_change(self):
        source=make_toy(layers=4,rows=16,cols=32)
        rng=np.random.default_rng(113)
        teacher={name:rng.normal(size=(16,w.shape[1])).astype(np.float32)
                 for name,w in source.items()}
        student={name:x+rng.normal(0,.02,x.shape).astype(np.float32)
                 for name,x in teacher.items()}
        old=create_archive(source,"role")
        new,_=recalibrate(old,source,teacher,student)
        self.assertEqual(len(old),len(new))
        a,b=Archive(old),Archive(new)
        for key in source:
            np.testing.assert_array_equal(a.books,b.books)
            self.assertEqual(bytes(a._parts(key)[2]),bytes(b._parts(key)[2]))
        self.assertNotEqual(old,new)
    def test_missing_fit_rejected(self):
        source=make_toy(layers=4,rows=16,cols=32)
        with self.assertRaises(ValueError):
            recalibrate(create_archive(source,"global"),source,{},{})

if __name__=="__main__":unittest.main()
