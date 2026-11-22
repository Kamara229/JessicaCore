"""S119q2i8m FP16 physical delta measurement math without models."""
import unittest
import numpy as np
from research.calibri_nra3.s119q2i8m_fp16_early23_delta_forensics import (
    quantized_diff,group,
)

class TestActualHalfDelta(unittest.TestCase):
    def test_01_equal_saved_values_not_a_different_trained_model(self):
        a={"k":np.array([1.,2.,3.],dtype=np.float16)}
        z=quantized_diff(a,a)
        self.assertEqual(z["changed_fp16_params"],0)
        self.assertEqual(z["delta_L2"],0)
    def test_02_detect_only_physical_fp16_difference(self):
        a={"x.language_head.up.weight":np.array([1.,2.,3.],dtype=np.float16)}
        b={"x.language_head.up.weight":np.array([1.,2.5,3.],dtype=np.float16)}
        d=quantized_diff(a,b)
        self.assertEqual(d["changed_fp16_params"],1)
        self.assertEqual(d["delta_L2"],.5)
        self.assertEqual(d["regions"]["native_49152_readout"]["parameters_different"],1)
    def test_03_invalid_no_checkpoint_or_wrong_shape_rejected(self):
        with self.assertRaises(ValueError):
            quantized_diff({"x":np.ones(2,dtype=np.float16)},
                           {"x":np.ones(3,dtype=np.float16)})
        with self.assertRaises(ValueError):
            quantized_diff({"x":np.ones(2,dtype=np.float32)},
                           {"x":np.ones(2,dtype=np.float32)})
    def test_04_detect_core_vs_protected_regions(self):
        self.assertEqual(group("model.nra3_calibri_joint.s119q2b.child.core_growth.up"),
                         "Core_growth")
        self.assertEqual(group("model.nra3_calibri_joint.s119q2b.x.nra3_neural_weights.x"),
                         "S115_protected_neural_state")
if __name__=="__main__":
    unittest.main()
