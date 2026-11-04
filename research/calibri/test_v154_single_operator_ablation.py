"""No dataset fetch: all original seven FP32-source paid operators ablated."""
import unittest
import numpy as np
from v153.codec import Physical,Patch
from v154.protocol import (ARMS,LAYER,MATRIX_KEYS,CB153_SHA,
    EXPECTED_ARMS,check_protocol)
from v154.arms import make_arms,subset
from v154.scoring import aggregate


def fake_physical():
    shapes={name:(64,64) for name in MATRIX_KEYS}
    patches={key:Patch(key,shapes[key],
          np.zeros((32,2,8),dtype=np.float32),
          np.zeros((64*64//8+7)//8,dtype=np.uint8))
             for key in MATRIX_KEYS}
    return Physical(b"no real source in synthetic tests",LAYER,patches,
                    1,CB153_SHA)


class FrozenSourceOperatorAblation(unittest.TestCase):
    def test_all_seven_source_operator_controls_and_immutable_archive(self):
        model=fake_physical()
        cases=check_protocol(model)
        self.assertEqual(len(cases),8)
        self.assertEqual(len(MATRIX_KEYS),7)
        self.assertEqual(EXPECTED_ARMS,16)
        self.assertEqual(len(ARMS),16)
        arms=make_arms(model)
        self.assertEqual(tuple(arms),ARMS)
        self.assertEqual(len(arms["CB59_original"]),0)
        self.assertEqual(len(arms["CB153_all7"]),7)
        for key in MATRIX_KEYS:
            short=key.split(".")[-2]
            self.assertEqual(arms[f"only_{short}"],(key,))
            self.assertEqual(len(arms[f"except_{short}"]),6)
            self.assertNotIn(key,arms[f"except_{short}"])
        paid=subset(model,arms["only_q_proj"])
        self.assertIsNot(paid,model)
        self.assertEqual(list(paid.patches),[MATRIX_KEYS[0]])
        self.assertEqual(len(model.patches),7)
        with self.assertRaises(ValueError):
            subset(model,("model.layers.0.mlp.up_proj.weight",))

    def test_corrupt_archive_sha_or_operator_count_refused(self):
        a=fake_physical()
        other=Physical(a.original,6,a.patches,a.size,"0"*64)
        with self.assertRaises(ValueError):
            check_protocol(other)
        missing=dict(a.patches)
        missing.pop(next(iter(missing)))
        other=Physical(a.original,6,missing,a.size,CB153_SHA)
        with self.assertRaises(ValueError):
            check_protocol(other)

    def test_causal_operator_sign_and_unhidden_negative_interaction(self):
        results={arm:[{
            "teacher_to_student_KL":9.0,
            "FP32_first_token_argmax_agreement":False,
            "first_EOS":False,
        } for _ in range(8)] for arm in ARMS}
        # One operator alone appears beneficial; another harmful.
        results["only_q_proj"]=[{
            "teacher_to_student_KL":7.5,
            "FP32_first_token_argmax_agreement":False,
            "first_EOS":False} for _ in range(8)]
        results["only_k_proj"]=[{
            "teacher_to_student_KL":10.5,
            "FP32_first_token_argmax_agreement":False,
            "first_EOS":False} for _ in range(8)]
        results["CB153_all7"]=[{
            "teacher_to_student_KL":9.25,
            "FP32_first_token_argmax_agreement":False,
            "first_EOS":False} for _ in range(8)]
        scores=aggregate(results,{arm:1.0 for arm in ARMS})
        self.assertEqual(scores["all7_KL_delta_vs_CB59"],.25)
        self.assertEqual(scores["per_original_operator_diagnostic"][0]
                         ["original_operator"],MATRIX_KEYS[0])
        results["only_q_proj"].pop()
        with self.assertRaises(ValueError):
            aggregate(results,{arm:1.0 for arm in ARMS})
        results.pop("except_q_proj")
        with self.assertRaises(ValueError):
            aggregate(results,{arm:1.0 for arm in ARMS})


if __name__=="__main__":
    unittest.main()
