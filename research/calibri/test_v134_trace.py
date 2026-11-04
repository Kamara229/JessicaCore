import unittest
from types import SimpleNamespace
import torch
from v134.protocol import ARMS,apply,corpus
from v134.trace import compare


class MockBridge:
    def __init__(self):
        self.blocks={f"model.layers.{i}.{name}":SimpleNamespace(gain=1.)
                     for i in range(30) for name in ("self_attn.o_proj","mlp.down_proj")}
    def control(self,controller,question):
        for block in self.blocks.values():block.gain=.7 if controller else 1.


class FrozenTraceTests(unittest.TestCase):
    def test_fixed_paired_inputs_and_band0(self):
        self.assertEqual(len(corpus()),16)
        self.assertEqual(len(ARMS),6)
        bridge=MockBridge()
        for arm in ARMS[2:]:
            apply(bridge,object(),"question",arm)
            for name,b in bridge.blocks.items():
                if "no_0to4" in arm and int(name.split(".")[2])<5:
                    self.assertEqual(b.gain,0.0)

    def test_identity_has_zero_KL_and_hidden_error(self):
        states={"embeddings":torch.ones(64)}
        states.update({f"layer_{i}":torch.ones(64) for i in range(30)})
        logits=torch.tensor([1.,0.,-2.,3.])
        report=compare(states,states,logits,logits,2)
        self.assertLess(abs(report["teacher_to_student_KL"]),1e-7)
        self.assertTrue(report["argmax_equal"])
        self.assertFalse(report["student_first_EOS"])
        self.assertEqual(sum(row["relative_L2"] for row in report["layer_trajectory"]),0)

    def test_first_layer_added_error_detected(self):
        states={"embeddings":torch.ones(64)}
        states.update({f"layer_{i}":torch.ones(64) for i in range(30)})
        damaged={key:value.clone() for key,value in states.items()}
        damaged["layer_0"][0]=10.
        logits=torch.tensor([1.,0.,-2.,3.])
        result=compare(states,damaged,logits,logits,2)
        self.assertEqual(result["max_increase_layer"],0)
        self.assertGreater(result["layer_trajectory"][1]["relative_L2"],0)

    def test_missing_layer_not_silently_ignored(self):
        states={"embeddings":torch.ones(64)}
        states.update({f"layer_{i}":torch.ones(64) for i in range(30)})
        logits=torch.tensor([1.,0.,-2.,3.])
        states.pop("layer_29")
        with self.assertRaises(ValueError):compare(states,states,logits,logits,2)


if __name__=="__main__":
    unittest.main()
