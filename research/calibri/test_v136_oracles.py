import unittest
from torch import nn
from v136.oracles import ARMS, teacher_components, parameters, expected_physical_oracle_bytes
from v136.scoring import summary
from v134.protocol import corpus


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=nn.Linear(8,8)
        self.mlp=nn.Linear(8,8)


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.embed_tokens=nn.Embedding(32,8)
        self.model.layers=nn.ModuleList([ToyLayer() for i in range(30)])
        self.lm_head=nn.Linear(8,32,bias=False)


class TestOracleAblation(unittest.TestCase):
    def test_arms_and_paired_prompts(self):
        self.assertEqual(len(ARMS),8)
        self.assertEqual(len(corpus()),16)
        self.assertEqual(parameters("oracle_input_attn0")["attn0"],True)
        with self.assertRaises(ValueError):
            parameters("unplanned_control")

    def test_first_five_and_input_are_swapped_and_restored(self):
        student,teacher=Toy(),Toy()
        own_input=student.model.embed_tokens
        own_layers=list(student.model.layers)
        with teacher_components(student,teacher,"oracle_input_first5"):
            self.assertIs(student.model.embed_tokens,teacher.model.embed_tokens)
            for index in range(5):
                self.assertIs(student.model.layers[index],teacher.model.layers[index])
            self.assertIs(student.model.layers[5],own_layers[5])
        self.assertIs(student.model.embed_tokens,own_input)
        for i in range(30):self.assertIs(student.model.layers[i],own_layers[i])

    def test_block0_attention_only_is_restored_even_on_error(self):
        student,teacher=Toy(),Toy()
        original=student.model.layers[0].self_attn
        old_mlp=student.model.layers[0].mlp
        with self.assertRaisesRegex(RuntimeError,"intentional"):
            with teacher_components(student,teacher,"oracle_input_attn0"):
                self.assertIs(student.model.layers[0].self_attn,teacher.model.layers[0].self_attn)
                self.assertIs(student.model.layers[0].mlp,old_mlp)
                raise RuntimeError("intentional")
        self.assertIs(student.model.layers[0].self_attn,original)

    def test_uncompressed_bytes_never_silent(self):
        teacher=Toy()
        self.assertEqual(expected_physical_oracle_bytes(teacher,"CB59"),0)
        self.assertGreater(expected_physical_oracle_bytes(teacher,"oracle_input"),0)
        self.assertGreater(expected_physical_oracle_bytes(teacher,"oracle_input_first5"),
                           expected_physical_oracle_bytes(teacher,"oracle_input"))

    def test_report_rejects_missing_arm(self):
        with self.assertRaises(ValueError):
            summary({"CB59": []})


if __name__=="__main__":
    unittest.main()
