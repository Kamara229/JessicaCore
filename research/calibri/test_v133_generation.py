import unittest
from types import SimpleNamespace
from v133.arms import ARMS,set_arm,assert_band0_gains
from v133.scoring import evaluate_case,report
from v132.protocol import all_examples,band_of


class MockBridge:
    def __init__(self):
        self.blocks={f"model.layers.{layer}.{role}":SimpleNamespace(gain=-1.0)
                     for layer in range(30) for role in ("self_attn.o_proj","mlp.down_proj")}
    def control(self,policy,question):
        value = 0.8 if policy is not None else 1.0
        for layer in self.blocks.values():layer.gain=value


class TestBand0Generation(unittest.TestCase):
    def test_exact_sixteen_counterfactual_prompts(self):
        self.assertEqual(len(list(all_examples())),16)
        self.assertEqual(len(ARMS),7)

    def test_gains_no_weight_change(self):
        bridge=MockBridge()
        for arm in ARMS[2:]:
            gains=set_arm(bridge,object(),"question","opposite",arm)
            assert_band0_gains(gains,arm)
            for role,gain in gains.items():
                self.assertEqual(gain,0.0 if "drop_band0" in arm and band_of(role)==0
                                 else (0.8 if "green_" in arm else 1.0))

    def test_no_semantic_credit_from_eos_or_token_appearance(self):
        tokenizer=SimpleNamespace(eos_token_id=2)
        generated={"ids":[2],"text":"Blue red","eos_generated":True}
        r=evaluate_case(tokenizer,"one","left","Q","blue","red",generated)
        self.assertTrue(r["first_generated_token_is_EOS"])
        self.assertFalse(r["observation"]["target_visible_alone_NOT_proven_semantic"])
        self.assertFalse(r["observation"]["strict_exact_text"])


if __name__=="__main__":
    unittest.main()
