"""Tests on true five-bit source-index entropy and adjacent-code structure."""
import unittest
import numpy as np
from v179.index_entropy import entropy, conditioned_entropy, family_name


class PhysicalIndexEntropy(unittest.TestCase):
    def test_entropy_of_uniform_and_constant(self):
        self.assertAlmostEqual(entropy(np.ones(32)),5.0,places=8)
        self.assertAlmostEqual(entropy([0,0,100,0]),0.0,places=10)
        self.assertAlmostEqual(entropy([50,50]),1.0,places=10)

    def test_row_prediction_and_invalid(self):
        constant=np.full((11,72),7,dtype=np.uint8)
        self.assertAlmostEqual(conditioned_entropy(constant),0.,places=10)
        repeated=np.tile(np.arange(32,dtype=np.uint8),(128,3))
        self.assertAlmostEqual(conditioned_entropy(repeated),0.,places=10)
        random=np.random.default_rng(179).integers(0,32,size=(2500,72),dtype=np.uint8)
        self.assertGreater(conditioned_entropy(random),4.9)
        with self.assertRaises(ValueError):
            conditioned_entropy(np.array([[32,0]],dtype=np.uint8))

    def test_all_family_names(self):
        self.assertEqual(family_name("model.embed_tokens.weight"),"embedding_head")
        self.assertEqual(family_name("model.layers.0.self_attn.q_proj.weight"),"attention")
        self.assertEqual(family_name("model.layers.0.mlp.down_proj.weight"),"mlp_down")
        self.assertEqual(family_name("model.layers.0.mlp.gate_proj.weight"),"mlp_gate_up")


if __name__=="__main__":
    unittest.main()
