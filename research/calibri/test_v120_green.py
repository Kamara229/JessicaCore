"""Byte-agnostic critic must be physically small and avoid gold-inference cheating."""
import unittest
import torch
from v120.protocol import MAX_CHECKPOINT_BYTES,VALID_A,VALID_B
from v120.critic import GreenTunnel
from v120.bytes_adapter import encode,pairs
from v120.examples import make_train_pairs,validation_pairs
class TestGreen(unittest.TestCase):
    def test_independent_float32_neural_size(self):
        model=GreenTunnel()
        self.assertLess(model.disk_parameters()*4,MAX_CHECKPOINT_BYTES)
        self.assertGreater(model.disk_parameters(),500_000)
        self.assertTrue(all(p.ndim<=2 for p in model.parameters()))
    def test_byte_interface_no_hidden_dimension_required(self):
        a=encode("Почему идёт дождь?","Потому что образуются капли воды.")
        self.assertEqual(len(a),160)
        self.assertTrue(all(0<=z<258 for z in a))
        model=GreenTunnel().eval()
        with torch.inference_mode():
            z=model(pairs([("What is 3+4?","7"),("Where?","Yes Yes Yes")]))
        self.assertEqual(tuple(z.shape),(2,))
        self.assertTrue(torch.isfinite(z).all())
    def test_gold_labels_only_fit_and_holdouts_disjoint(self):
        train=make_train_pairs()
        val=validation_pairs(VALID_A+VALID_B)
        self.assertEqual(len(val),72)
        self.assertFalse({x for x,_,_ in train}&{x for x,_,_ in val})
        self.assertEqual(sum(label for q,a,label in train),34)
if __name__=="__main__":
    unittest.main()
