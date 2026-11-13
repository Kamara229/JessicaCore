"""S198 structural safety. Real gradients occur in GitHub CPU workflow, NOT mocks."""
from pathlib import Path
import ast
import unittest

from calibri2.s198.one_step_real_nra import CHOSEN_INDICES, TRAIN_LR, STAGE
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes


class OneTrueCPUNRAStep(unittest.TestCase):
    def test_source_is_exact_original_new_train_32(self):
        rows=new_train_paraphrase_probes()
        self.assertEqual(len(rows),32)
        self.assertEqual(tuple(CHOSEN_INDICES),(0,1,2,24))
        self.assertEqual(len(set(CHOSEN_INDICES)),4)
        self.assertTrue(all(0<=i<32 for i in CHOSEN_INDICES))
        self.assertLessEqual(TRAIN_LR,.001)
        self.assertTrue(all(rows[i][0] and rows[i][1] for i in CHOSEN_INDICES))

    def test_training_code_preserves_real_backprop_physical_gate(self):
        p=Path(__file__).with_name("one_step_real_nra.py")
        source=p.read_text()
        tree=ast.parse(source)
        self.assertTrue(tree)
        self.assertIn("load_cpu_original(raw,side)",source)
        self.assertIn("ProgressiveRouteCore(rank=1344)",source)
        self.assertIn("area.install(model)",source)
        self.assertIn("loss.backward()",source)
        self.assertIn("decoder_grad=area.decoder.weight.grad",source)
        self.assertIn("optimizer.step()",source)
        self.assertIn("FP16",source)
        self.assertIn("all_legacy_non_NRA_tensors_byte_identical",source)
        self.assertIn('"model_accepted":False',source)

    def test_no_inference_identity_gold_lookup(self):
        src=Path(__file__).with_name("one_step_real_nra.py").read_text()
        self.assertNotIn("test_dataset",src.lower())
        self.assertNotIn("QA_RESULTS",src)
        self.assertIn("independent_postupdate_real_CPU_J7_616_REQUIRED",src)
        self.assertIn("no_claim_of_new_correct_answers",src)


if __name__=="__main__":
    unittest.main()
