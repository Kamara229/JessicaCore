"""Fail-closed CPU contract for one S144 inherited physical J7 (no GPU/QA)."""
import ast
import json
from pathlib import Path
import unittest


HERE=Path(__file__).parent


class S144ContinuationContract(unittest.TestCase):
    def test_one_two_step_J7_from_exact_S142(self):
        script=(HERE/"train_one_continued_j7.py").read_text()
        tree=ast.parse(script)
        self.assertEqual(ast.get_docstring(tree),"CALIBRI S144: ONE real protected MPS J7 continuing from committed S142 FP16.\nFreeze original S101/Q2 + first 24 S132 channels; train only new 24 channels.\nAtomic CPU 456+128 full+first rollback guard; no unseen QA and no LIVE changes.\n")
        self.assertIn("STEPS=2",script)
        self.assertIn("TRANSACTION_STEPS=2",script)
        self.assertIn("parent_sha=sha(a.S142_parent_sidecar.read_bytes())",script)
        self.assertIn("inherited=load_file(str(a.S142_parent_sidecar))",script)
        self.assertIn("s132_inherited=load_file(str(a.S132_parent_sidecar))",script)
        self.assertIn("s132_inherited[WEIGHTS_KEY][:,:,:24]",script)
        self.assertIn("prior={\"old_456_full_correct\":362,\"new_128_full_correct\":17}",script)
        self.assertIn("parent_guard.get(\"committed_gain_no_forgetting\") is not True",script)
        self.assertIn("parent_guard.get(\"physical_FP16_reload_pass\") is not True",script)
        self.assertIn("parent_guard.get(\"old_456_full\")!=362",script)
        self.assertIn("parent_guard.get(\"S122_128_full\")!=17",script)
        self.assertIn("parent_guard.get(\"CPU_speed_ratio\",0)<.9",script)
        self.assertIn("parent_guard.get(\"physical_bpp\",1)>.7",script)
        self.assertIn("correct_adam_stage2_displacement(",script)
        self.assertIn("atomic_train_block(",script)
        self.assertIn("parent_correct_new_full",script)
        self.assertIn("parent_correct_new_first",script)
        self.assertIn("verify_frozen_inherited(",script)
        self.assertIn("classify_j7_development_gain(",script)
        self.assertIn("S144_ACTUAL_43M_CONTINUED_J7_FP16.safetensors",script)
        self.assertIn("S144_ONE_J7_ACTUAL_PHYSICAL_GUARD.json",script)
        self.assertIn('"model_accepted":False',script)
        self.assertIn('"no_heldout_QA_or_inspected_S118_access":True',script)
        self.assertNotIn("load_heldout",script)

    def test_s142_source_manifest_remains_pinned(self):
        m=json.loads((HERE/"source_manifest.json").read_text())
        self.assertEqual(m["source_run_id"],43078895081)
        self.assertEqual(m["source_artifact_id"],13885634834)
        self.assertEqual(m["checkpoint_sha256"],"5a706d9012a61f24678929a2a6ee20f77c922e7e0f4bb516c061f32742cc56c7")
        self.assertTrue(m["never_promote_model"])
        self.assertTrue(m["never_run_jessica_chat_tests"])


if __name__=="__main__":
    unittest.main()
