"""S105 only synthetic statistical/protocol tests, never open Taiga-C gold.

First actual held C score runs ONLY once after original protected616
real source and both physical in-Calibri checkpoint SHAs verified.
"""
import ast
import unittest
from pathlib import Path

from research.calibri_nra3 import s105_compare_frozen_S99_S102_ONCE_TaigaC_original12 as s105
from research.calibri_nra3 import s105_prospective_TaigaC_offline_gold_AFTER_original12_SHA as gold


class TestS105OneShotFrozenNoTuning(unittest.TestCase):
    def test_model_hashes_frozen_before_all_held_source_and_gold_access(self):
        self.assertEqual(s105.S99_SHA,
          "c77a6964ae4e02cb3aac98d69978a25cec2b491a9624cc9668b3b307699bff8a")
        self.assertEqual(s105.S102_SHA,
          "f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df")
        self.assertEqual(gold.SOURCE_SHA,
          "7c960fd28b14449cfb9fa3611576aa21a88dbd6afccbbde4e2a02d015707bcdb")
        self.assertEqual(gold.GOLD_SHA,
          "e039c3c6a8ac2221cea774895c09da5838b3d50e7d71079c9ff3ca0d285827eb")
        self.assertEqual(s105.EXPECTED_PAID,28887)

    def test_paired_bootstrap_fixed_seed_does_not_use_checkpoint_selection(self):
        self.assertEqual(s105.BOOTSTRAP_SEED,1051119)
        self.assertEqual(s105.BOOTSTRAP_DRAWS,2000)
        self.assertEqual(s105.wilson95(0,0),None)
        ci=s105.wilson95(6,10)
        self.assertTrue(0.25<ci[0]<0.4)
        self.assertTrue(0.8<ci[1]<0.9)
        a=[];b=[]
        for i in range(100):
            x=s105.empty_summary();y=s105.empty_summary()
            x["role"]={"correct":int(i%3==0),"slots":1}
            y["role"]={"correct":int(i%3!=0),"slots":1}
            a.append(x);b.append(y)
        z=s105.paired_row_bootstrap_s102_minus_s99(a,b)
        q=s105.paired_row_bootstrap_s102_minus_s99(a,b)
        self.assertEqual(z,q)
        self.assertTrue(0<z["S102_minus_S99_95_percentile_CI"][0]<
                        z["S102_minus_S99_95_percentile_CI"][1])

    def test_held_GOLD_import_occurs_ONLY_after_real_original_source_SHA_AND_both_weights(self):
        p=Path("research/calibri_nra3/s105_compare_frozen_S99_S102_ONCE_TaigaC_original12.py")
        s=p.read_text(encoding="utf8")
        first=s.index("model99=frozen_child(")
        second=s.index("model102=frozen_child(")
        goldread=s.index("original,aligned,audit=load_source_VERIFY_sha_THEN_OPEN_gold(")
        self.assertLess(first,goldread)
        self.assertLess(second,goldread)
        self.assertLess(s.index("sha(s104_source)==report["),first)
        self.assertNotIn("torch.optim.",s)
        self.assertNotIn("loss.backward()",s)
        self.assertNotIn("opt.step()",s)
        self.assertIn("torch.inference_mode()",s)

    def test_no_UD_parser_or_train_only_spent_old_sets_imported_in_inference(self):
        p=Path("research/calibri_nra3/s97_in_calibri_morphological_12layer_neural_graph.py")
        tree=ast.parse(p.read_text(encoding="utf8"))
        modules=[x.module or "" for x in ast.walk(tree)
                 if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s105" in x.lower() or "s103" in x.lower()
                             or "udpipe" in x.lower() or "stanza" in x.lower()
                             for x in modules))
        q=Path("research/calibri_nra3/s105_prospective_TaigaC_offline_gold_AFTER_original12_SHA.py")
        self.assertIn("report[\"ALL12_CPU_SAFETENSORS_SHA256\"]",q.read_text())
        self.assertIn("Taiga_TRAIN_c_firstread_burned_after_ONE_log_and_NEVER_train",
                      q.read_text())


if __name__=="__main__":
    unittest.main()
