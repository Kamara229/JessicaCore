"""S110 synthetic ABI: GSD source/GOLD hard wall, neural-only 16,656 gradients."""
import ast
from pathlib import Path
import unittest
import torch
from research.calibri_nra3.s110_native_BPE_GSD_gold_AFTER_S109_sourceSHA import (
    STAGE,SOURCE_SHA,GOLD_SHA,UPSTREAM_BLOB,S109_SOURCE,S109_REPORT,
)
from research.calibri_nra3.s110_GSD_16656_structural_neural_TRAIN_real12 import (
    STAGE as TRAIN_STAGE,OLD_PARAMS,TRAINED_NEW_PARAMS,TOTAL_PARAMS,
    EPOCHS,NEW_KEYS,batches_fixed,
)
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights,
)
R=Path("research/calibri_nra3")

class TestS110StrictGSDTrue12NewStructuralWeights(unittest.TestCase):
    def test_pin_different_new_GSD_TRAIN_only_and_real_originalS109(self):
        self.assertEqual(SOURCE_SHA,"4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14")
        self.assertEqual(GOLD_SHA,"407ee8a5ca720ebad96379859e647623aa154e32277376f24e29be6681a1fb0b")
        self.assertEqual(UPSTREAM_BLOB,"de392eb8d1a92fe1d12bc2e49c21cb717ae89859")
        self.assertIn("S110",STAGE)
        self.assertIn("S110",TRAIN_STAGE)
        self.assertIn("S109_GSD_TRAIN256",S109_SOURCE)
        self.assertIn("S109",S109_REPORT)
    def test_new_16656_inside_calibri_eight_NRA2_areas_retained(self):
        self.assertEqual((OLD_PARAMS,TRAINED_NEW_PARAMS,TOTAL_PARAMS),(37087,16656,53743))
        self.assertEqual(EPOCHS,24)
        m=NRA3StructuralCompetitionInCalibriWeights()
        self.assertEqual(m.budget()["new_neural_parameters"],53743)
        for name,p in m.named_parameters():
            p.requires_grad_(name in NEW_KEYS)
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),16656)
        self.assertEqual(sum(p.numel() for p in m.parameters() if not p.requires_grad),37087)
        self.assertEqual(m.budget()["NRA2_eight_original_mini_areas_preserved"],True)
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.activation_allowed)
    def test_first_48_per_64_each_stratum_only(self):
        indices=torch.arange(256)
        mask=torch.ones(256,dtype=torch.bool)
        batches=batches_fixed(indices,mask)
        self.assertEqual(len(batches),24)
        got=torch.cat(batches)
        self.assertEqual(len(got),192)
        self.assertTrue(all(int(x)%64<48 for x in got))
    def test_no_model_forward_before_true_source_SHA_then_offline_gold(self):
        mapper=(R/"s110_native_BPE_GSD_gold_AFTER_S109_sourceSHA.py").read_text()
        train=(R/"s110_GSD_16656_structural_neural_TRAIN_real12.py").read_text()
        self.assertLess(mapper.index("source_tensor=load_file(str(true12_source)"),
                        mapper.index("source=json.loads((separate_S108_gold_folder/SOURCE_FILE)"))
        self.assertLess(mapper.index("report=json.loads(source_report.read_text())"),
                        mapper.index("source=json.loads((separate_S108_gold_folder/SOURCE_FILE)"))
        for marker in (
            "all_prior_37087_FP16_NRA2_mini_areas_router_morph_memory_preserved",
            "GSD_UD_gold_opened_ONLY_AFTER_S109_original12_SHA_verified",
            "GSD_same_corpus_INTERNAL_developer_now_SPENT_NO_retraining_tuning",
            "entire_official_Taiga_TRAIN_c_prospective_RESERVED_UNOPENED",
            "only_NEW_16656_structural_predicate_arg_neural_params_trained",
            "aligned.train[ix]","not bool(aligned.internal_dev[ix].any())"):
            self.assertIn(marker,train)
        imported=[n.module or "" for n in ast.walk(ast.parse(train))
                  if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any("s106_TaigaE_TRAINONLY" in m
                             or "s102_real12_TaigaTRAINd" in m
                             or "train_c" in m.lower()
                             for m in imported))
if __name__=="__main__":
    unittest.main()
