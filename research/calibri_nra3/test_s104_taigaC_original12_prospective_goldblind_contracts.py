"""S104 REAL original Calibri12 + GT protected616 source-only prospective train-c checks."""
import ast
import unittest
from pathlib import Path
from research.calibri_nra3 import s104_TRUE43MGT_firstread_TaigaC_all12_goldblind as s104


class S104SourceFirstReadABI(unittest.TestCase):
    def test_exact_new_precommitted_source_SHA_before_real_CPU(self):
        self.assertEqual(s104.STAGE,
            "CALIBRI_NRA3_S104_FIRSTREAD_TAIGA_TRAINC256_REAL43MGT_ALL12_SOURCEONLY_AFTER616")
        self.assertEqual(s104.SOURCE_SHA,
            "7c960fd28b14449cfb9fa3611576aa21a88dbd6afccbbde4e2a02d015707bcdb")
        self.assertEqual(s104.TENSOR,"NRA3_S104_TAIGA_TRAINC256_FIRSTREAD_ALL12_ORIGINAL_H320_GOLDBLIND.safetensors")
        self.assertEqual(s104.WEIGHTS,"NRA3_S104_IN_CALIBRI_UNTRAINED_28887_FP16_GOLDBLIND.safetensors")

    def test_real_all_twelve_frozen_original_and_independent_H8_port(self):
        text=Path("research/calibri_nra3/s104_TRUE43MGT_firstread_TaigaC_all12_goldblind.py").read_text()
        for fragment in (
            "len(model.model.layers)==N_LAYERS",
            "NRA3MorphologyInCalibriWeights()",
            "child.attach_to_RESEARCH_calibri(model)",
            "child.read_only_all12_source_scope(model)",
            "layer8.capture(model,text,ids)",
            "torch.equal(true_hidden[0,7],capture.hidden320_fp16)",
            "old_calibri_parameter_sha(model)",
            "all_original_43M_GT_CB59_params_sha_pre_and_post_identical",
            "source_ONLY_no_S103_FIRSTREAD_C_UD_gold_loaded_in_original_CPU",
            "old_616_J7_physically_replayed_BEFORE_NRA3_hooks",
        ):
            self.assertIn(fragment,text)

    def test_no_S103_held_gold_module_imported_by_any_original_Calibri_forward(self):
        path=Path("research/calibri_nra3/s104_TRUE43MGT_firstread_TaigaC_all12_goldblind.py")
        t=ast.parse(path.read_text())
        imports=[x.module or "" for x in ast.walk(t) if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s103_precommit" in v or "s102_real12_" in v
                             or "s100_newTRAINd_" in v or "stanza" in v.lower()
                             or "udpipe" in v.lower()
                             for v in imports))
        wrapper=Path("research/nra3_s20/s66_true43m_gt_new32_goldblind_cpu_and_KV_parity.py").read_text()
        self.assertIn('os.environ["NRA3_S104_TAIGA_C_FIRSTREAD_SOURCE_ONLY_JSON"]',wrapper)
        self.assertIn("nra3_s104_capture(",wrapper)
        self.assertIn("nra3_s104_save(",wrapper)


if __name__=="__main__":
    unittest.main()
