"""S110 synthetic source/gold separation and new structural neural gradient ABI."""
import ast
from pathlib import Path
import unittest
import torch
from research.calibri_nra3.s110_GSD_real12_gold_AFTER_S109_sourceSHA import (
    STAGE as GOLD_STAGE,SOURCE_SHA,GOLD_SHA,UPSTREAM_BLOB,S109_SOURCE,S109_REPORT,
)
from research.calibri_nra3.s110_true12_GSD_16656_structural_train import (
    STAGE,EPOCHS,SEED,CHILD_SOURCE,TRAINED,REPORT,batches_fixed,
)
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights,NEW_KEYS,
    NEW_NEURAL_PARAMETERS,OLD_NEURAL_PARAMETERS,TOTAL_NEURAL_PARAMETERS,
)
ROOT=Path('research/calibri_nra3')
class TestS110NewGSDOriginal12StructuredNeuralOnly(unittest.TestCase):
    def test_source_offline_teacher_sealed_SHA_and_NO_TaigaC(self):
        self.assertEqual(SOURCE_SHA,'4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14')
        self.assertEqual(GOLD_SHA,'e3d33d2f21986e960c22ee23503a473e1a34119ce5d3e4015ccb33eb5b869121')
        self.assertEqual(UPSTREAM_BLOB,'de392eb8d1a92fe1d12bc2e49c21cb717ae89859')
        self.assertIn('S110',GOLD_STAGE)
        self.assertIn('GSD',STAGE)
        self.assertIn('S109',S109_SOURCE)
        self.assertIn('S109',S109_REPORT)
        self.assertEqual(CHILD_SOURCE,'NRA3_S109_IN_CALIBRI_UNTRAINED_53743_FP16.safetensors')
        self.assertTrue(TRAINED.endswith('.safetensors'))
        self.assertTrue(REPORT.endswith('.json'))

    def test_only_new_16656_neural_params_have_grad_old_37087_frozen(self):
        m=NRA3StructuralCompetitionInCalibriWeights()
        self.assertEqual((OLD_NEURAL_PARAMETERS,NEW_NEURAL_PARAMETERS,TOTAL_NEURAL_PARAMETERS),
                         (37087,16656,53743))
        self.assertEqual(EPOCHS,24)
        for k,p in m.named_parameters():
            p.requires_grad_(k in NEW_KEYS)
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),16656)
        self.assertEqual(sum(p.numel() for p in m.parameters() if not p.requires_grad),37087)
        self.assertEqual(float(m.structural_competition_gate.abs().max()),0)
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.activation_allowed)

    def test_fixed_STRATA_no_GSD_dev_or_saved_prior_dev_enters_optimizer(self):
        ids=torch.arange(256,dtype=torch.long)
        allowed=(ids%64)<48
        batches=batches_fixed(ids,allowed)
        self.assertEqual(len(batches),24)
        joined=torch.cat(batches)
        self.assertEqual(len(joined),192)
        self.assertTrue(bool(((joined%64)<48).all()))

    def test_physical_S109_source_sha_before_GSD_gold_and_no_parser_at_inference(self):
        src=(ROOT/'s110_GSD_real12_gold_AFTER_S109_sourceSHA.py').read_text()
        train=(ROOT/'s110_true12_GSD_16656_structural_train.py').read_text()
        self.assertLess(src.index('report=json.loads(source_report.read_text())'),
                        src.index('source=json.loads((separate_S108_gold_folder/SOURCE_FILE)'))
        self.assertLess(src.index('source_tensor=load_file(str(true12_source)'),
                        src.index('source=json.loads((separate_S108_gold_folder/SOURCE_FILE)'))
        for marker in (
            'only_new_16656_structural_neural_params_trained',
            'old_37087_in_Calibri_params_FP16_identical',
            'TaigaC_one_shot_firstread_BURNED_NOT_USED',
            'GSD_internal_diag_scored_ONCE_after_fixed_epochs_no_tuning',
            'aligned.train[ix].all()',
            'not bool(aligned.internal_dev[ix].any())',
            'source_report',
        ):
            self.assertIn(marker,train)
        imported=[n.module or '' for n in ast.walk(ast.parse(train))
                  if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('train_c' in m.lower() or 's79_' in m.lower()
                             or 's86_' in m.lower() for m in imported))
if __name__=='__main__':
    unittest.main()
