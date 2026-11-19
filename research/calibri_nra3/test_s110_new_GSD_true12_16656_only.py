"""S110 synthetic source→GOLD + immutable neural-training control, no live or old DEV."""
from pathlib import Path
import ast
import unittest
import torch
from research.calibri_nra3.s110_original12_GSD_gold_after_S109_SHA import (
    STAGE,SOURCE_SHA,GOLD_SHA,UPSTREAM_BLOB,S109_SOURCE,S109_REPORT,
)
from research.calibri_nra3.s110_GSD_real12_ONLY_16656_structural_train import (
    STAGE as TRAIN_STAGE,EPOCHS,LR,WD,
    OLD_NEURAL_PARAMETERS,NEW_NEURAL_PARAMETERS,TOTAL_NEURAL_PARAMETERS,
    NEW_KEYS,SEED,batches_fixed,
)
from research.calibri_nra3.s107_in_calibri_structural_role_competition_53743 import (
    NRA3StructuralCompetitionInCalibriWeights,
)

class TestS110NewGSDTrainOnlyAndOriginal16kInModel(unittest.TestCase):
    def test_new_GSD_original_SHA_locked_separate_labels_and_original12(self):
        self.assertEqual(SOURCE_SHA,'4078f408f2355c55c29bca18504355379e2a42ef2144c9113dda9448b086de14')
        self.assertEqual(GOLD_SHA,'e3d33d2f21986e960c22ee23503a473e1a34119ce5d3e4015ccb33eb5b869121')
        self.assertEqual(UPSTREAM_BLOB,'de392eb8d1a92fe1d12bc2e49c21cb717ae89859')
        self.assertIn('S110',STAGE)
        self.assertIn('GSD',TRAIN_STAGE)
        self.assertEqual(S109_SOURCE,'NRA3_S109_GSD_TRAIN256_ALL12_ORIGINAL_H320_NO_GOLD.safetensors')
        self.assertEqual(S109_REPORT,'NRA3_S109_REAL43MGT_GSD_TRAIN256_ALLVERBS_SOURCE_PROVENANCE.json')

    def test_16656_structural_gradients_only_in_CALIBRI_old37087_retained(self):
        self.assertEqual((OLD_NEURAL_PARAMETERS,NEW_NEURAL_PARAMETERS,TOTAL_NEURAL_PARAMETERS),
                         (37087,16656,53743))
        m=NRA3StructuralCompetitionInCalibriWeights()
        self.assertEqual(m.budget()['new_neural_parameters'],53743)
        for k,p in m.named_parameters():
            p.requires_grad_(k in NEW_KEYS)
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),16656)
        self.assertEqual(sum(p.numel() for p in m.parameters() if not p.requires_grad),37087)
        self.assertEqual(len(NEW_KEYS),5)
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.candidate_accepted)
        self.assertFalse(m.activation_allowed)

    def test_fixed_20_epochs_train_rows_0_to_47_per_stratum_only(self):
        self.assertEqual((SEED,EPOCHS,LR,WD),(1101119,20,0.0012,0.01))
        ids=torch.arange(256,dtype=torch.long)
        chosen=torch.ones(256,dtype=torch.bool)
        batches=batches_fixed(ids,chosen)
        self.assertEqual(len(batches),24)
        all_ids=torch.cat(batches).tolist()
        self.assertEqual(len(all_ids),192)
        self.assertTrue(all((i%64)<48 for i in all_ids))
        self.assertFalse(any((i%64)>=48 for i in all_ids))

    def test_no_spent_DEV_interpretation_before_frozen_20_epochs(self):
        a=Path('research/calibri_nra3/s110_GSD_real12_ONLY_16656_structural_train.py').read_text()
        b=Path('research/calibri_nra3/s110_original12_GSD_gold_after_S109_SHA.py').read_text()
        self.assertLess(a.index('for epoch in range(EPOCHS):'),
                        a.index('teacher_anchored_diagnostic(prev,hidden,aligned,aligned.internal_dev)'))
        self.assertIn('aligned.train[ix].all()',a)
        self.assertIn('not bool(aligned.internal_dev[ix].any())',a)
        self.assertIn('all_prior_37087_FP16_neural_weights_exact',a)
        self.assertIn('Taiga_TRAIN_c_new_prospective_ONE_SHOT_reserved_unopened',a)
        self.assertIn('all_original_eight_NRA2_neural_mini_areas_unchanged',a)
        self.assertIn('new_NRA3_structural_in_CALIBRI_neural_child_53743',b)
        self.assertLess(b.index('report=json.loads(source_report.read_text())'),
                        b.index('source=json.loads((separate_S108_gold_folder/SOURCE_FILE)'))
        modules=[n.module or '' for n in ast.walk(ast.parse(a))
                 if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('s79_' in z.lower() or 's86_' in z.lower() or
                             'train_c' in z.lower() for z in modules))

if __name__=='__main__':
    unittest.main()
