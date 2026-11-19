"""S105 ONLY synthetic no-gold contracts: no old dev, no prospective TRAIN-c."""
import ast
from pathlib import Path
import unittest
import torch
from research.calibri_nra3.s105_original12_TaigaE_UD_gold_after_S104_SHA import (
    STAGE,SOURCE_SHA,GOLD_SHA,UPSTREAM_BLOB,S104_SOURCE,S104_REPORT,
)
from research.calibri_nra3.s105_TaigaE_S104_real12_only_8200_affinity_train import (
    OLD_PARAMS,EXTRA_PARAMS,TOTAL_PARAMS,EPOCHS,NEW_ONLY,S102_SHA,
    batches_fixed,STAGE as TRAIN_STAGE,
)
from research.calibri_nra3.s103_in_calibri_morphosyntactic_affinity_37087 import (
    NRA3RelationalMorphologyInCalibriWeights,
)
ROOT=Path('research/calibri_nra3')

class TestS105TrainOnlyGoldWallAndOldWeightProtection(unittest.TestCase):
    def test_pinned_immutable_Taiga_E_gold_shas_and_real_S104_source(self):
        self.assertEqual(SOURCE_SHA,'5fa1c234cf2bd53aee62eb8c4faba9a8ec116c0e70f7b8c5e2b13a11f4d574fc')
        self.assertEqual(GOLD_SHA,'6f0c05aa988c500693d088a4147159ad106b10601e388ee317690fab7570a2bb')
        self.assertEqual(UPSTREAM_BLOB,'8df5a8b13b934576aa50b7fb55f8a16cb0151003')
        self.assertIn('TAIGAE',STAGE)
        self.assertIn('TAIGA_TRAINE256',S104_SOURCE)
        self.assertIn('S104',S104_REPORT)
        self.assertIn('TAIGA_E',TRAIN_STAGE)
        self.assertEqual(S102_SHA,'f40dddbc03282a6c45c4f807746275d76a4ba776aa613247e73fbdf7f43329df')

    def test_only_new_8200_neural_affinity_and_gate_receive_gradient(self):
        self.assertEqual((OLD_PARAMS,EXTRA_PARAMS,TOTAL_PARAMS),(28887,8200,37087))
        self.assertEqual(EPOCHS,24)
        m=NRA3RelationalMorphologyInCalibriWeights()
        self.assertEqual(m.budget()['new_neural_parameters'],TOTAL_PARAMS)
        self.assertEqual(NEW_ONLY,frozenset({'role_morph_affinity','role_morph_affinity_gate'}))
        for name,p in m.named_parameters():
            p.requires_grad_(name in NEW_ONLY)
        self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),8200)
        self.assertEqual(sum(p.numel() for p in m.parameters() if not p.requires_grad),28887)
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.activation_allowed)

    def test_fixed_train_index_splits_never_see_48_to_63_dev(self):
        source_ids=torch.arange(256,dtype=torch.long)
        included=torch.ones(256,dtype=torch.bool)
        batches=batches_fixed(source_ids,included)
        self.assertEqual(len(batches),24)
        selected=torch.cat(batches)
        self.assertTrue(all((int(v)%64)<48 for v in selected))
        self.assertEqual(len(selected),192)

    def test_no_teacher_forward_no_heldout_and_physical_SHA_before_gold(self):
        mapper=(ROOT/'s105_original12_TaigaE_UD_gold_after_S104_SHA.py').read_text()
        trainer=(ROOT/'s105_TaigaE_S104_real12_only_8200_affinity_train.py').read_text()
        self.assertLess(mapper.index('report=json.loads(source_report.read_text())'),
                        mapper.index('source=json.loads((separate_S103_gold_folder/SOURCE_FILE)'))
        self.assertLess(mapper.index('source_tensor=load_file(str(true12_source)'),
                        mapper.index('source=json.loads((separate_S103_gold_folder/SOURCE_FILE)'))
        for x in ('only_new_8200_neural_affinity_parameters_trained',
                  'all_prior_28887_neural_weights_remained_FP16_exact',
                  'prospective_Taiga_TRAIN_c_RESERVED_UNOPENED',
                  'labels.train[ix]','not bool(labels.internal_dev[ix].any())'):
            self.assertIn(x,trainer)
        from_mods=[n.module or '' for n in ast.walk(ast.parse(trainer))
                   if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('train_c' in m.lower() or
                             's79_' in m.lower() or
                             's86_' in m.lower()
                             for m in from_mods))
        self.assertIn('S103_offline_gold_opened_ONLY_after_physical_S104_source_SHA_verified',trainer)

if __name__=='__main__':
    unittest.main()
