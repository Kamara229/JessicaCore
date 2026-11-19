"""S115: unscored SynTagRus dev, NC license and only 8 neural trust gradients."""
import ast
from pathlib import Path
import unittest
import torch
from research.calibri_nra3.s115_SynTagRusNC_original12_GOLD_AFTER_S114_SHA import (
    STAGE as GOLD_STAGE,SOURCE_SHA,GOLD_SHA,UPSTREAM_BLOB,S114_SOURCE,S114_REPORT,
)
from research.calibri_nra3.s115_original12_SynTagRusNC_ONLY8_trust_neural_TRAIN import (
    STAGE,EPOCHS,CHILD,TRAINED,REPORT,batches_fixed,
)
from research.calibri_nra3.s112_in_calibri_bounded_structural_trust_53751 import (
    NRA3BoundedStructuralTrustInCalibriWeights,TOTAL,OLD,NEW,
)

class TestS115True12NewNCResearchTrustOnly(unittest.TestCase):
    def test_new_NC_gold_after_physical_source_SHAs(self):
        self.assertEqual(SOURCE_SHA,'e9972e4f2204887cd1a507289dc0e4936f18c9f8ec9e18a0a2630e1009a1f209')
        self.assertEqual(GOLD_SHA,'a7d41b9b6925c2ad9b2051294813a7af91c963dfae534e3ceca7bf3e4eeb7395')
        self.assertEqual(UPSTREAM_BLOB,'eb7638bf3680414e49123412962e511b998a3b01')
        self.assertIn('SYNTAGRUS',GOLD_STAGE)
        self.assertIn('SYNTAGRUS',STAGE)
        self.assertIn('S114',S114_SOURCE)
        self.assertIn('S114',S114_REPORT)
        self.assertEqual(CHILD,'NRA3_S114_IN_CALIBRI_ZERO_TRUST_53751_FP16.safetensors')
        self.assertTrue(TRAINED.endswith('.safetensors'))
        self.assertTrue(REPORT.endswith('.json'))

    def test_only_eight_in_calibri_new_neural_trust_parameters_train(self):
        n=NRA3BoundedStructuralTrustInCalibriWeights()
        self.assertEqual((OLD,NEW,TOTAL),(53743,8,53751))
        self.assertEqual(EPOCHS,24)
        self.assertEqual(n.budget()['new_neural_parameters'],53751)
        for k,p in n.named_parameters():
            p.requires_grad_(k=='structural_trust_gate')
        self.assertEqual(sum(p.numel() for p in n.parameters() if p.requires_grad),8)
        self.assertEqual(sum(p.numel() for p in n.parameters() if not p.requires_grad),53743)
        self.assertEqual(n.output_gate,0)
        self.assertFalse(n.activation_allowed)

    def test_developer_last16_never_used_by_fixed_train_optimizer(self):
        idx=torch.arange(256,dtype=torch.long)
        train=(idx%64)<48
        b=batches_fixed(idx,train)
        self.assertEqual(len(b),24)
        selected=torch.cat(b)
        self.assertEqual(len(selected),192)
        self.assertTrue(bool(((selected%64)<48).all()))

    def test_source_SHA_checked_before_gold_and_only_TRAIN_diagnostics(self):
        a=Path('research/calibri_nra3/s115_SynTagRusNC_original12_GOLD_AFTER_S114_SHA.py').read_text()
        b=Path('research/calibri_nra3/s115_original12_SynTagRusNC_ONLY8_trust_neural_TRAIN.py').read_text()
        self.assertLess(a.index('require(hashlib.sha256(true12_source.read_bytes()).hexdigest()=='),
                        a.index('source=json.loads((separate_S113_gold_folder/SOURCE_FILE)'))
        for marker in ('S113_teacher_gold_opened_ONLY_AFTER_S114_original12_source_SHA',
                       'SynTagRus_INTERNAL_16per64_NEVER_SCORED_OR_TUNED',
                       'TaigaC_one_shot_firstread_ALREADY_BURNED_NOT_USED',
                       'NONCOMMERCIAL_research_only_NO_model_licensing_commercial_use',
                       'labels.train[ix].all()',
                       'not bool(labels.internal_dev[ix].any())',
                       'ALL_PREVIOUS_53743_NRA2_mini8_memory16_router12_structural_FP16_EXACT'):
            self.assertIn(marker,b)
        self.assertNotIn('labels,labels.internal_dev)',b)
        imported=[x.module or '' for x in ast.walk(ast.parse(b)) if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any('train_c' in x.lower() or 's86_' in x.lower() for x in imported))

if __name__=='__main__':
    unittest.main()
