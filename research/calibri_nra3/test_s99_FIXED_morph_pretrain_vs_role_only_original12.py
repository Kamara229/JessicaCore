"""NRA3 S99 fixed morphology-then-relational TRAIN-only schedule contracts.

No true original CPU weights, Taiga gold, S96 diagnostic or new sealed
first-read opened here. The actual old616 replay and physical source
S98 are separate CI prerequisites.
"""
import ast
from pathlib import Path
import unittest

import torch

from research.calibri_nra3.s99_REAL12_Taiga_B_morph_pretrain_then_role_28887_ONLY import (
    STAGE,SEED,MORPH_EPOCHS,JOINT_EPOCHS,LR,
    PAID,fixed_8source_stratified_batches,
)
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,
)


class TestS99FixedTrainingOnlySafety(unittest.TestCase):
    def test_exact_preset_three_neural_labels_morphology_then_UD_roles(self):
        self.assertEqual(STAGE,
            "CALIBRI_NRA3_S99_NEW_TAIGA_TRAIN_B_REAL12_MORPH_PRETRAIN_THEN_UD_ROLE_28887_FP16")
        self.assertEqual(SEED,991119)
        self.assertEqual(MORPH_EPOCHS,10)
        self.assertEqual(JOINT_EPOCHS,24)
        self.assertEqual(LR,0.0015)
        self.assertEqual(PAID,28887)
        m=NRA3MorphologyInCalibriWeights()
        self.assertEqual(m.budget()["new_neural_parameters"],28887)
        self.assertEqual(m.budget()["NRA2_independent_mini_areas_preserved"],8)
        self.assertEqual(m.output_gate,0)
        self.assertFalse(m.is_trained)

    def test_fixed_disjoint_TRAIN_48_and_INTERNAL_DEV_16_no_target_ranking(self):
        index=torch.arange(256)
        tr=index%64<48
        dev=index%64>=48
        batches=fixed_8source_stratified_batches(index,tr)
        self.assertEqual(len(batches),24)
        seen=[]
        for batch in batches:
            self.assertEqual(len(batch),8)
            self.assertEqual(
                [int(((batch>=k*64)&(batch<(k+1)*64)).sum())
                 for k in range(4)],[2,2,2,2])
            seen.extend(batch.tolist())
        self.assertEqual(len(seen),192)
        self.assertEqual(len(set(seen)),192)
        self.assertFalse(any(bool(dev[i]) for i in seen))
        self.assertEqual(int(dev.sum()),64)

    def test_developer_diag_alone_cannot_change_training_fixed_schedule(self):
        ix=torch.arange(256)
        tr=ix%64<48
        batches=fixed_8source_stratified_batches(ix,tr)
        self.assertEqual([tuple(b.tolist()) for b in batches],
                         [tuple(b.tolist()) for b in
                          fixed_8source_stratified_batches(ix,tr.clone())])

    def test_neural_model_contains_no_UD_teacher_or_production_activation(self):
        p=Path("research/calibri_nra3/s97_in_calibri_morphological_12layer_neural_graph.py")
        q=Path("research/calibri_nra3/s99_REAL12_Taiga_B_morph_pretrain_then_role_28887_ONLY.py")
        tree=ast.parse(p.read_text())
        mods=[x.module or "" for x in ast.walk(tree)
              if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s99" in x.lower()
                             or "s97_pin" in x.lower()
                             or "stanza" in x.lower()
                             or "udpipe" in x.lower()
                             for x in mods))
        program=q.read_text()
        self.assertIn("physical_source_then_separate_gold(",program)
        self.assertIn("offline_native_morphology_supervision_loss",program)
        self.assertIn("graph_train_only_loss(",program)
        self.assertIn("fixed_8source_stratified_batches(",program)
        self.assertIn('model_forward_only_frozen_source12_no_UD_teacher',program)
        self.assertIn('"output_gate":0',program)
        self.assertIn('"candidate_accepted":False',program)
        self.assertIn('"activation_allowed":False',program)
        self.assertNotIn("S96_train_REAL12",program)


if __name__=="__main__":
    unittest.main()
