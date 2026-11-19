"""NRA3 S93: offline TRAIN-only native-BPE graph-supervision ABI.

Tests are synthetic tensor contracts, NOT morphology/grammar accuracy.
Actual Calibri43M+GT 12-layer source was physically captured in S92B,
but it is never loaded here. S79/S86 one-time firstreads are SPENT.
"""
import ast
from dataclasses import replace
from pathlib import Path
import unittest

import torch

from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,N_ROLES,N_STATES,N_MICRO,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    NRA3TrainOnlyTargets,graph_train_only_loss,IGNORE,UNKNOWN,
    STAGE,ROLE_WEIGHT,
)


def fixtures():
    torch.manual_seed(931119)
    neural=NRA3InCalibriNeuralWeights()
    hidden=torch.randn(2,12,6,320)
    valid=torch.tensor([[True]*6,[True]*5+[False]],dtype=torch.bool)
    graph=neural(hidden,valid)
    mention=torch.full((2,6),IGNORE,dtype=torch.long)
    # Synthetic EXPLICIT labels; all other event slots are UNKNOWN,
    # not assumed to be a non-event.
    mention[0,1]=1
    mention[1,2]=1
    mention[0,4]=0
    role=torch.full((2,N_ROLES,6),IGNORE,dtype=torch.long)
    role[0,0,1]=2   # first event -- actor native-BPE token2
    role[1,2,2]=3   # second event -- recipient native-BPE token3
    state=torch.full((2,6,N_STATES),float(UNKNOWN),dtype=torch.float32)
    state[0,1,1]=1
    state[1,2,2]=0
    areas=torch.full((2,6,N_MICRO),float(UNKNOWN),dtype=torch.float32)
    areas[0,1,0]=1
    areas[1,2,1]=0
    truth=NRA3TrainOnlyTargets(
        mention=mention,role_argument_token=role,
        state=state,mini_area=areas,original_native_valid=valid)
    return neural,graph,truth


class TestS93StrictOfflineTrainOnly(unittest.TestCase):
    def test_exact_neural_role_state_eight_mini_area_train_supervision(self):
        neural,graph,truth=fixtures()
        loss,parts=graph_train_only_loss(graph,truth)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["annotated_event_positions"],2)
        self.assertEqual(parts["annotated_role_edges"],2)
        self.assertEqual(parts["annotated_state_bits"],2)
        self.assertEqual(parts["annotated_mini_area_bits"],2)
        self.assertTrue(parts["unannotated_ignored_by_loss"])
        self.assertTrue(parts["labels_only_offline_training_not_in_model_forward"])
        self.assertFalse(parts["activation_allowed"])
        loss.backward()
        self.assertIsNotNone(neural.layer_router_logits.grad)
        self.assertIsNotNone(neural.role_bilinear.grad)
        self.assertIsNotNone(neural.event_memory_slots.grad)
        self.assertIsNotNone(neural.mini_areas[list(neural.mini_areas)[0]][0].weight.grad)
        for key in ("layer_router_logits","role_bilinear","event_memory_slots"):
            self.assertTrue(torch.isfinite(dict(neural.named_parameters())[key].grad).all())

    def test_unknown_unsupervised_logits_never_change_loss(self):
        _,graph,truth=fixtures()
        base,_=graph_train_only_loss(graph,truth)
        mention=graph.event_mention_logits.clone()
        mention[1,0]+=50.
        role=graph.role_arc_logits.clone()
        role[0,:,4,:]+=100.
        state=graph.event_state_logits.clone()
        state[0,0,:]+=40.
        areas=graph.mini_area_logits.clone()
        areas[0,0,:]+=80.
        altered=replace(graph,event_mention_logits=mention,
                        role_arc_logits=role,event_state_logits=state,
                        mini_area_logits=areas)
        modified,_=graph_train_only_loss(altered,truth)
        self.assertTrue(torch.allclose(base,modified,atol=1e-6,rtol=1e-6))

    def test_role_target_must_be_valid_original_native_BPE(self):
        _,graph,truth=fixtures()
        invalid=truth.role_argument_token.clone()
        invalid[1,2,2]=5  # row1 BPE5 is padded, NOT a source argument
        with self.assertRaisesRegex(ValueError,"native BPE valid"):
            graph_train_only_loss(graph,replace(truth,role_argument_token=invalid))

    def test_role_event_anchor_must_be_explicitly_positive(self):
        _,graph,truth=fixtures()
        mentions=truth.mention.clone()
        mentions[0,1]=IGNORE
        with self.assertRaisesRegex(ValueError,"annotated EVENT"):
            graph_train_only_loss(graph,replace(truth,mention=mentions))

    def test_unknown_labels_never_silently_equate_to_zero(self):
        _,graph,truth=fixtures()
        all_unknown=truth.mention.clone()
        all_unknown[:]=IGNORE
        with self.assertRaisesRegex(ValueError,"explicitly annotated TRAIN"):
            graph_train_only_loss(graph,replace(truth,mention=all_unknown))
        modified=truth.mini_area.clone()
        modified[0,0,0]=2.
        with self.assertRaisesRegex(ValueError,"unknown sentinel"):
            graph_train_only_loss(graph,replace(truth,mini_area=modified))

    def test_optional_neural_memory_consolidation_is_distinct_train_only_loss(self):
        neural,graph,truth=fixtures()
        frozen=neural.event_memory_slots.detach().clone()
        base,_=graph_train_only_loss(graph,truth)
        with torch.no_grad():
            neural.event_memory_slots.add_(0.2)
        plus,parts=graph_train_only_loss(
            graph,truth,old_memory_snapshot=frozen,new_neural_model=neural)
        self.assertGreater(float(plus),float(base))
        self.assertGreater(float(parts["optional_memory_prototype_distillation"]),0)
        with self.assertRaisesRegex(ValueError,"SAME in-Calibri neural child"):
            graph_train_only_loss(graph,truth,old_memory_snapshot=frozen)

    def test_no_external_teacher_or_spent_source_import_inside_neural_forward(self):
        neural_path=Path("research/calibri_nra3/s91_neural_multilayer_event_graph.py")
        teacher_path=Path("research/calibri_nra3/s93_offline_ud_grammatical_role_train_loss_ONLY.py")
        forward_tree=ast.parse(neural_path.read_text(encoding="utf8"))
        imports=[node.module or "" for node in ast.walk(forward_tree)
                 if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any("s93_offline_ud" in p or
                             "s79_NEW_NOVEL48" in p or
                             "s86_NOVEL48" in p or
                             "stanza" in p or "udpipe" in p
                             for p in imports))
        teacher=ast.parse(teacher_path.read_text(encoding="utf8"))
        names=[x.name for x in teacher.body if isinstance(x,ast.FunctionDef)]
        self.assertEqual(names,["require","graph_train_only_loss"])
        self.assertEqual(
            STAGE,"CALIBRI_NRA3_S93_OFFLINE_TEACHER_EVENT_DEPENDENCY_TRAIN_ONLY")
        self.assertEqual(ROLE_WEIGHT,2.0)
        self.assertEqual(neural_path.exists(),True)


if __name__=="__main__":
    unittest.main()
