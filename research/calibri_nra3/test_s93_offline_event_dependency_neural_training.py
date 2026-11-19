"""S93 eight synthetic tests of training ONLY neural Calibri graph weights.

Not an UD accuracy, multi-layer effective transfer or old616 proof.
"""
import ast
import unittest
from pathlib import Path

import torch
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    NRA3TrainOnlyTargets,graph_train_only_loss,IGNORE,UNKNOWN,STAGE,
)


def make():
    torch.manual_seed(9301)
    model=NRA3InCalibriNeuralWeights()
    hidden=torch.randn(2,12,7,320)
    valid=torch.tensor([[True]*7,[True]*5+[False]*2])
    graph=model(hidden,valid)
    mention=torch.full((2,7),IGNORE,dtype=torch.long)
    mention[0,2]=1
    mention[0,4]=0
    mention[1,1]=1
    role=torch.full((2,8,7),IGNORE,dtype=torch.long)
    role[0,0,2]=0   # event at 2 actor at 0
    role[0,1,2]=3   # event at 2 patient at 3
    role[1,0,1]=2   # passive event at 1 actor at 2
    role[1,1,1]=3   # passive event patient at 3
    state=torch.full((2,7,6),float(UNKNOWN))
    state[0,2,0]=1
    state[0,2,1]=0
    state[1,1,5]=1
    areas=torch.full((2,7,8),float(UNKNOWN))
    areas[0,2,0]=0
    areas[0,2,1]=1
    areas[1,1,2]=1
    areas[1,1,7]=0
    targets=NRA3TrainOnlyTargets(mention,role,state,areas,valid)
    return model,graph,targets


class TestS93OfflineGrammaticalLoss(unittest.TestCase):
    def test_new_neural_relation_loss_finite_and_all_graph_heads_trainable(self):
        model,graph,targets=make()
        loss,parts=graph_train_only_loss(graph,targets)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertEqual(parts["annotated_event_positions"],2)
        self.assertEqual(parts["annotated_role_edges"],4)
        self.assertEqual(parts["annotated_state_bits"],3)
        self.assertEqual(parts["annotated_mini_area_bits"],4)
        self.assertTrue(parts["unannotated_ignored_by_loss"])
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertFalse(model.activation_allowed)
        self.assertEqual(model.output_gate,0)

    def test_state_and_micro_area_UNKNOWN_not_supervised_negative(self):
        model,graph,target=make()
        _,old=graph_train_only_loss(graph,target)
        state=target.state.clone()
        state[0,3,0]=0  # NEW explicit labeled negative; previous unknown
        changed=NRA3TrainOnlyTargets(
            target.mention,target.role_argument_token,state,
            target.mini_area,target.original_native_valid)
        _,new=graph_train_only_loss(graph,changed)
        self.assertEqual(old["annotated_state_bits"],3)
        self.assertEqual(new["annotated_state_bits"],4)

    def test_role_edges_never_use_invalid_original_BPE(self):
        _,graph,t=make()
        role=t.role_argument_token.clone()
        role[1,0,1]=6  # padded source in row 1
        bad=NRA3TrainOnlyTargets(t.mention,role,t.state,t.mini_area,
                                t.original_native_valid)
        with self.assertRaisesRegex(ValueError,"source-native BPE valid"):
            graph_train_only_loss(graph,bad)

    def test_role_gold_requires_annotated_true_event_not_unknown(self):
        _,graph,t=make()
        m=t.mention.clone();m[0,2]=IGNORE
        with self.assertRaisesRegex(ValueError,"annotated EVENT"):
            graph_train_only_loss(graph,NRA3TrainOnlyTargets(
                m,t.role_argument_token,t.state,t.mini_area,
                t.original_native_valid))

    def test_optional_snapshot_penalty_on_in_calibri_slots(self):
        model,graph,t=make()
        old=model.event_memory_slots.detach().clone()
        with torch.no_grad():
            model.event_memory_slots.add_(0.03)
        loss,p=graph_train_only_loss(graph,t,old_memory_snapshot=old,
                                    new_neural_model=model)
        self.assertIsNotNone(p["optional_memory_prototype_distillation"])
        self.assertGreater(float(p["optional_memory_prototype_distillation"]),0.)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_mismatch_source_native_alignment_rejected(self):
        _,graph,t=make()
        mask=t.original_native_valid.clone();mask[0,-1]=False
        bad=NRA3TrainOnlyTargets(t.mention,t.role_argument_token,
                                t.state,t.mini_area,mask)
        with self.assertRaisesRegex(ValueError,"native BPE supervised labels"):
            graph_train_only_loss(graph,bad)

    def test_all_labels_zero_requires_explicit_annotations_not_false_negative(self):
        _,graph,t=make()
        m=torch.full_like(t.mention,IGNORE)
        role=torch.full_like(t.role_argument_token,IGNORE)
        state=torch.full_like(t.state,UNKNOWN)
        mini=torch.full_like(t.mini_area,UNKNOWN)
        targets=NRA3TrainOnlyTargets(m,role,state,mini,t.original_native_valid)
        with self.assertRaisesRegex(ValueError,"explicitly annotated TRAIN"):
            graph_train_only_loss(graph,targets)

    def test_source_only_neural_forward_never_imports_gold_or_offline_teacher(self):
        path=Path("research/calibri_nra3/s91_neural_multilayer_event_graph.py")
        source=path.read_text(encoding="utf8")
        tree=ast.parse(source)
        imports=[n.module or "" for n in tree.body if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any("s93_offline" in x or "stanza" in x or
                             "udpipe" in x or "gold" in x.lower()
                             for x in imports))
        self.assertEqual(STAGE,
            "CALIBRI_NRA3_S93_OFFLINE_TEACHER_EVENT_DEPENDENCY_TRAIN_ONLY")
        self.assertNotIn("test_s86",source.lower())


if __name__=="__main__":
    unittest.main()
