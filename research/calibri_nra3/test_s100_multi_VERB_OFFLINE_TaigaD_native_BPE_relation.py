"""S100 TRAIN-d all-verb gram-dependency labels and in-Calibri multi-edge ABI.

SYNTHETIC ONLY. These tests do not benchmark original model,
Russian grammatical transfer or retained skills. Held original S96
development, S79/S86 firstreads and Taiga dev/test never imported.
"""
import ast
from dataclasses import replace
from pathlib import Path
import unittest

import torch

from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import UDNode
from research.calibri_nra3.s91_neural_multilayer_event_graph import ROLE_NAMES
from research.calibri_nra3.s97_in_calibri_morphological_12layer_neural_graph import (
    NRA3MorphologyInCalibriWeights,
)
from research.calibri_nra3.s100_newTRAINd_all_VERB_dependencies_multiedge_SOURCE_GOLD_wall import (
    STAGE as SOURCE_STAGE,TRAIN_FILE,TRAIN_GIT_BLOB,
    all_ud_verbs_roles_with_multiedges,offline_allword_morphology,
)
from research.calibri_nra3.s100_offline_allverb_multiargument_nativeBPE_train_loss import (
    STAGE as LOSS_STAGE,OriginalBPEAllVerbsEdges,
    offline_all_verb_multi_positive_relation_loss,
)


def node(index,form,upos,head,deprel,feats="_",lemma=None):
    return UDNode(index,form,lemma or form,upos,feats,head,deprel,"_")


def two_predicate_nodes():
    # Synthetic UD word tree, not actual real linguistic benchmark.
    return (
        node(1,"Курьер","NOUN",2,"nsubj","Case=Nom|Number=Sing"),
        node(2,"передал","VERB",0,"root","VerbForm=Fin|Voice=Act"),
        node(3,"письмо","NOUN",2,"obj","Case=Acc|Number=Sing"),
        node(4,"и","CCONJ",5,"cc"),
        node(5,"подписал","VERB",2,"conj","VerbForm=Fin|Voice=Act"),
        node(6,"водитель","NOUN",5,"nsubj","Case=Nom|Number=Sing"),
        node(7,"акт","NOUN",5,"obj","Case=Acc|Number=Sing"),
    )


def mock_graph():
    torch.manual_seed(100020)
    network=NRA3MorphologyInCalibriWeights()
    h=torch.randn(1,12,6,320)
    valid=torch.tensor([[True]*5+[False]])
    graph=network(h,valid)
    pos=torch.zeros(1,len(ROLE_NAMES),6,6,dtype=torch.bool)
    actor=ROLE_NAMES.index("actor");patient=ROLE_NAMES.index("patient")
    pos[0,actor,1,0]=True
    pos[0,patient,1,2]=True
    pos[0,patient,1,3]=True  # TWO same-role positives
    mention=torch.zeros(1,6,dtype=torch.bool)
    mention[0,1]=True
    return network,graph,OriginalBPEAllVerbsEdges(
        native_BPE_valid=valid,argument_gold_positive=pos,
        event_role_supervised=pos.any(-1),
        event_verb_positive_only=mention)


class TestS100MultiVerbRelationResearch(unittest.TestCase):
    def test_all_predicates_not_just_first_verbal_event(self):
        graph=all_ud_verbs_roles_with_multiedges(two_predicate_nodes())
        self.assertEqual(len(graph),2)
        self.assertEqual([x["predicate_UD_word_zero_index_TRAIN_ONLY"] for x in graph],[1,4])
        e1=graph[0]["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]
        e2=graph[1]["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]
        self.assertIn({"role":"actor","arg_ud_word":0},e1)
        self.assertIn({"role":"patient","arg_ud_word":2},e1)
        self.assertIn({"role":"actor","arg_ud_word":5},e2)
        self.assertIn({"role":"patient","arg_ud_word":6},e2)

    def test_multiedge_NEVER_collapse_same_role_arguments(self):
        nodes=[
            node(1,"Курьер","NOUN",2,"nsubj"),
            node(2,"передал","VERB",0,"root"),
            node(3,"письмо","NOUN",2,"obj"),
            node(4,"договор","NOUN",2,"obj"),
        ]
        g=all_ud_verbs_roles_with_multiedges(nodes)
        self.assertEqual(len(g),1)
        pairs=g[0]["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]
        self.assertIn({"role":"patient","arg_ud_word":2},pairs)
        self.assertIn({"role":"patient","arg_ud_word":3},pairs)
        self.assertEqual(sum(x["role"]=="patient" for x in pairs),2)

    def test_actual_UD_voice_negator_and_passive_role_reversal(self):
        nodes=[
            node(1,"Письмо","NOUN",2,"nsubj:pass","Case=Nom"),
            node(2,"передано","VERB",0,"root","VerbForm=Part|Voice=Pass"),
            node(3,"не","PART",2,"advmod",lemma="не"),
            node(4,"курьером","NOUN",2,"obl:agent","Case=Ins"),
        ]
        events=all_ud_verbs_roles_with_multiedges(nodes)
        self.assertEqual(len(events),1)
        self.assertTrue(events[0]["UD_voice_passive_TRAIN_ONLY"])
        self.assertTrue(events[0]["UD_negation_particle_link_present_TRAIN_ONLY"])
        links=events[0]["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]
        self.assertIn({"role":"patient","arg_ud_word":0},links)
        self.assertIn({"role":"actor","arg_ud_word":3},links)
        self.assertIn({"role":"negation_scope","arg_ud_word":2},links)

    def test_all_UD_word_morphology_without_fabricating_absent_case_voice(self):
        m=offline_allword_morphology(two_predicate_nodes())
        self.assertEqual(len(m["offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY"]),7)
        self.assertEqual(m["offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY"][1],"VERB")
        self.assertEqual(m["offline_Case_ALL_UD_WORDS_TRAIN_ONLY"][0],"Nom")
        self.assertEqual(m["offline_Case_ALL_UD_WORDS_TRAIN_ONLY"][1],None)
        self.assertEqual(m["offline_Voice_ALL_UD_WORDS_TRAIN_ONLY"][1],"Act")
        self.assertEqual(m["offline_Voice_ALL_UD_WORDS_TRAIN_ONLY"][0],None)

    def test_real_CALIBRI_neural_graph_2_same_role_positive_arguments_kept(self):
        net,graph,target=mock_graph()
        loss,report=offline_all_verb_multi_positive_relation_loss(graph,target)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.detach()),0)
        self.assertEqual(report["n_source_BPE_verbs_explicit"],1)
        self.assertEqual(report["n_supervised_event_role_slots"],2)
        self.assertEqual(report["n_supervised_event_to_argument_edges"],3)
        self.assertTrue(report["multiple_distinct_role_args_preserved"])
        self.assertTrue(report["unknown_event_role_slots_NEVER_negative"])
        loss.backward()
        self.assertIsNotNone(net.layer_router_logits.grad)
        self.assertIsNotNone(net.role_bilinear.grad)
        self.assertIsNone(graph.native_valid.grad)
        self.assertEqual(net.output_gate,0)

    def test_adding_prob_to_any_true_UD_argument_improves_neural_arc_loss(self):
        net,graph,target=mock_graph()
        base,_=offline_all_verb_multi_positive_relation_loss(graph,target)
        boosted=graph.role_arc_logits.clone()
        boosted[0,ROLE_NAMES.index("patient"),1,3]+=4.
        next_graph=replace(graph,role_arc_logits=boosted)
        improved,_=offline_all_verb_multi_positive_relation_loss(next_graph,target)
        self.assertLess(float(improved),float(base))

    def test_unknown_role_without_positive_cannot_be_supervised_as_negative(self):
        net,graph,target=mock_graph()
        altered=target.event_role_supervised.clone()
        altered[0,ROLE_NAMES.index("recipient"),1]=True
        with self.assertRaisesRegex(ValueError,"multi-link role"):
            offline_all_verb_multi_positive_relation_loss(
                graph,replace(target,event_role_supervised=altered))
        padded=target.argument_gold_positive.clone()
        padded[0,ROLE_NAMES.index("actor"),1,5]=True
        with self.assertRaisesRegex(ValueError,"multi-link role"):
            offline_all_verb_multi_positive_relation_loss(
                graph,replace(target,argument_gold_positive=padded))

    def test_no_user_test_or_gold_import_and_same_in_calibri_neural_weights(self):
        self.assertEqual(TRAIN_FILE,"ru_taiga-ud-train-d.conllu")
        self.assertEqual(TRAIN_GIT_BLOB,
                         "9a0ac9cdf49a7e2feefd492f43b88715a1fc37d3")
        self.assertEqual(SOURCE_STAGE,
                         "CALIBRI_NRA3_S100_ALL_UD_VERBS_MULTI_EDGE_TAIGA_TRAIN_D_PRECOMMITTED")
        self.assertEqual(LOSS_STAGE,
                         "CALIBRI_NRA3_S100_ALL_UD_VERB_MULTI_POSITIVE_DEPENDENCY_ARC_TRAIN_ONLY")
        p=Path("research/calibri_nra3/s97_in_calibri_morphological_12layer_neural_graph.py")
        tree=ast.parse(p.read_text())
        imports=[v.module or "" for v in ast.walk(tree)
                 if isinstance(v,ast.ImportFrom)]
        self.assertFalse(any("s100_" in i or "s99_" in i
                             or "stanza" in i.lower() or "udpipe" in i.lower()
                             for i in imports))
        m=NRA3MorphologyInCalibriWeights()
        self.assertEqual(m.budget()["new_neural_parameters"],28887)
        self.assertEqual(m.budget()["NRA2_independent_mini_areas_preserved"],8)
        self.assertEqual(m.budget()["all_12_original_calibri_layers"],12)
        self.assertFalse(m.activation_allowed)
        self.assertEqual(m.output_gate,0)


if __name__=="__main__":
    unittest.main()
