"""S94 synthetic, NOT real-Russian accuracy, exact native-BPE contract."""
import ast
import unittest
from pathlib import Path

import torch

from research.nra3_s19.graph_schema import raw_model_input
from research.nra3_s20.s51_fractional_native_bpe_overlap import (
    align_offsets_from_raw_text,
)
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    NRA3InCalibriNeuralWeights,ROLE_NAMES,STATE_NAMES,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    graph_train_only_loss,IGNORE,UNKNOWN,
)
from research.calibri_nra3.s94_OFFLINE_UD_conllu_to_real_native_BPE_train_only import (
    STAGE,read_one_strict_conllu,
    align_one_sentence_source_to_original_native_BPE,
    offline_native_BPE_training_targets,
)


def co(text,tokens):
    return "# sent_id = constructed_ONLY_TRAIN\n# text = "+text+"\n"+\
        "\n".join("\t".join(map(str,(i,word,"_",pos,"_",feat,head,dep,"_","_")))
                  for i,(word,pos,feat,head,dep) in enumerate(tokens,1))


TRAIN_CASES=(
    co("Инженер передал проект заказчику",(
        ("Инженер","NOUN","_",2,"nsubj"),
        ("передал","VERB","VerbForm=Fin",0,"root"),
        ("проект","NOUN","_",2,"obj"),
        ("заказчику","NOUN","_",2,"iobj"))),
    co("Проект утверждён инженером",(
        ("Проект","NOUN","_",2,"nsubj:pass"),
        ("утверждён","VERB","VerbForm=Part|Voice=Pass",0,"root"),
        ("инженером","NOUN","_",2,"obl:agent"))),
    co("Инженер не подписал документ",(
        ("Инженер","NOUN","_",3,"nsubj"),
        ("не","PART","Polarity=Neg",3,"advmod"),
        ("подписал","VERB","VerbForm=Fin",0,"root"),
        ("документ","NOUN","_",3,"obj"))),
)


def make_align(sentence):
    text=sentence.text
    raw=raw_model_input(text).tokens
    ids=tuple(range(200,200+len(raw)))
    offsets=tuple((w.start,w.end) for w in raw)
    align=align_offsets_from_raw_text(
        text,ids,offsets,0,len(text))
    return align


def prepare():
    docs=tuple(read_one_strict_conllu(z) for z in TRAIN_CASES)
    align=tuple(make_align(z) for z in docs)
    width=max(z.original_token_count for z in align)
    valid=torch.zeros((len(docs),width),dtype=torch.bool)
    for i,m in enumerate(align):
        valid[i,:m.original_token_count]=True
    out=offline_native_BPE_training_targets(docs,align,valid)
    return docs,align,out,valid


class TestS94OfflineUDNotRuntime(unittest.TestCase):
    def test_three_distinct_manual_constructed_grammatical_training_rows(self):
        docs,align,truth,valid=prepare()
        self.assertEqual(len(docs),3)
        self.assertEqual([z.original_token_count for z in align],[4,3,4])
        self.assertTrue(torch.equal(truth.original_native_valid,valid))
        self.assertEqual(truth.evidence,
            "offline_UD_teacher_only_not_independently_verified")
        self.assertEqual(STAGE,
            "CALIBRI_NRA3_S94_OFFLINE_UD_CONLLU_NATIVE_BPE_EVENT_ROLE_TRAIN_LABELS")

    def test_actor_patient_recipient_and_passive_agent_neural_roles(self):
        docs,align,t,v=prepare()
        r=t.role_argument_token
        actor=ROLE_NAMES.index("actor")
        patient=ROLE_NAMES.index("patient")
        recipient=ROLE_NAMES.index("recipient")
        self.assertEqual(int(r[0,actor,1]),0)      # Engineer -> passed
        self.assertEqual(int(r[0,patient,1]),2)    # project
        self.assertEqual(int(r[0,recipient,1]),3)  # client
        self.assertEqual(int(r[1,patient,1]),0)    # passive subject becomes patient
        self.assertEqual(int(r[1,actor,1]),2)      # instrumental agent
        self.assertEqual(float(t.state[1,1,STATE_NAMES.index("passive")]),1.)
        self.assertEqual(float(t.mini_area[1,1,2]),1.)
        self.assertEqual(int(t.mention[1,1]),1)
        self.assertEqual(int(t.mention[1,0]),IGNORE)

    def test_negation_scope_token_and_NRA2_neural_mini_area(self):
        docs,align,t,v=prepare()
        self.assertEqual(
            int(t.role_argument_token[2,ROLE_NAMES.index("negation_scope"),2]),1)
        self.assertEqual(float(t.state[2,2,STATE_NAMES.index("negated")]),1.)
        self.assertEqual(float(t.mini_area[2,2,1]),1.)
        self.assertEqual(float(t.state[0,1,STATE_NAMES.index("negated")]),float(UNKNOWN))
        self.assertEqual(float(t.state[1,1,STATE_NAMES.index("assertion")]),float(UNKNOWN))
        self.assertEqual(float(t.mini_area[0,1,7]),float(UNKNOWN))

    def test_strict_native_BPE_UD_word_mismatch_rejected(self):
        docs,align,t,v=prepare()
        bad=TRAIN_CASES[0].replace("\tпередал\t","\tотослал\t")
        with self.assertRaisesRegex(ValueError,"reconstruct"):
            read_one_strict_conllu(bad)
        doc=docs[0]
        import dataclasses
        wrong=dataclasses.replace(align[0],
            original_plaintext_sha256="0"*64)
        with self.assertRaisesRegex(ValueError,"source text SHA"):
            align_one_sentence_source_to_original_native_BPE(doc,wrong)
        misaligned=dataclasses.replace(align[0],original_token_count=5)
        with self.assertRaisesRegex(ValueError,"original native BPE valid"):
            offline_native_BPE_training_targets(
                docs,(misaligned,align[1],align[2]),v)

    def test_new_native_targets_can_train_actual_in_Calibri_eight_graph_heads(self):
        docs,align,labels,valid=prepare()
        torch.manual_seed(941119)
        model=NRA3InCalibriNeuralWeights()
        hidden=torch.randn(len(docs),12,valid.shape[1],320)
        graph=model(hidden,valid)
        loss,parts=graph_train_only_loss(graph,labels)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(parts["annotated_event_positions"],3)
        self.assertGreaterEqual(parts["annotated_role_edges"],8)
        self.assertEqual(parts["annotated_state_bits"],2)
        self.assertEqual(parts["annotated_mini_area_bits"],2)
        loss.backward()
        self.assertIsNotNone(model.layer_router_logits.grad)
        self.assertIsNotNone(model.role_bilinear.grad)
        self.assertTrue(torch.isfinite(model.role_bilinear.grad).all())
        self.assertFalse(model.is_trained)
        self.assertEqual(model.output_gate,0)

    def test_supervision_is_not_an_external_inference_plugin(self):
        p=Path("research/calibri_nra3/s91_neural_multilayer_event_graph.py")
        q=Path("research/calibri_nra3/s94_OFFLINE_UD_conllu_to_real_native_BPE_train_only.py")
        t=ast.parse(p.read_text(encoding="utf8"))
        imports=[x.module or "" for x in ast.walk(t)
                 if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s94_OFFLINE" in z or "stanza" in z.lower()
                             or "udpipe" in z.lower() for z in imports))
        u=ast.parse(q.read_text(encoding="utf8"))
        imports2=[x.module or "" for x in ast.walk(u)
                  if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any("s79_" in z or "s86_" in z or "S88_" in z
                             for z in imports2))


if __name__=="__main__":
    unittest.main()
