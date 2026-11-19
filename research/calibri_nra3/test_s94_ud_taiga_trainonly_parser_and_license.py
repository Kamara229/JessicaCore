"""S94: synthetic tiny CoNLL-U parser/annotation hard-wall tests.

NO THIRD-PARTY corpus included in tests or repository; CI separately
downloads pinned 35MB train-a under CC BY-SA 4.0 for research only.
"""
import ast
import unittest
from pathlib import Path
from unittest.mock import patch

from research.calibri_nra3 import s94_ud_taiga_trainonly_grammatical_source_gold_WALL as u

def sent(sid,text,words):
    return "\n".join(
        [f"# sent_id = {sid}",f"# text = {text}"]+
        ["\t".join((str(i),form,lemma,upos,"_",feat,str(head),dep,"_","_"))
         for i,form,lemma,upos,feat,head,dep in words]
    )+"\n\n"


FIXTURE=(
    sent("S94_SYNTHETIC_ACTIVE","Курьер передал письмо сегодня.",(
        (1,"Курьер","курьер","NOUN","Case=Nom",2,"nsubj"),
        (2,"передал","передать","VERB","VerbForm=Fin",0,"root"),
        (3,"письмо","письмо","NOUN","Case=Acc",2,"obj"),
        (4,"сегодня","сегодня","ADV","_",2,"advmod"),
        (5,".",".","PUNCT","_",2,"punct"),
    ))+
    sent("S94_SYNTHETIC_PASSIVE","Письмо было передано курьером.",(
        (1,"Письмо","письмо","NOUN","Case=Nom",3,"nsubj:pass"),
        (2,"было","быть","AUX","_",3,"aux:pass"),
        (3,"передано","передать","VERB","Voice=Pass|VerbForm=Part",0,"root"),
        (4,"курьером","курьер","NOUN","Case=Ins",3,"obl:agent"),
        (5,".",".","PUNCT","_",3,"punct"),
    ))+
    sent("S94_SYNTHETIC_NEG","Курьер не отправил письмо сегодня.",(
        (1,"Курьер","курьер","NOUN","Case=Nom",3,"nsubj"),
        (2,"не","не","PART","_",3,"advmod"),
        (3,"отправил","отправить","VERB","VerbForm=Fin",0,"root"),
        (4,"письмо","письмо","NOUN","Case=Acc",3,"obj"),
        (5,"сегодня","сегодня","ADV","_",3,"advmod"),
        (6,".",".","PUNCT","_",3,"punct"),
    ))+
    sent("S94_SYNTHETIC_OTHER","Позже поезд остановился у платформы.",(
        (1,"Позже","позже","ADV","_",3,"advmod"),
        (2,"поезд","поезд","NOUN","Case=Nom",3,"nsubj"),
        (3,"остановился","остановиться","VERB","VerbForm=Fin",0,"root"),
        (4,"у","у","ADP","_",5,"case"),
        (5,"платформы","платформа","NOUN","Case=Gen",3,"obl"),
        (6,".",".","PUNCT","_",3,"punct"),
    ))
)


class TestS94TaigaTrainOnly(unittest.TestCase):
    def test_four_new_russian_grammar_types_from_UD_without_hardcoded_suffix(self):
        rows=list(u.parse_conllu(FIXTURE))
        self.assertEqual(len(rows),4)
        self.assertEqual(
            [u.class_of(*u.supervised_verb_candidate(nodes))
             for _,_,nodes in rows],
            ["active_transitive","passive","negated","other_intransitive"])
        for _,t,nodes in rows:
            self.assertEqual(len(u.align_ud_surfaces_exact(t,nodes)),len(nodes))

    def test_active_vs_passive_actor_patient_word_indices(self):
        rows=list(u.parse_conllu(FIXTURE))
        a,roles,passive=u.supervised_verb_candidate(rows[0][2])
        self.assertFalse(passive)
        self.assertEqual(a.index-1,1)
        self.assertEqual(roles["actor"],0)
        self.assertEqual(roles["patient"],2)
        p,roles2,passive2=u.supervised_verb_candidate(rows[1][2])
        self.assertTrue(passive2)
        self.assertEqual(p.index-1,2)
        self.assertEqual(roles2["actor"],3)
        self.assertEqual(roles2["patient"],0)

    def test_Ud_negation_scope_attached_to_predicate_not_whole_sentence(self):
        rows=list(u.parse_conllu(FIXTURE))
        event,roles,passive=u.supervised_verb_candidate(rows[2][2])
        self.assertEqual(event.index-1,2)
        self.assertEqual(roles["negation_scope"],1)
        self.assertFalse(passive)

    def test_source_only_and_offline_labels_are_separate(self):
        with patch.object(u,"LIMIT_PER_STRATUM",1):
            source,gold,manifest=u.select_locked_UD_train256(
                FIXTURE.encode(),require_pinned_blob=False)
        self.assertEqual(len(source),len(gold))
        self.assertEqual(len(source),4)
        self.assertTrue(all(set(z)=={"row_id","text"} for z in source))
        self.assertEqual([x["row_id"] for x in source],
                         [z["row_id"] for z in gold])
        self.assertTrue(all("ud_" not in " ".join(x.keys()) for x in source))
        self.assertTrue(all(z["not_yet_original_native_BPE_coordinates"]
                            for z in gold))
        self.assertTrue(manifest["source_rows_have_no_label_or_UD_POS_or_role"])
        self.assertFalse(manifest["candidate_accepted"])

    def test_pinned_upstream_trust_boundaries_license_and_no_prod(self):
        self.assertEqual(u.UPSTREAM_COMMIT,
            "445ea57e47a3adae3146d0a42d2a8876214f2d20")
        self.assertEqual(u.TRAIN_BLOB_SHA,
            "c63aa8de90f2260e9681fde49d56d2c933366ef0")
        self.assertEqual(u.LICENSE,"CC BY-SA 4.0")
        with self.assertRaisesRegex(ValueError,"pinned Git blob SHA"):
            u.select_locked_UD_train256(FIXTURE.encode(),require_pinned_blob=True)

    def test_refuse_semantically_unaligned_surface_as_UD_gold(self):
        rows=list(u.parse_conllu(FIXTURE))
        _,text,nodes=rows[0]
        self.assertIsNone(u.align_ud_surfaces_exact(
            text.replace("Курьер","Другой"),nodes))
        self.assertIsNone(u.align_ud_surfaces_exact(
            text.replace("письмо","конверт"),nodes))

    def test_no_upstream_dev_test_spent_syntactic_tests_or_parser_in_model(self):
        p=Path("research/calibri_nra3/s94_ud_taiga_trainonly_grammatical_source_gold_WALL.py")
        tree=ast.parse(p.read_text(encoding="utf8"))
        modules=[n.module or "" for n in ast.walk(tree)
                 if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any("s79" in z.lower() or "s86" in z.lower()
                             or "s83" in z.lower() or
                             "stanza" in z.lower() or "udpipe" in z.lower()
                             for z in modules))
        self.assertEqual(u.TRAIN_FILE,"ru_taiga-ud-train-a.conllu")
        self.assertNotIn("test.conllu",p.read_text(encoding="utf8"))
        self.assertNotIn("dev.conllu",p.read_text(encoding="utf8"))

    def test_no_model_mutation_and_exact_source_gold_SHA_format(self):
        with patch.object(u,"LIMIT_PER_STRATUM",1):
            source,gold,manifest=u.select_locked_UD_train256(
                FIXTURE.encode(),require_pinned_blob=False)
        self.assertEqual(manifest["source_only_sha256"],u.sha256_serialized(source))
        self.assertEqual(manifest["separately_supervised_only_gold_sha256"],
                         u.sha256_serialized(gold))
        self.assertTrue(manifest["no_production_or_commercial_use_until_license_review"])
        self.assertTrue(manifest["share_alike_requirements_review_before_downstream_use"])
        self.assertTrue(manifest["only_upstream_TRAIN_a_used_NO_upstream_dev_or_test"])
        self.assertTrue(manifest["UD_word_indices_REQUIRE_fail_closed_native_Calibri_BPE_alignment_later"])
        self.assertTrue(manifest["original_43M_GT_616_J7_and_weights_not_touched_by_S94"])
        self.assertEqual(manifest["output_gate"],0)


if __name__=="__main__":
    unittest.main()
