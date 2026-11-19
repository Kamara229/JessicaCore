"""S108 synthetic/source-contract only; real pinned official GSD TRAIN CI required."""
import ast
from pathlib import Path
import unittest
from research.calibri_nra3.s108_new_GSD_TRAIN256_multiverb_source_gold_wall import (
    STAGE,TRAIN_FILE,TRAIN_GIT_BLOB,UPSTREAM_REPO,UPSTREAM_COMMIT,
    SOURCE_FILE,GOLD_FILE,MANIFEST_FILE,all_ud_verbs_roles_with_multiedges,
    offline_allword_morphology,
)
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    UDNode,STRATA,
)

def node(i,form,lemma,upos,feats,head,rel):
    return UDNode(i,form,lemma,upos,feats,head,rel,"_")

class TestS108NewGSDSeparateSourceGold(unittest.TestCase):
    def test_immutable_REAL_new_corpus_git_blob_no_spent_Taiga_teacher(self):
        self.assertEqual(UPSTREAM_REPO,'UniversalDependencies/UD_Russian-GSD')
        self.assertEqual(UPSTREAM_COMMIT,'71b92ce4c8da2b779450b0297abada721796425f')
        self.assertEqual(TRAIN_GIT_BLOB,'de392eb8d1a92fe1d12bc2e49c21cb717ae89859')
        self.assertEqual(TRAIN_FILE,'ru_gsd-ud-train.conllu')
        self.assertIn('S108',STAGE)
        self.assertTrue(SOURCE_FILE.endswith('SOURCE_ONLY_NO_GOLD.json'))
        self.assertIn('ALL_VERBS',GOLD_FILE)
        self.assertIn('PROVENANCE',MANIFEST_FILE)
        self.assertEqual(len(STRATA),4)

    def test_passive_actor_patient_and_two_verbs_all_preserved(self):
        x=[
            node(1,'Иван','Иван','PROPN','Case=Nom|Number=Sing',2,'nsubj'),
            node(2,'собрал','собрать','VERB','Voice=Act',0,'root'),
            node(3,'машину','машина','NOUN','Case=Acc|Number=Sing',2,'obj'),
            node(4,'а','а','CCONJ','_',6,'cc'),
            node(5,'Петр','Петр','PROPN','Case=Nom',6,'nsubj'),
            node(6,'доставил','доставить','VERB','Voice=Act',2,'conj'),
            node(7,'документы','документ','NOUN','Case=Acc|Number=Plur',6,'obj'),
        ]
        events=all_ud_verbs_roles_with_multiedges(tuple(x))
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]['predicate_UD_word_zero_index_TRAIN_ONLY'],1)
        self.assertEqual(events[1]['predicate_UD_word_zero_index_TRAIN_ONLY'],5)
        self.assertEqual({e['role'] for e in events[0]['multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY']},
                         {'actor','patient'})
        tags=offline_allword_morphology(tuple(x))
        self.assertEqual(len(tags['offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY']),7)
        self.assertIn('Sing',tags['offline_Number_ALL_UD_WORDS_TRAIN_ONLY'])

    def test_gold_not_in_source_and_no_gold_in_neural_s107(self):
        t=ast.parse(Path('research/calibri_nra3/s108_new_GSD_TRAIN256_multiverb_source_gold_wall.py').read_text())
        mods=[x.module or '' for x in ast.walk(t) if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any('s105_TaigaE_S104_real12_only_8200' in x for x in mods))
        script=Path('research/calibri_nra3/s108_new_GSD_TRAIN256_multiverb_source_gold_wall.py').read_text()
        self.assertIn('len(source)==len(gold)==256',script)
        self.assertIn('all(set(x)=={"row_id","text"} for x in source)',script)
        self.assertIn('no_negative_event_labels_invented',script)
        self.assertIn('Taiga_TRAIN_c_REMAINS_RESERVED_unopened_for_future_firstread',script)
        self.assertIn('new_inmodel_NRA3_training_steps',script)
        neural=Path('research/calibri_nra3/s107_in_calibri_structural_role_competition_53743.py').read_text()
        self.assertNotIn('s108_new_GSD',neural)

if __name__=='__main__':
    unittest.main()
