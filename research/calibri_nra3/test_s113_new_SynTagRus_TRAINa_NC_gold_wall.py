"""S113 SynTagRus NC source/GOLD only contracts, no calibration from spent heldout."""
import ast
from pathlib import Path
import unittest
from research.calibri_nra3.s113_new_SynTagRus_TRAINa256_source_GOLD_NC_wall import (
    STAGE,TRAIN_FILE,TRAIN_GIT_BLOB,UPSTREAM_REPO,UPSTREAM_COMMIT,
    LICENSE,LICENSE_GIT_BLOB,SOURCE_FILE,GOLD_FILE,MANIFEST_FILE,
    all_ud_verbs_roles_with_multiedges,offline_allword_morphology,
)
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    UDNode,STRATA,
)

def node(i,form,lemma,upos,feats,head,rel):
    return UDNode(i,form,lemma,upos,feats,head,rel,"_")

class TestS113NewSynTagRusSeparateGoldNC(unittest.TestCase):
    def test_pinned_new_immutable_corpus_NC_license(self):
        self.assertEqual(UPSTREAM_REPO,'UniversalDependencies/UD_Russian-SynTagRus')
        self.assertEqual(UPSTREAM_COMMIT,'8b32a94d090be3f023c0c7bbd30804c84efe5f96')
        self.assertEqual(TRAIN_GIT_BLOB,'eb7638bf3680414e49123412962e511b998a3b01')
        self.assertEqual(LICENSE_GIT_BLOB,'2c19e1b9c8249671d92d464b9678391b0c3cca19')
        self.assertEqual(TRAIN_FILE,'ru_syntagrus-ud-train-a.conllu')
        self.assertEqual(LICENSE,'CC BY-NC-SA 4.0')
        self.assertIn('S113',STAGE)
        self.assertTrue(SOURCE_FILE.endswith('SOURCE_ONLY_NO_GOLD.json'))
        self.assertIn('ALL_VERBS',GOLD_FILE)
        self.assertIn('PROVENANCE',MANIFEST_FILE)
        self.assertEqual(len(STRATA),4)

    def test_passive_and_actor_patient_multiple_verbs_offline_all_only(self):
        n=[node(1,'Текст','текст','NOUN','Case=Nom',2,'nsubj:pass'),
           node(2,'написан','написать','VERB','Voice=Pass',0,'root'),
           node(3,'автором','автор','NOUN','Case=Ins',2,'obl:agent'),
           node(4,'и','и','CCONJ','_',6,'cc'),
           node(5,'книги','книга','NOUN','Case=Acc',6,'obj'),
           node(6,'обсужден','обсудить','VERB','Voice=Pass',2,'conj')]
        arcs=all_ud_verbs_roles_with_multiedges(tuple(n))
        self.assertEqual(len(arcs),2)
        self.assertEqual({x['role'] for x in arcs[0]['multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY']},{'actor','patient'})
        tags=offline_allword_morphology(tuple(n))
        self.assertEqual(len(tags['offline_UPOS_ALL_UD_WORDS_TRAIN_ONLY']),len(n))

    def test_only_raw_text_source_no_gold_in_model_and_burned_firstread(self):
        path=Path('research/calibri_nra3/s113_new_SynTagRus_TRAINa256_source_GOLD_NC_wall.py')
        s=path.read_text()
        self.assertIn('len(source)==len(gold)==256',s)
        self.assertIn('all(set(x)=={"row_id","text"} for x in source)',s)
        self.assertIn('no_negative_event_labels_invented',s)
        self.assertIn('Taiga_TRAIN_c_FIRSTREAD_ALREADY_BURNED_never_reuse',s)
        self.assertIn('GSD_S110_INTERNAL_DEV_SPENT_never_used',s)
        self.assertIn('license_noncommercial_use_only_explicitly_required',s)
        self.assertIn('new_inmodel_NRA3_training_steps',s)
        mods=[n.module or '' for n in ast.walk(ast.parse(s)) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('s110_true12_GSD_16656' in x or 's106_one' in x for x in mods))
        neural=Path('research/calibri_nra3/s112_in_calibri_bounded_structural_trust_53751.py').read_text()
        self.assertNotIn('SynTagRus',neural)
if __name__=="__main__":
    unittest.main()
