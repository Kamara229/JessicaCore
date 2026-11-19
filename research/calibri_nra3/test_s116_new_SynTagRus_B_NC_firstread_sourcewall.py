"""S116 prospective source/GOLD preregistration on disjoint SynTagRus train-B NC."""
import ast
from pathlib import Path
import unittest
from research.calibri_nra3.s116_SynTagRusB256_NC_ONESHOT_source_GOLD_precommit import (
    STAGE,TRAIN_FILE,TRAIN_GIT_BLOB,UPSTREAM_COMMIT,LICENSE,LICENSE_GIT_BLOB,
    SOURCE_FILE,GOLD_FILE,MANIFEST_FILE,
    all_ud_verbs_roles_with_multiedges,
)
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import UDNode,STRATA

class TestS116SourceGOLDBurnRule(unittest.TestCase):
    def test_exact_NC_official_new_separate_train_B_source(self):
        self.assertEqual(TRAIN_FILE,'ru_syntagrus-ud-train-b.conllu')
        self.assertEqual(TRAIN_GIT_BLOB,'416fcb93e223869e2718be30629dad9fadd4c290')
        self.assertEqual(UPSTREAM_COMMIT,'8b32a94d090be3f023c0c7bbd30804c84efe5f96')
        self.assertEqual(LICENSE_GIT_BLOB,'2c19e1b9c8249671d92d464b9678391b0c3cca19')
        self.assertEqual(LICENSE,'CC BY-NC-SA 4.0')
        self.assertIn('S116',STAGE)
        self.assertIn('TRAINB256',SOURCE_FILE)
        self.assertIn('TRAINB256',GOLD_FILE)
        self.assertIn('TRAINB256',MANIFEST_FILE)
        self.assertEqual(len(STRATA),4)
    def test_future_firstread_no_model_eval_or_used_dev_in_precommit_code(self):
        s=Path('research/calibri_nra3/s116_SynTagRusB256_NC_ONESHOT_source_GOLD_precommit.py').read_text()
        for name in ('PROSPECTIVE_ONESHOT_intermodeled_checkpoint_must_be_PINNED_BEFORE_first_score',
                     'NO_training_tuning_model_selection_on_train_b',
                     'len(source)==len(gold)==256','all(set(x)=={"row_id","text"} for x in source)',
                     'license_noncommercial_use_only_explicitly_required',
                     'Taiga_TRAIN_c_FIRSTREAD_ALREADY_BURNED_never_reuse'):
            self.assertIn(name,s)
        mods=[x.module or '' for x in ast.walk(ast.parse(s)) if isinstance(x,ast.ImportFrom)]
        self.assertFalse(any('s115_original12' in x or 's110_true12' in x for x in mods))
    def test_known_predicate_role_extraction_available_only_OFFLINE(self):
        nodes=[UDNode(1,'Иван','Иван','PROPN','Case=Nom',2,'nsubj','_'),
               UDNode(2,'передал','передать','VERB','Voice=Act',0,'root','_'),
               UDNode(3,'письмо','письмо','NOUN','Case=Acc',2,'obj','_'),
               UDNode(4,'сестре','сестра','NOUN','Case=Dat',2,'iobj','_'),
               UDNode(5,'вчера','вчера','ADV','_',2,'advmod','_')]
        xs=all_ud_verbs_roles_with_multiedges(tuple(nodes))
        self.assertEqual(len(xs),1)
        self.assertEqual({x['role'] for x in xs[0]['multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY']},
                         {'actor','patient','recipient'})
if __name__=="__main__":
    unittest.main()
