"""S108 new GSD vs spent Taiga corpus, word-offset and non-negative UD contracts."""
import ast
from pathlib import Path
import unittest
from research.calibri_nra3.s108_new_Russian_GSD_TRAIN256_source_gold_wall import (
    STAGE,UPSTREAM_COMMIT,TRAIN_GIT_BLOB,LICENSE_FILE_BLOB,LICENSE,
    construct_gsd_new256,
)
from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import UDNode
from research.calibri_nra3.s103_new_Taiga_TRAINe_allVERB_source_gold_wall import (
    all_ud_verbs_roles_with_multiedges,
)

class TestS108GSDSourceVsGoldIsolation(unittest.TestCase):
    def test_pinned_different_original_train_only_GSD(self):
        self.assertEqual(UPSTREAM_COMMIT,"71b92ce4c8da2b779450b0297abada721796425f")
        self.assertEqual(TRAIN_GIT_BLOB,"de392eb8d1a92fe1d12bc2e49c21cb717ae89859")
        self.assertEqual(LICENSE_FILE_BLOB,"9af2e0b2f73cf29fe1f60cb6376b176d9c2ba846")
        self.assertEqual(LICENSE,"CC BY-SA 4.0")
        self.assertIn("S108",STAGE)
        self.assertNotIn("TRAIN_c",STAGE)
    def test_synthetic_multiverb_multiargument_gold_is_positive_only(self):
        nodes=(
            UDNode(1,"Мария","Мария","PROPN","Case=Nom|Number=Sing",2,"nsubj","_"),
            UDNode(2,"читала","читать","VERB","Voice=Act",0,"root","_"),
            UDNode(3,"книгу","книга","NOUN","Case=Acc",2,"obj","_"),
            UDNode(4,"и","и","CCONJ","_",5,"cc","_"),
            UDNode(5,"пересказывала","пересказывать","VERB","Voice=Act",2,"conj","_"),
            UDNode(6,"историю","история","NOUN","Case=Acc",5,"obj","_"),
        )
        events=all_ud_verbs_roles_with_multiedges(nodes)
        self.assertEqual(len(events),2)
        self.assertEqual(events[0]["predicate_UD_word_zero_index_TRAIN_ONLY"],1)
        self.assertTrue(any(e["role"]=="actor" for e in events[0]["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]))
        self.assertTrue(any(e["role"]=="patient" for e in events[1]["multi_argument_role_edges_UD_zero_word_indices_TRAIN_ONLY"]))
    def test_reject_unpinned_bytes_before_corpus_selection(self):
        with self.assertRaises(ValueError):
            construct_gsd_new256(b"# fake\n")
    def test_static_no_GOLD_in_original_runtime_and_no_old_DEV_used(self):
        s=Path("research/calibri_nra3/s108_new_Russian_GSD_TRAIN256_source_gold_wall.py").read_text()
        for marker in ("source_only_output_contains_no_UD_POS_grammar_roles_or_parser",
                       "offline_UD_gold_NEVER_passed_to_original_Calibri",
                       "gold_ONLY_after_real_original12_source_SHA_pinned",
                       "original_Taiga_TRAIN_c_reserved_UNOPENED",
                       "old_Taiga_S100_d_S103_e_256_source_text_overlap",
                       "attribution_sharealike_legal_review_before_commercial_or_production"):
            self.assertIn(marker,s)
        imports=[n.module or "" for n in ast.walk(ast.parse(s))
                 if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any("s105_TaigaE_S104_real12_only_8200_affinity_train" in p
                             or "s106_TaigaE_TRAINONLY" in p
                             or "s102_real12_TaigaTRAINd_multiverb_gold" in p
                             for p in imports))
if __name__=="__main__":
    unittest.main()
