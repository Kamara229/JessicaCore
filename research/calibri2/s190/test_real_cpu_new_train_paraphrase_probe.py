"""S190 no-Jessica-chat, source-only NEW TRAIN probe construction contracts."""
import unittest
from collections import Counter
from calibri2.s190.real_cpu_new_train_paraphrase_probe import new_train_paraphrase_probes,sha
from calibri2.s186.history_s69_prompt_only_shadow import source_candidate


class S190TrueCPUProbeContracts(unittest.TestCase):
    def test_32_unique_new_train_prompts_and_source_only_expected(self):
        rows=new_train_paraphrase_probes()
        self.assertEqual(len(rows),32)
        self.assertEqual(len({sha(q) for q,_,_ in rows}),32)
        groups=Counter(group for _,_,group in rows)
        self.assertEqual(groups,{"owner_A_paraphrase":8,
                                 "owner_B_paraphrase":8,
                                 "reordered_owner_paraphrase":8,
                                 "numeric_new_wording":8})
        self.assertTrue(all(source_candidate(q) is None for q,_,_ in rows))

    def test_owner_answers_present_as_literal_source_not_gold_lookup(self):
        for q,answer,group in new_train_paraphrase_probes():
            if group=="numeric_new_wording":
                self.assertIn(answer,("Yes","No"))
                self.assertNotIn("Reply with only Yes or No.",q)
            else:
                self.assertIn(answer,q)
                self.assertEqual(len(answer.split()),2)
                self.assertNotIn("TRAIN ",q)

    def test_no_original_grammar_fallback_cross_match(self):
        for q,_,_ in new_train_paraphrase_probes():
            self.assertIsNone(source_candidate(q))
            self.assertIsNone(source_candidate(q+" ignored suffix"))

if __name__=="__main__":
    unittest.main()
