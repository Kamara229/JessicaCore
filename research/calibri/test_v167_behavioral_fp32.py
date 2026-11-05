"""v167 negative controls: prevent false FP32/compression-equivalence claims."""
import unittest
from v167.protocol import ARMS,DOMAINS,check,counts
from v167.sources import math_cases,binding_cases,logic_cases,generation_probes
from v167.score import parse_strict,choices,wilson

class Behavioral(unittest.TestCase):
    def test_predeclared(self):
        self.assertTrue(check())
        self.assertEqual(sum(counts().values()),288)
        self.assertEqual(ARMS,("FP32","CB59","CB161"))
        self.assertEqual(len(DOMAINS),5)

    def test_synthetic_counts_and_answer_placement(self):
        for domain,fn in (("Arithmetic",math_cases),("Binding",binding_cases),("Logic",logic_cases)):
            rows=fn()
            self.assertEqual(len(rows),32)
            self.assertEqual(len(set(x.sha for x in rows)),32)
            self.assertTrue(all(x.domain==domain and 0<=x.gold<len(x.labels) for x in rows))
        self.assertEqual([x.gold for x in math_cases()[:4]],[0,1,2,3])

    def test_paired_binding_changes_truth_not_answers_options(self):
        pairs=binding_cases()
        for i in range(0,len(pairs),2):
            a,b=pairs[i:i+2]
            self.assertEqual(a.pair_id,b.pair_id)
            self.assertNotEqual(a.gold,b.gold)
            self.assertEqual(a.labels,b.labels)
            self.assertNotEqual(a.sha,b.sha)
            # The same candidate letters/candidate words are used in both variants.
            self.assertEqual(a.prompt.split("\n")[1:],b.prompt.split("\n")[1:])

    def test_genuine_free_response_domains(self):
        cases=generation_probes()
        self.assertEqual(len(cases),48)
        self.assertEqual(len(set(c["sha"] for c in cases)),48)
        self.assertTrue(any(c["gold"] is None for c in cases))
        self.assertEqual(sum(c["domain"]=="binding" for c in cases),16)

    def test_strict_lexical_grade_does_not_invent_semantics(self):
        self.assertTrue(parse_strict("42.", "42"))
        self.assertFalse(parse_strict("The answer is 42.", "42"))
        self.assertFalse(parse_strict("42 42", "42"))
        self.assertIsNone(parse_strict("Hello robot.",None))

    def test_fp32_weak_must_not_be_called_compression_win(self):
        fake={}
        # FP32 wrong on one item; packed right does NOT prove source was good.
        for arm in ARMS:
            fake[arm]={}
            for domain,n in counts().items():
                rows=[]
                for j in range(n):
                    good=(arm=="CB161" and j==0) or (arm=="FP32" and j==1) or (arm=="CB59" and j==1)
                    rows.append({"prompt_sha":f"{domain}-{j}","index":j,
                                 "chosen_index":0 if good else 1,
                                 "chosen_label":"A" if good else "B",
                                 "correct":good})
                fake[arm][domain]=rows
        from collections import namedtuple
        C=namedtuple("C","sha index")
        tasks={d:[C(f"{d}-{i}",i) for i in range(n)] for d,n in counts().items()}
        report=choices(tasks,fake)["by_domain"]["SciQ"]
        self.assertEqual(report["CB161"]["FP32_correct_lost"],1)
        self.assertEqual(report["CB161"]["FP32_wrong_but_packed_correct"],1)
        self.assertEqual(report["CB161"]["net_correct_delta_vs_FP32"],0)
        self.assertEqual(report["CB161"]["retention_conditional_on_FP32_correct"],0)
        self.assertEqual(len(wilson(4,16)),2)

if __name__=="__main__":
    unittest.main()
