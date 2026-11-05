"""v190 new original model QA: rigorously paired, ungraded!=wrong."""
import unittest
from collections import Counter
from v190 import protocol as p
from v190.cases import cases
from v190.grade import grade,summarize

class NewOriginalBehaviorQA(unittest.TestCase):
    def test_new_questions_all_distinct_balanced_and_pairs_opposite(self):
        examples=cases()
        self.assertEqual(len(examples),p.TOTAL)
        self.assertEqual(Counter(x["domain"] for x in examples),
                         {name:16 for name in p.DOMAINS})
        self.assertEqual(len({x["question_sha256"] for x in examples}),64)
        for domain in ("opposite_yesno","fictional_binding"):
            pairs={}
            for row in examples:
                if row["domain"]==domain:
                    pairs.setdefault(row["pair_id"],[]).append(row)
            self.assertEqual(len(pairs),8)
            for rows in pairs.values():
                self.assertEqual(len(rows),2)
                self.assertNotEqual(rows[0]["gold"],rows[1]["gold"])

    def test_strict_does_not_claim_negated_or_incidental_words(self):
        e=next(x for x in cases() if x["domain"]=="opposite_yesno" and x["gold"]=="Yes")
        self.assertIs(grade(e,{"verbatim_text":"Yes."})["machine_correct"],True)
        self.assertIs(grade(e,{"verbatim_text":"The answer is No."})["machine_correct"],False)
        for ambiguous in ("Yes, but not necessarily","No or Yes",
                          "Not Yes","The answer may be Yes or No"):
            self.assertIsNone(grade(e,{"verbatim_text":ambiguous})["machine_correct"])
        f=next(x for x in cases() if x["domain"]=="factual_short" and x["gold"]=="Mercury")
        self.assertIs(grade(f,{"verbatim_text":"Mercury"})["machine_correct"],True)
        self.assertIs(grade(f,{"verbatim_text":"The answer is Mercury."})["machine_correct"],True)
        self.assertIsNone(grade(f,{"verbatim_text":"Mercury is not correct."})["machine_correct"])

    def test_verified_pair_requires_both_real_answers(self):
        source=cases()
        rows=[]
        for c in source:
            rows.append(grade(c,{"verbatim_text":c["gold"]}))
        report=summarize(source,rows)
        self.assertEqual(report["confirmed"],64)
        self.assertEqual(report["ungraded"],0)
        for d in ("opposite_yesno","fictional_binding"):
            self.assertEqual(report["paired"][d]["both_machine_correct"],8)
        rows[0]=grade(source[0],{"verbatim_text":"I cannot decide"})
        report=summarize(source,rows)
        self.assertEqual(report["confirmed"],63)
        self.assertEqual(report["ungraded"],1)
        self.assertEqual(report["parsed_wrong"],0)

if __name__=="__main__":
    unittest.main()
