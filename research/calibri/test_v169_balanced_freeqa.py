"""No cherry-picked examples or unreliable forced-choice-only scoring."""
import unittest
from collections import Counter
from v169.protocol import ARMS,DOMAINS,DOMAIN_COUNTS,FORMATS,check
from v169.sources import questions,all_cases
from v169.grade import detect,grade

class RealGeneratedQA(unittest.TestCase):
    def test_two_real_inference_prompts_per_each_base_question(self):
        self.assertTrue(check())
        qs=questions()
        rows=all_cases()
        self.assertEqual(len(qs),96)
        self.assertEqual(len(rows),192)
        self.assertEqual({q.id for q in qs},set(range(96)))
        self.assertEqual(len(set(r["prompt_sha"] for r in rows)),192)
        self.assertEqual(Counter(x["domain"] for x in rows),
                         {d:2*DOMAIN_COUNTS[d] for d in DOMAINS})
        self.assertEqual(Counter(x["style"] for x in rows),{"direct":96,"brief":96})

    def test_balanced_paired_question_values(self):
        qs=questions()
        for group in ("binding","yesno"):
            pairs={}
            for q in qs:
                if q.domain==group:
                    pairs.setdefault(q.pair_id,[]).append(q)
            for pair in pairs.values():
                self.assertEqual(len(pair),2)
                self.assertNotEqual(pair[0].gold,pair[1].gold)
        self.assertEqual(Counter(q.gold for q in qs if q.domain=="yesno"),
                         {"Yes":8,"No":8})

    def test_real_answer_extraction_with_ambiguous_prose(self):
        for kind,text,expected in (
            ("number","7 + 5 = 12.","12"),
            ("number","The answer is 27.","27"),
            ("yesno","Yes, that's correct.","Yes"),
            ("yesno","No.","No"),
            ("color","Mira's badge is red.","red"),
            ("number","A triangle has three sides.","3"),
        ):
            label,answer=detect({"kind":kind,"verbatim_text":text,"gold":expected})
            self.assertEqual(answer,expected)
        self.assertEqual(detect({"kind":"color","verbatim_text":"A yellow badge is green.","gold":"yellow"})[0],"ambiguous")
        self.assertEqual(detect({"kind":"yesno","verbatim_text":"Maybe yes and no.","gold":"No"})[0],"ambiguous")

    def test_word_presence_does_not_fake_semantic_correctness(self):
        c={"kind":"word","gold":"Paris"}
        score=grade(c,{"verbatim_text":"Paris is a word but I have not answered."})
        self.assertIsNone(score["verified_format_correct"])
        self.assertTrue(score["gold_lexical_presence_only"])

    def test_no_teacher_math_metric_dependency(self):
        from v169 import grade as scorer
        import inspect
        src=inspect.getsource(scorer)
        self.assertNotIn("KL_divergence",src)
        self.assertNotIn("calculate_NLL",src)

if __name__=="__main__":
    unittest.main()
