"""Scientific safeguards: v193 prompts disjoint, frozen grader, paired golds."""
import unittest
from collections import Counter,defaultdict

from v193.questions import cases
from v193.protocol import COUNT,DOMAIN_COUNTS,GRADER_SHA256
from v193.precommit import grading_fingerprint,checksum_qa
from v169.sources import all_cases
from v190.cases import cases as older_public_64
from v192.grade import summarize


class NewOriginalSourceQA(unittest.TestCase):
    def test_96_new_question_domain_balance_and_disjointness(self):
        mine=cases()
        self.assertEqual(len(mine),COUNT)
        self.assertEqual(dict(Counter(x["domain"] for x in mine)),DOMAIN_COUNTS)
        previous={x["question"] for x in all_cases()}
        previous.update(x["question"] for x in older_public_64())
        self.assertFalse(set(x["question"] for x in mine)&previous)
        self.assertEqual(len({x["question_sha256"] for x in mine}),COUNT)
        self.assertEqual(len(checksum_qa(mine)),64)

    def test_exact_earlier_v192_grader_source_SHA_before_inference(self):
        same=grading_fingerprint()
        self.assertEqual(same["sha256"],GRADER_SHA256)
        self.assertEqual(same["version"],"v192-locked-grader-v1")

    def test_twenty_four_reversed_questions_need_both_right(self):
        mine=cases()
        for d in ("opposite_yesno","fictional_binding"):
            groups=defaultdict(list)
            for x in mine:
                if x["domain"]==d:groups[x["pair_id"]].append(x)
            self.assertEqual(len(groups),12)
            for rows in groups.values():
                self.assertEqual(len(rows),2)
                self.assertNotEqual(rows[0]["gold"],rows[1]["gold"])
        same_yes={x["case_id"]:("Yes" if x["domain"]=="opposite_yesno"
                    else x["gold"]) for x in mine}
        y=summarize(mine,same_yes)
        self.assertEqual(y["opposite_pairs"]["opposite_yesno"]["both_correct"],0)
        oracle=summarize(mine,{x["case_id"]:x["gold"] for x in mine})
        self.assertEqual(oracle["machine_confirmed"],COUNT)
        self.assertEqual(oracle["opposite_pairs"]["opposite_yesno"]["both_correct"],12)
        self.assertEqual(oracle["opposite_pairs"]["fictional_binding"]["both_correct"],12)

if __name__=="__main__":
    unittest.main()
