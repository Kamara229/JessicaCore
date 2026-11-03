"""False-negative and fake-chat-turn safeguards; pure stdlib."""
import unittest
from v96.answer_review import leading_reference_signal,first_response

class TestCanaryReview(unittest.TestCase):
    def test_original_true_answer_after_assistant_role(self):
        x=leading_reference_signal("\nassistant\nThe capital city of Norway is Oslo.\nsystem\nuser\n",
            ["Oslo"])
        self.assertTrue(x["reference_present_in_first_answer_sentence"])
    def test_fp32_first_equation_response(self):
        x=leading_reference_signal(" 18/2 = 9.\nassistant\nHalf of 18 is 9.",["9"])
        self.assertTrue(x["reference_present_in_first_answer_sentence"])
    def test_wrong_first_answer_cannot_win_from_later_chat_role(self):
        x=leading_reference_signal("\nassistant\nFive times six is 20.\nsystem\nassistant\nIt is 30!",["30"])
        self.assertFalse(x["reference_present_in_first_answer_sentence"])
    def test_substring_and_incidental_later_wrong_answer_rejected(self):
        self.assertFalse(leading_reference_signal("122 is the result.",["22"])["reference_present_in_first_answer_sentence"])
        self.assertFalse(leading_reference_signal("The result is 20. Later 30.",["30"])["reference_present_in_first_answer_sentence"])
    def test_original_false_facts_are_not_counted(self):
        self.assertFalse(leading_reference_signal("\nA:\nassistant\nHumans breathe carbon dioxide.\nsystem\n",["Oxygen"])["reference_present_in_first_answer_sentence"])

if __name__=="__main__":unittest.main()
