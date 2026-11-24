import unittest
from research.calibri_nra3 import s119q2i8z56_gpt_fullanswer_prereg as m
class TestZ56GPT(unittest.TestCase):
    def test_source(self):
        a=m.verify()
        self.assertEqual(a["cases"],28)
        self.assertEqual(a,m.verify())
        self.assertTrue(a["FP32_report_only_never_gate_or_training"])
        self.assertEqual(a["output_gate"],0)
    def test_full_answer_not_substring(self):
        r={"accepted_answers":["Лида."]}
        self.assertTrue(m.score_text(r," ЛИДА ",eos=True,valid_utf8=True)["whole_answer_and_EOS"])
        self.assertFalse(m.score_text(r,"Не Лида.",eos=True,valid_utf8=True)["whole_answer_and_EOS"])
        self.assertFalse(m.score_text(r,"Лида. Нет.",eos=True,valid_utf8=True)["whole_answer_and_EOS"])
        self.assertFalse(m.score_text(r,"Лида",eos=False,valid_utf8=True)["whole_answer_and_EOS"])
        self.assertFalse(m.score_text(r,"Лида",eos=True,valid_utf8=False)["whole_answer_and_EOS"])
    def test_frozen_variants(self):
        r={"accepted_answers":["Артём."]}
        self.assertTrue(m.score_text(r,"Артем",eos=True,valid_utf8=True)["whole_answer_and_EOS"])
        self.assertEqual(m.MAX_NATIVE_STEPS,48)
if __name__=="__main__":unittest.main()
