import unittest
from research.calibri_nra3 import s119q2i8z66_gpt_paired_eval_prereg as m
class TestZ66(unittest.TestCase):
    def test_manifest(self):
        p=m.verify()
        self.assertEqual((p["cases"],p["pairs"]),(42,21))
        self.assertEqual(p,m.verify())
        self.assertTrue(p["FP32_comparison_only_not_gate"])
    def test_whole_answer(self):
        row={"accepted_answers":["да"]}
        self.assertTrue(m.score(row,"Да.",eos=True,valid_utf8=True)["complete_correct"])
        self.assertFalse(m.score(row,"Да, но нет.",eos=True,valid_utf8=True)["complete_correct"])
        self.assertFalse(m.score(row,"да",eos=False,valid_utf8=True)["complete_correct"])
        self.assertFalse(m.score(row,"да",eos=True,valid_utf8=False)["text_correct"])
    def test_pair_accuracy_not_individual(self):
        data=[]
        for i,r in enumerate(m.rows()):
            ok=i%2==0
            data.append({"id":r["id"],"generated_text":r["accepted_answers"][0] if ok else "ошибка",
                         "complete_correct":ok,"text_correct":ok,"valid_UTF8":True,"EOS":True})
        s=m.summarize(data)
        self.assertEqual(s["complete_correct"],21)
        self.assertEqual(s["both_members_complete_correct"],0)
if __name__=="__main__":unittest.main()
