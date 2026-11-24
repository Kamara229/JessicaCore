import unittest
from research.calibri_nra3.s119q2i8z106_train_word_forensics import label,aggregate
class TestForensics(unittest.TestCase):
    def row(self,text,valid=True,eos=True):
        return dict(target="мост",generated_text=text,valid_UTF8=valid,EOS=eos,complete_correct=valid and eos and text.strip()=="мост",family="cloze",prefix_match=1)
    def test_labels(self):
        for text,valid,eos,want in [("мост",True,True,"exact"),("мост",True,False,"exact_no_eos"),("река",True,True,"wrong_known_word"),("мо",True,True,"truncated_target"),("мостик",True,True,"target_with_suffix"),("",True,True,"empty"),("x",False,True,"invalid_utf8"),("слон",True,True,"other_text")]:
            self.assertEqual(label(self.row(text,valid,eos)),want)
    def test_aggregate(self):
        rows=[self.row("мост") for _ in range(4)]
        self.assertEqual(aggregate(rows)["all_four_contexts_correct"],1)
        rows[-1]=self.row("река")
        self.assertEqual(aggregate(rows)["all_four_contexts_correct"],0)
        self.assertEqual(aggregate(rows)["distinct_valid_texts"],2)
if __name__=="__main__":unittest.main()
