import unittest
from research.calibri_nra3.s119q2i8z88_short_word_train import short_metrics,EPOCHS,NATIVE_SHA
class TestZ88(unittest.TestCase):
    def test_score(self):
        rows=[{"id":str(i),"family":"one","complete_correct":False,"text_correct":True,"valid_UTF8":True,"EOS":False,"prefix_match":2,"generated_ids":[1,3]} for i in range(48)]
        self.assertEqual(short_metrics(rows)["correct"],0)
        self.assertEqual(short_metrics(rows)["first_two_correct"],48)
        self.assertEqual(short_metrics(rows)["forwards"],96)
        self.assertEqual(EPOCHS,4)
        self.assertEqual(len(NATIVE_SHA),64)
if __name__=="__main__":unittest.main()
