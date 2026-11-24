import unittest
from research.calibri_nra3.s119q2i8z98_micro_transfer import own_metrics,verify_z96
class TestZ98(unittest.TestCase):
    def test_groups(self):
        rows=[{"id":str(i),"family":("cloze","contrast","plural")[i//12],"complete_correct":i<12,"text_correct":i<12,"EOS":True,"valid_UTF8":True,"prefix_match":2,"generated_ids":[1,2]} for i in range(36)]
        r=own_metrics(rows)
        self.assertEqual(r["complete_correct"],12)
        self.assertEqual(r["by_family"]["cloze"]["complete_correct"],12)
        self.assertEqual(r["by_family"]["plural"]["complete_correct"],0)
        self.assertEqual(r["forwards"],72)
    def test_missing_pin(self):
        with self.assertRaises(Exception):verify_z96({})
if __name__=="__main__":unittest.main()
