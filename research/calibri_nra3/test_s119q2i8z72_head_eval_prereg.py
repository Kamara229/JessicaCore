import unittest
from research.calibri_nra3 import s119q2i8z72_head_eval_prereg as m
class TestZ72(unittest.TestCase):
    def test_prereg(self):
        r=m.verify()
        self.assertEqual((r["cases"],r["pairs"]),(28,14))
        self.assertEqual(r,m.verify())
    def test_empty_not_language(self):
        a=m.score({"accepted_answers":["да"]},"",eos=True,valid_utf8=True)
        self.assertTrue(a["valid_UTF8"])
        self.assertFalse(a["nonempty_valid_UTF8"])
        self.assertFalse(a["complete_correct"])
    def test_refuse_failed_run(self):
        with self.assertRaises(ValueError):
            m.verify_checkpoint_report({},{"run_id":45438751425,"head_sha":"fca63dae986f6b3fc4fd9ffd77622d66de9238fa","conclusion":"failure"})
if __name__=="__main__":unittest.main()
