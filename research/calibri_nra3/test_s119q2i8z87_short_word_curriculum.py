import unittest
from research.calibri_nra3.s119q2i8z87_short_word_curriculum import fresh,verify,FAMILIES
class TestZ87(unittest.TestCase):
    def test_exact_balance(self):
        r=fresh()
        self.assertEqual(len(r),48)
        for word in {x["completion"] for x in r}:
            self.assertEqual({x["family"] for x in r if x["completion"]==word},set(FAMILIES))
        self.assertEqual(verify()["total_TRAIN"],146)
if __name__=="__main__":unittest.main()
