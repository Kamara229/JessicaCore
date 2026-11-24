import unittest
from research.calibri_nra3.s119q2i8z82_first_error import prefix_match
class TestZ82(unittest.TestCase):
    def test_prefix(self):
        self.assertEqual(prefix_match([1,9,3],[1,2,3]),1)
        self.assertEqual(prefix_match([1,2],[1,2,3]),2)
        self.assertEqual(prefix_match([1,2],[1,2]),2)
        self.assertEqual(prefix_match([],[1]),0)
if __name__=="__main__":unittest.main()
