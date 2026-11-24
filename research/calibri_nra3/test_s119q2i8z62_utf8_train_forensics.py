import unittest
from research.calibri_nra3.s119q2i8z62_utf8_train_forensics import classify
class ByteTok:
    eos_token_id=2
    all_special_ids=[2,3]
    def convert_ids_to_tokens(self,i):
        return {2:"<|end|>",3:"<|other|>",4:"Ð",5:"Ŀ",6:"a",7:"Ñ"}[i]
class TestZ62(unittest.TestCase):
    def test_incomplete_not_invalid(self):
        t=ByteTok()
        self.assertEqual(classify(t,[],4)[2],"VALID_INCOMPLETE")
        self.assertEqual(classify(t,[4],5)[2],"COMPLETES_PENDING_UTF8")
        self.assertEqual(classify(t,[4],6)[2],"BROKEN_BYTE_TRANSITION")
        self.assertEqual(classify(t,[4],2)[2],"EOS_ON_INCOMPLETE")
        self.assertEqual(classify(t,[4,5],2)[2],"EOS_ON_COMPLETE")
        self.assertEqual(classify(t,[],3)[2],"NON_EOS_SPECIAL")
if __name__=="__main__":unittest.main()
