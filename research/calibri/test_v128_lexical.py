"""Regression test: token ID 216 is a bare space, not an answer '42'."""
import unittest
import torch
from v128.lexical import content,choices,rank
class FakeTokenizer:
    def encode(self,txt,add_special_tokens=False):
        return {"42":[36,34]," 42":[216,36,34]}[txt]
    def decode(self,ids,skip_special_tokens=True):
        return "".join({216:" ",36:"4",34:"2"}[x] for x in ids)
class Test(unittest.TestCase):
    def test_no_bare_space_as_answer(self):
        source=FakeTokenizer()
        opt=choices(source,"42")
        self.assertEqual(opt[0],([36,34],[],36))
        self.assertEqual(opt[1],([216,36,34],[216],36))
        self.assertNotEqual(opt[1][0][0],opt[1][2])
        logits=torch.zeros(258)
        logits[216]=10.
        logits[36]=2.
        self.assertEqual(rank(logits,216),1)
        self.assertEqual(rank(logits,36),2)
        self.assertTrue(bool(opt[1][1]))
if __name__=="__main__":unittest.main()
