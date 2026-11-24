import unittest
from research.calibri_nra3.s119q2i8z103_terminal_train import partition,STEPS,GROUPS
class TestZ103(unittest.TestCase):
    def test_partition(self):
        s=[{"id":"S119Q2I8Z87_"+str(i),"gold":[0]*(5 if i<12 else 4)} for i in range(48)]
        r=[{"id":"replay"+str(i),"gold":[0]*(17 if i<50 else 16)} for i in range(98)]
        a,g=partition(s+r)
        self.assertEqual(len(a),48)
        self.assertEqual(len(g),14)
        self.assertEqual(sum(len(x["gold"]) for y in g for x in y),1618)
        self.assertEqual(STEPS//GROUPS,126)
        self.assertEqual(STEPS*204+126*1618,563724)
if __name__=="__main__":unittest.main()
