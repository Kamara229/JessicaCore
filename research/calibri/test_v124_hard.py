"""Do not leak gold to generator and ensure fixed physical controller."""
import unittest
from v124.protocol import TRAIN,VALID,MAX_CHECKPOINT_BYTES
from v124.hard_examples import examples,positive_and_negative,canonical
from v120.critic import GreenTunnel

class Test(unittest.TestCase):
    def test_disjoint_task_prompts(self):
        self.assertEqual(len(TRAIN),51)
        self.assertEqual(len(VALID),20)
        self.assertFalse({q for q,a in TRAIN}&{q for q,a in VALID})
        self.assertLess(sum(p.numel()*4 for p in GreenTunnel().parameters()),
                        MAX_CHECKPOINT_BYTES)
    def test_real_candidate_is_not_gold_injected(self):
        two=TRAIN[:2]
        generated={"CB59_v119":{q:"Yes Yes Yes Yes" for q,a in two},
                   "Pythia14M":{q:"22 22 22" for q,a in two}}
        cases=examples(two,generated)
        self.assertEqual(len(cases),8)
        self.assertEqual(positive_and_negative(cases),(2,6))
        self.assertEqual(canonical("London. "),"london")
        self.assertEqual(canonical("HeLLo??"),"hello")
    def test_genuine_generated_correct_not_forced_negative(self):
        q,ans=TRAIN[0]
        actual={"CB59_v119":{q:ans},"Pythia14M":{q:ans.lower()+"."}}
        cases=examples(((q,ans),),actual)
        self.assertEqual(positive_and_negative(cases),(3,1))

if __name__=="__main__":
    unittest.main()
