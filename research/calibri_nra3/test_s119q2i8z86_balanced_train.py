import unittest
from research.calibri_nra3.s119q2i8z86_balanced_train import ARMS,validate_epoch
class TestZ86(unittest.TestCase):
    def test_state_guard(self):
        base={"pending_positions":123,"complete_nonterminal_positions":1397,"pending_margin":0.,"complete_margin":0.}
        validate_epoch(base,ARMS[0],1618)
        with self.assertRaises(ValueError):validate_epoch(base,ARMS[1],1618)
        treatment={**base,"pending_margin":1.,"complete_margin":2.}
        validate_epoch(treatment,ARMS[1],1618)
        with self.assertRaises(ValueError):validate_epoch(treatment,ARMS[0],1618)
        with self.assertRaises(ValueError):validate_epoch({**treatment,"pending_positions":124},ARMS[1],1618)
if __name__=="__main__":unittest.main()
