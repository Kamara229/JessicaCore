"""Structural isolation tests for the 2x2x2 experimental factor hooks."""
import unittest
import torch
from v111.factors import compose, arrangements
from test_v110_localization import Toy

class TestCases(unittest.TestCase):
    def test_eight_predeclared_cases_and_rollback(self):
        a,b=Toy(),Toy()
        layers=tuple(a.model.layers)
        emb,head=a.model.embed_tokens,a.lm_head
        self.assertEqual(len(arrangements()),8)
        for e,blocks,h in arrangements():
            with compose(a,b,embed_int6=bool(e),blocks_int6=range(30) if blocks else (),head_int6=bool(h)):
                self.assertIs(a.model.embed_tokens,b.model.embed_tokens if e else emb)
                self.assertIs(a.lm_head,b.lm_head if h else head)
                self.assertIs(a.model.layers[0],b.model.layers[0] if blocks else layers[0])
            self.assertIs(a.model.embed_tokens,emb)
            self.assertIs(a.lm_head,head)
            self.assertTrue(all(x is y for x,y in zip(a.model.layers,layers)))

    def test_partial_sweep_reverts_even_if_raised(self):
        a,b=Toy(),Toy()
        layers=tuple(a.model.layers)
        with self.assertRaises(RuntimeError):
            with compose(a,b,embed_int6=True,blocks_int6=(25,26,27),head_int6=True):
                raise RuntimeError("forced toy failure")
        self.assertTrue(all(x is y for x,y in zip(a.model.layers,layers)))
        with self.assertRaises(ValueError):
            with compose(a,b,blocks_int6=(2,2)):
                pass

if __name__=="__main__": unittest.main()
