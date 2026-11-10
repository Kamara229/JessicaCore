"""Synthetic-only S120 deterministic historical TRAIN pair-selection checks."""
import unittest

from calibri2.s120.paired_train_selection import select_train_counterfactual_pairs


class TestHistoricalCounterfactualCohort(unittest.TestCase):
    def setUp(self):
        names=("S30","S54","history","S69","S73","S82_new")
        numbers=(32,32,4,16,16,128)
        self.rows={}
        self.routes={}
        self.output={}
        for fam,n in zip(names,numbers):
            for k in range(n):
                for side in ("left","right"):
                    key=f"{fam}/{k}/{side}"
                    self.rows[key]={"expected":[12 if side=="left" else 16,20,0]}
                    self.routes[key]="S93"
                    self.output[key]=[22,20,0] if (k<2 and side=="left") else list(self.rows[key]["expected"])
        self.assertEqual(len(self.rows),456)

    def test_8_balanced_pair_selection_with_correct_counterparts(self):
        got=select_train_counterfactual_pairs(self.rows,self.routes,self.output,limit=8)
        self.assertEqual(len(got["pairs"]),8)
        self.assertEqual(len(got["training_keys"]),16)
        self.assertEqual(len(got["first_wrong_keys"]),8)
        self.assertEqual(len(got["protected_counterparts"]),8)
        self.assertEqual(sum(got["family_counts"].values()),8)
        self.assertEqual(len(got["family_counts"]),6)
        self.assertEqual(len(set(got["training_keys"])),16)
        for a,b in got["pairs"]:
            self.assertEqual(a.rsplit("/",1)[0],b.rsplit("/",1)[0])
            self.assertEqual({a.rsplit("/",1)[1],b.rsplit("/",1)[1]},{"left","right"})

    def test_routed_S87_S91_pairs_are_excluded_whole(self):
        self.routes["S30/0/right"]="S91"
        self.routes["S54/0/right"]="S87"
        got=select_train_counterfactual_pairs(self.rows,self.routes,self.output,limit=8)
        ids=set(got["training_keys"])
        self.assertNotIn("S30/0/left",ids)
        self.assertNotIn("S54/0/left",ids)

    def test_malformed_and_missing_examples_failclosed(self):
        with self.assertRaises(ValueError):
            select_train_counterfactual_pairs(self.rows,self.routes,self.output,limit=0)
        truncated=dict(self.rows)
        truncated.pop("S82_new/127/right")
        with self.assertRaises(AssertionError):
            select_train_counterfactual_pairs(truncated,self.routes,self.output)
        short=dict(self.output)
        short.pop("S30/0/left")
        with self.assertRaises(AssertionError):
            select_train_counterfactual_pairs(self.rows,self.routes,short)

    def test_insufficient_wrong_first_pairs_fails(self):
        everything_correct={k:list(v["expected"]) for k,v in self.rows.items()}
        with self.assertRaises(AssertionError):
            select_train_counterfactual_pairs(self.rows,self.routes,everything_correct)


if __name__=="__main__":
    unittest.main(verbosity=2)
