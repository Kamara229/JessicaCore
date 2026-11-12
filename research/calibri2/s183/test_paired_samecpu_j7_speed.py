"""S183 synthetic contract tests: deterministic labels-free paired speed logic."""
import unittest
from calibri2.s183.paired_samecpu_j7_speed import chosen_keys, describe

class S183SpeedContracts(unittest.TestCase):
    def test_non_outcome_source_stratification(self):
        ids=[f"S30/{5000+i}/left" for i in range(20)]+[
            f"S54/{9000+i}/left" for i in range(20)]+[
            f"S82_new/{82000+i}/right" for i in range(20)]
        routes={k:"S93" for k in ids}
        g=chosen_keys(ids,routes)
        self.assertEqual({x:len(y) for x,y in g.items()},
                         {"S30":12,"S54":12,"non_shadow_control":12})
        self.assertEqual(sum(len(y) for y in g.values()),36)

    def test_fails_closed_insufficient_source_sample(self):
        ids=[f"S30/{5000+i}/left" for i in range(11)]+[
            f"S54/{9000+i}/left" for i in range(20)]+[
            f"S82_new/{82000+i}/left" for i in range(20)]
        with self.assertRaisesRegex(AssertionError,"12\+12"):
            chosen_keys(ids,{k:"S93" for k in ids})

    def test_speed_ratio_is_baseline_seconds_over_shadow_seconds(self):
        x=describe([(2.0,1.0),(4.0,2.0)])
        self.assertEqual(x["paired_trials"],2)
        self.assertEqual(x["s176_median_cpu_seconds"],3)
        self.assertEqual(x["s182_shadow_median_cpu_seconds"],1.5)
        self.assertEqual(x["s182_over_s176_speed"],2)

    def test_invalid_zero_or_negative_timings_rejected(self):
        for rows in ([(1,0),(2,1)],[(1,-3),(2,1)],[(1,1)]):
            with self.assertRaises(AssertionError):
                describe(rows)


if __name__=="__main__":
    unittest.main()
