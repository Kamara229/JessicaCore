"""S188 true CPU paired benchmark source-only determinism synthetic contracts."""
import unittest
from calibri2.s188.paired_samecpu_s182_vs_s187 import groups_for,stats


class S188TruePairedContracts(unittest.TestCase):
    def test_5_source_family_groups_predeclare_20_frozen_S93(self):
        ids=[]
        for family in ("S69","history","S30","S54","S82_new"):
            ids.extend(f"{family}/{i}/left" for i in range(5))
        routes={k:"S93" for k in ids}
        groups=groups_for(ids,routes)
        self.assertEqual(list(groups),[
            "S69","history","S30_unchanged","S54_unchanged","S82_control"])
        self.assertTrue(all(len(v)==4 for v in groups.values()))
        self.assertEqual(len({k for group in groups.values() for k in group}),20)

    def test_fails_closed_missing_S93_history(self):
        ids=[f"{family}/{i}/left" for family in (
            "S69","history","S30","S54","S82_new") for i in range(6)]
        routes={k:("S91" if k.startswith("history/") else "S93") for k in ids}
        with self.assertRaisesRegex(AssertionError,"4 S93"):
            groups_for(ids,routes)

    def test_baseline_over_candidate_speed_ratio(self):
        r=stats([(2.0,1.0),(4.0,2.0)])
        self.assertEqual(r["paired_trials"],2)
        self.assertEqual(r["s182_median_cpu_seconds"],3.0)
        self.assertEqual(r["s187_median_cpu_seconds"],1.5)
        self.assertEqual(r["s187_over_s182_speed_ratio"],2.0)

    def test_invalid_unpaired_or_zero_times_fail(self):
        for rows in ([(1.,0.),(2.,1.)],[(1.,-1.),(2.,1.)],[(1.,1.)]):
            with self.assertRaises(AssertionError):
                stats(rows)


if __name__=="__main__":
    unittest.main()
