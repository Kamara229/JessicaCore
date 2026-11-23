"""S119q2i8z9 fully preregistered author TRAIN21 / untouched DEV14."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8z9_new_ru_train_dev_source import KINDS,rows,verify,digest
from research.calibri_nra3.s119q2i8z2_new_ru_train_dev_source import rows as spent_z2_rows

class TestFreshZ9Source(unittest.TestCase):
    def test_source_sha_and_21_train_14_dev(self):
        m=verify()
        self.assertEqual(m["fresh_Z9_TRAIN_rows"],21)
        self.assertEqual(m["fresh_Z9_DEV_rows"],14)
        self.assertEqual(m["Z9_21_TRAIN_sha256"],digest(rows("TRAIN")))
        self.assertEqual(m["Z9_14_DEV_sha256"],digest(rows("DEV")))
        self.assertTrue(m["all_Z9_prompts_disjoint_prior_E_G_K_R_T_W_X_Z2"])
    def test_all_seven_domains_and_strict_unique_sources(self):
        self.assertEqual(Counter(x["kind"] for x in rows("TRAIN")),{k:3 for k in KINDS})
        self.assertEqual(Counter(x["kind"] for x in rows("DEV")),{k:2 for k in KINDS})
        allrows=rows("TRAIN")+rows("DEV")
        self.assertEqual(len({x["id"] for x in allrows}),35)
        self.assertEqual(len({x["prompt"].casefold().strip() for x in allrows}),35)
        self.assertTrue({x["prompt"].strip().casefold() for x in allrows}.isdisjoint(
            y["prompt"].strip().casefold() for y in spent_z2_rows("TRAIN")+spent_z2_rows("DEV")))
    def test_split_fail_closed_and_no_protected_sealed(self):
        for bad in ("616","J7","GOLD","EVAL128","TEST","DEV+TRAIN"):
            with self.subTest(bad=bad),self.assertRaises(ValueError):rows(bad)
if __name__=="__main__":unittest.main()
