"""S119q2i8e TRAIN provenance and overlap fail-closed smoke tests."""
import unittest
from collections import Counter
from research.calibri_nra3.s119q2i8e_new_ru_multiword_train_source import (
    STAGE,ORIGIN,KINDS,PER_KIND,rows,verify,
)

class TestNewRussianMultiwordTrain(unittest.TestCase):
    def test_01_registration_sha_and_source_limitations(self):
        info=verify()
        self.assertEqual(info["stage"],STAGE)
        self.assertEqual(info["new_TRAIN_only"],70)
        self.assertEqual(len(info["TRAIN_sha256"]),64)
        self.assertTrue(info["NFC_casefold_no_exact_old_train_or_DEV_prompt_collision"])
        self.assertTrue(info["synthetic_and_visible_not_natural_distribution_or_external_blind"])
    def test_02_fail_closed_to_non_training_splits(self):
        for split in ("DEV","EVAL","SEALED_EVAL","CALIBRATION","SOURCE_PROBE",None):
            with self.subTest(split=split),self.assertRaises(ValueError):
                rows(split)
    def test_03_seven_groups_and_true_full_sentences(self):
        r=rows()
        self.assertEqual(Counter(x["kind"] for x in r),{k:PER_KIND for k in KINDS})
        self.assertEqual(len({x["id"] for x in r}),70)
        self.assertTrue(all(x["split"]=="TRAIN" and x["origin"]==ORIGIN
                            and len(x["completion"].split())>=3
                            and x["completion"].endswith(".") for x in r))
    def test_04_repeated_digest_is_deterministic(self):
        self.assertEqual(verify()["TRAIN_sha256"],verify()["TRAIN_sha256"])

if __name__=="__main__":
    unittest.main()
