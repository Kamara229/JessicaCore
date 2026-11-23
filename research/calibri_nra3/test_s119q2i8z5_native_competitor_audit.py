"""S119q2i8z5 strict original43M physical source-only 252 49k checks."""
import unittest
from research.calibri_nra3.s119q2i8z5_native_competitor_audit import (
    STAGE,examine
)
class TestZ5SourceOnly(unittest.TestCase):
    def test_01_never_claim_unproven_43M_native_source(self):
        self.assertIn("ORIGINAL43M_NATIVE49K",STAGE)
        for doc in ({},{"stage":"S119Q2I8Z4"},{"candidate_accepted":True}):
            with self.assertRaises(Exception):examine(doc)
    def test_02_report_requires_SHA_model_SOURCE(self):
        with self.assertRaises(Exception):
            examine({"stage":"FAKE","real_original_S142_43M_original_12GT_49152_native_forwards":252})
if __name__=="__main__":unittest.main()
