"""S201 fail-closed candidate validation; no QA, TEST, or LIVE."""
import inspect
import tempfile
import unittest
from pathlib import Path
from calibri2.s201 import real_s200_snapshots_full_j7 as s201


class S201NativeCPUCheckpointContracts(unittest.TestCase):
    def test_all_three_precommitted_steps(self):
        self.assertEqual(s201.STEPS,(2,4,8))
        self.assertEqual(s201.S200_RUN,43375165217)
        self.assertGreaterEqual(len(s201.SOURCE_FILES),8)

    def test_rejects_unrecognized_step_before_file_lookup(self):
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d)/"parent"
            report=Path(d)/"report"
            output=Path(d)/"out"
            for step in (0,1,3,5,9):
                with self.assertRaises(AssertionError):
                    s201.load_precommitted_s200(
                        step=step,parent_zlib=parent,parent_report=report,
                        output_path=output)
            self.assertFalse(output.exists())

    def test_source_never_promotes_model_or_modifies_inference_oracle(self):
        source=inspect.getsource(s201)
        self.assertIn('"model_accepted":False',source)
        self.assertIn("full_456_correct_EOS",source)
        self.assertIn("full_128_correct_EOS",source)
        self.assertIn("S200_REAL_NRA_MULTISTEP_TRAIN_REPORT.json",source)
        self.assertNotIn("git checkout main",source)


if __name__=="__main__":
    unittest.main()
