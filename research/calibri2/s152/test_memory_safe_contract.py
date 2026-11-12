"""S152 guard that real MPS Skill training is scheduled without dual backbone loads."""
import inspect
import unittest
from pathlib import Path
from calibri2.s152.memory_safe_s142_skills import WORKERS, candidate_is_safe, run_shared_host

class TestS152MemorySafety(unittest.TestCase):
    def test_exactly_two_gpu_one_cpu_domains(self):
        self.assertEqual(len(WORKERS),3)
        self.assertEqual([x[1] for x in WORKERS],["mps","mps","cpu"])
        self.assertEqual(len({x[0] for x in WORKERS}),3)
    def test_fail_closed_without_real_train_gain(self):
        self.assertFalse(candidate_is_safe({}))
        self.assertFalse(candidate_is_safe(None))
        self.assertFalse(candidate_is_safe({"research_safe_development_candidate": True}))
    def test_explicit_staged_mps_and_cpu_concurrency(self):
        source=inspect.getsource(run_shared_host)
        self.assertIn('spawn_one(WORKERS[0])',source)
        self.assertIn('spawn_one(WORKERS[2])',source)
        self.assertIn('if "language_gpu" not in state and state["binding_gpu"]["returncode"] is not None',source)
        self.assertIn('if device=="mps" and any(',source)
        self.assertIn("q[\"device\"]==\"mps\" and q[\"returncode\"] is None",source)

if __name__=="__main__":
    unittest.main()
