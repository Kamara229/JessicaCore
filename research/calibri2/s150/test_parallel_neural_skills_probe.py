"""S150 small CPU skill gradient smoke, separate from macOS MPS capacity test."""
import unittest
from calibri2.s150.parallel_neural_skills_probe import run_training

class TestS150SkillProbe(unittest.TestCase):
    def test_cpu_real_autograd_skill_only(self):
        p=run_training("cpu_test", "cpu", (2,8), 150100, steps=10,
                       seq_len=12, ref_params=4096)
        self.assertEqual(p["device"],"cpu")
        self.assertEqual(p["skill_trainable_parameters"],10882)
        self.assertTrue(p["genuine_autograd"])
        self.assertLess(p["last_synthetic_loss"],p["first_synthetic_loss"])
        self.assertFalse(p["model_accepted"])

if __name__ == "__main__":
    unittest.main()
