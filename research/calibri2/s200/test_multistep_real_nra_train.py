"""S200 structural checks; native CPU learning runs only in research CI."""
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load
import lzma

from calibri2.s195_neuro_area.representation_area import PREFIX
from calibri2.s200.multistep_real_nra_train import (
    STAGE,MILESTONES,LEARNING_RATE,save_candidate,
)


class S200RealMultiStepContracts(unittest.TestCase):
    def test_stages_never_claim_success_without_j7(self):
        self.assertEqual(STAGE,"S200_REAL_MULTISTEP_NRA_TRAIN_ONLY_UNACCEPTED")
        self.assertEqual(MILESTONES,(2,4,8))
        self.assertTrue(0<LEARNING_RATE<.01)

    def test_snapshot_roundtrips_and_cannot_overwrite(self):
        original={
            "unmodified_parent.weight":torch.tensor([1,2,3],dtype=torch.float16),
            PREFIX+"decoder.weight":torch.zeros(2,2,dtype=torch.float16),
            PREFIX+"encoder.weight":torch.ones(2,2,dtype=torch.float16),
        }
        class StubArea:
            def fp16_state(self):
                return {
                    PREFIX+"decoder.weight":torch.full((2,2),.25,dtype=torch.float16),
                    PREFIX+"encoder.weight":torch.ones(2,2,dtype=torch.float16),
                }
        with tempfile.TemporaryDirectory() as d:
            p=save_candidate(original=original,area=StubArea(),out=Path(d),step=2)
            self.assertFalse(p["accepted"])
            self.assertTrue(p["full_616_CPU_J7_still_required"])
            raw=lzma.decompress(Path(p["filename"]).read_bytes())
            restored=load(raw)
            self.assertTrue(torch.equal(restored["unmodified_parent.weight"],original["unmodified_parent.weight"]))
            with self.assertRaises(FileExistsError):
                save_candidate(original=original,area=StubArea(),out=Path(d),step=2)

    def test_no_neural_change_is_rejected(self):
        original={"legacy":torch.tensor([1],dtype=torch.float16),
                  PREFIX+"decoder.weight":torch.zeros(2,2,dtype=torch.float16)}
        class StubArea:
            def fp16_state(self):
                return {PREFIX+"decoder.weight":torch.zeros(2,2,dtype=torch.float16)}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(AssertionError):
                save_candidate(original=original,area=StubArea(),out=Path(d),step=2)


if __name__=="__main__":
    unittest.main()
