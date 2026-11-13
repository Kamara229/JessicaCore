"""S199 physical pass/fail regression; complete 616 generated CPU J7 runs in CI."""
import hashlib
import json
import lzma
from pathlib import Path
import tempfile
import unittest
import zlib

import torch
from safetensors.torch import load,save

from calibri2.s132.progressive_core import IDS_KEY,WEIGHTS_KEY,initial_active_indices
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea,PREFIX
from calibri2.s199.real_trained_nra_cpu_j7 import (
    physical_and_parent,physical_invoice,sha_bytes,MAX_BPP,
)
from calibri2.s196.core_growth_budget import NEXT_TOTAL_PARAMS,PREVIOUS_PARAMS


def physical_fixtures(base):
    src={
        IDS_KEY:torch.tensor(initial_active_indices(1344),dtype=torch.int32),
        WEIGHTS_KEY:torch.zeros((12,320,72),dtype=torch.float16),
    }
    area=NeuroRepresentationArea().fp16_state()
    src.update(area)
    p=base/"parent.zlib"
    p.write_bytes(zlib.compress(save(src),9))
    current={k:v.clone() for k,v in src.items()}
    current[PREFIX+"decoder.weight"][0,0]=0.125
    c=base/"trained.zlib"
    c.write_bytes(zlib.compress(save(current),9))
    parent={
        "stage":"S197_ACTUAL_PHYSICAL_46M_NRA_REAL_CPU_616_J7",
        "exact_protected_584_generated_ids_equal_S194":True,
        "all_in_physical_bpp":{
            "conservative_excluding_new_zero_growth_bpp":.694,
            "all_in_paid_bits":.694*(PREVIOUS_PARAMS+21900)},
        "checkpoint_runtime_physical":{"runtime_packed_sha256":sha_bytes(p.read_bytes())},
    }
    trained={
        "stage":"S198_GENUINE_ONE_CPU_NRA_GRADIENT_STEP_TRAIN_ONLY",
        "actual_CPU_train_steps":1,
        "model_accepted":False,
        "all_legacy_non_NRA_tensors_byte_identical":True,
        "parent_package_SHA256":sha_bytes(p.read_bytes()),
        "candidate_package_SHA256":sha_bytes(c.read_bytes()),
        "train_NRA_Fp16_changed_elements":{PREFIX+"decoder.weight":1},
    }
    (base/"parent.json").write_text(json.dumps(parent))
    (base/"train.json").write_text(json.dumps(trained))
    return p,c,base/"parent.json",base/"train.json"


class S199RealTrainedFull616J7Tests(unittest.TestCase):
    def test_lossless_lzma_real_changed_neural_fp16_and_original_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            p,c,pr,tr=physical_fixtures(folder)
            parent,train,got,proof=physical_and_parent(
                parent_zlib=p,parent_report=pr,candidate_zlib=c,train_report=tr,
                destination=folder/"runtime.safetensors.lzma")
            self.assertEqual(proof["new_NRA_FP16_train_changed"][PREFIX+"decoder.weight"],1)
            self.assertTrue(torch.equal(got[WEIGHTS_KEY],
                                        load(zlib.decompress(p.read_bytes()))[WEIGHTS_KEY]))
            self.assertEqual(
                sha_bytes(lzma.decompress((folder/"runtime.safetensors.lzma").read_bytes())),
                proof["actual_runtime_FP16_sha256"])
            with self.assertRaises(FileExistsError):
                physical_and_parent(
                    parent_zlib=p,parent_report=pr,candidate_zlib=c,train_report=tr,
                    destination=folder/"runtime.safetensors.lzma")

    def test_fail_closed_if_prior_source_or_cpu_immutable_tensors_changed(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            p,c,pr,tr=physical_fixtures(folder)
            candidate=load(zlib.decompress(c.read_bytes()))
            candidate[WEIGHTS_KEY][0,1,0]=.125
            c.write_bytes(zlib.compress(save(candidate),9))
            metadata=json.loads(tr.read_text())
            metadata["candidate_package_SHA256"]=sha_bytes(c.read_bytes())
            tr.write_text(json.dumps(metadata))
            with self.assertRaises(AssertionError):
                physical_and_parent(
                    parent_zlib=p,parent_report=pr,candidate_zlib=c,train_report=tr,
                    destination=folder/"should_reject.lzma")
            self.assertFalse((folder/"should_reject.lzma").exists())

    def test_physical_invoice_does_not_use_future135m_to_hide_bpp(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            file=folder/"a.py"
            file.write_text("x=1\n")
            budget=physical_invoice(
                parent_report={"all_in_physical_bpp":{"all_in_paid_bits":30_000_000}},
                physical={"parent_package_bytes":900_000,
                          "actual_single_runtime_lzma_bytes":860_000},
                manifest_bytes=2000,source_paths=[file])
            self.assertEqual(budget["architectural_bpp"],
                             budget["all_in_bits"]/NEXT_TOTAL_PARAMS)
            self.assertGreater(budget["conservative_bpp_excluding_zero_structural_growth"],
                               budget["architectural_bpp"])
            self.assertLess(MAX_BPP,1.0)
            more=physical_invoice(
                parent_report={"all_in_physical_bpp":{"all_in_paid_bits":30_000_000}},
                physical={"parent_package_bytes":900_000,
                          "actual_single_runtime_lzma_bytes":885_000},
                manifest_bytes=2000,source_paths=[file])
            self.assertGreater(more["all_in_bits"],budget["all_in_bits"])

if __name__=="__main__":
    unittest.main()
