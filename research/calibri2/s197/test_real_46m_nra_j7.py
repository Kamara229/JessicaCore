"""S197 structural regression only: no QA or passing unverified J7 counts."""
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load,load_file,save_file

from calibri2.s132.progressive_core import (
    ProgressiveRouteCore,IDS_KEY,WEIGHTS_KEY,initial_active_indices,
)
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea, PREFIX
from calibri2.s197.real_46m_nra_j7 import (
    pack_46m_from_s142,new_46m_core_type,
    physical_bpp_for_single_package,validate_stage197_runtime,
    S135_PARAMS,NEXT_TOTAL_PARAMS,AREA_PARAMS,PREVIOUS_PARAMS,
)


def pinned43m(path):
    original=ProgressiveRouteCore(rank=896)
    with torch.no_grad():
        original.up[0,3,0]=.0625
        original.up[5,10,448]=-.03125
        original.up[11,319,455]=.0078125
    save_file(original.export_active_fp16(),str(path))
    return original


class Real46MPhysicalJ7Tests(unittest.TestCase):
    def test_real_physical_43m_to_46m_preserves_48_and_adds_zero24(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"s142.safetensors"
            prior=pinned43m(source)
            packaged=Path(d)/"s197_full_46m.safetensors.zlib"
            sizes,physical=pack_46m_from_s142(source,packaged)
            self.assertEqual(len(physical[IDS_KEY]),72)
            self.assertEqual(physical[WEIGHTS_KEY].shape,(12,320,72))
            self.assertTrue(torch.equal(physical[WEIGHTS_KEY][:,:,:48],
                                        prior.export_active_fp16()[WEIGHTS_KEY]))
            self.assertEqual(int(torch.count_nonzero(physical[WEIGHTS_KEY][:,:,48:])),0)
            self.assertEqual(len([k for k in physical if k.startswith(PREFIX)]),5)
            self.assertEqual(sizes["area_physical_weight_count"],AREA_PARAMS)
            self.assertEqual(load(zlib.decompress(packaged.read_bytes()))[IDS_KEY].tolist(),
                             list(initial_active_indices(1344)))
            self.assertGreater(sizes["runtime_plain_bytes"],sizes["runtime_packed_bytes"])

            cores=[]
            Expanded=new_46m_core_type(physical,cores)
            obj=Expanded(rank=896)
            obj.load_active_fp16({
                IDS_KEY:prior.export_active_fp16()[IDS_KEY],
                WEIGHTS_KEY:prior.export_active_fp16()[WEIGHTS_KEY]})
            self.assertEqual(obj.rank,1344)
            self.assertEqual(len(cores),1)
            self.assertTrue(torch.equal(obj.export_active_fp16()[WEIGHTS_KEY],
                                        physical[WEIGHTS_KEY]))
            self.assertEqual(obj.architectural_new_parameter_count(),
                             S135_PARAMS-36_380_480)
            # Cannot silently replace one of the protected original 48 values.
            bad=prior.export_active_fp16()
            bad[WEIGHTS_KEY]=bad[WEIGHTS_KEY].clone()
            bad[WEIGHTS_KEY][0,3,0]=.0009765625
            with self.assertRaises(AssertionError):
                obj.load_active_fp16(bad)

    def test_physically_counted_full_package_and_conservative_gate(self):
        with tempfile.TemporaryDirectory() as d:
            source=Path(d)/"p.safetensors"
            pinned43m(source)
            package,saved=pack_46m_from_s142(source,Path(d)/"new.zlib")
            code_file=Path(d)/"runtime.py"
            code_file.write_text("print('true runtime')",encoding="utf-8")
            baseline={"physical_bpp_including_S194_source":.6990419453181559}
            budget=physical_bpp_for_single_package(
                frozen_s194=baseline,package=package,source_files=[code_file],
                manifest_extra_bytes=4000)
            self.assertGreater(budget["main_core_plus_neural_area_bpp"],0)
            self.assertGreater(budget["conservative_excluding_new_zero_growth_bpp"],
                               budget["main_core_plus_neural_area_bpp"])
            self.assertEqual(budget["new_package_bytes"],package["runtime_packed_bytes"])
            self.assertEqual(NEXT_TOTAL_PARAMS,46_724_300)
            change=physical_bpp_for_single_package(
                frozen_s194=baseline,package=package,source_files=[code_file],
                manifest_extra_bytes=14000)
            self.assertGreater(change["all_in_paid_bits"],budget["all_in_paid_bits"])
            self.assertGreater(change["conservative_excluding_new_zero_growth_bpp"],
                               budget["conservative_excluding_new_zero_growth_bpp"])

    def test_s197_fails_if_parent_checkpoints_or_new_region_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"s142.safetensors"
            pinned43m(p)
            package,full=pack_46m_from_s142(p,Path(d)/"payload.zlib")
            old={
                "stage":"S194_TRUE_FROZEN_CPU_J7_584_PLUS32_BASELINE_ADAPTER",
                "physical_FP16_reload_verified":True,
                "candidate_weight_update_executed":False,
                "J7Retention584_frozen_baseline":{
                    "old_full_eos_correct":456,"new_full_eos_correct":128}}
            new={"stage":"S196_REAL_S135_46M_LOSSLESS_RUNTIME_BPP_PLUS_195_NRA",
                "main_core_params":S135_PARAMS,"neural_area_params":AREA_PARAMS,
                "budget":{"architecture_gate_le_0_7":True},"not_accepted":True}
            validate_stage197_runtime(frozen_s194=old,frozen_s196=new,
                                       source=p,payload=full,report=package)
            compromised=dict(full)
            compromised.pop(next(k for k in full if k.startswith(PREFIX)))
            with self.assertRaises(AssertionError):
                validate_stage197_runtime(frozen_s194=old,frozen_s196=new,
                                           source=p,payload=compromised,report=package)
            bad=dict(new);bad["not_accepted"]=False
            with self.assertRaises(AssertionError):
                validate_stage197_runtime(frozen_s194=old,frozen_s196=bad,
                                           source=p,payload=full,report=package)


if __name__=="__main__":
    unittest.main()
