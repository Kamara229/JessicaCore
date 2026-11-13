"""S196 CPU physics tests: rank1344 is real, no frozen weight deception."""
import hashlib
import json
import tempfile
import unittest
import zlib
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from calibri2.s132.progressive_core import (
    ProgressiveRouteCore, expand_preserving_previous, IDS_KEY, WEIGHTS_KEY,
    initial_active_indices,
)
from calibri2.s196.core_growth_budget import (
    verify_real_growth_provenance, pack_lossless, physical_budget, plan_to_135m,
    PREVIOUS_PARAMS, S135_PARAMS, NEXT_TOTAL_PARAMS, AREA_PARAMS,
)
from calibri2.s195_neuro_area.representation_area import NeuroRepresentationArea


def make_test_source(path):
    parent = ProgressiveRouteCore(rank=896)
    with torch.no_grad():
        parent.up[0, 4, 0] = .125
        parent.up[5, 23, 448] = .0625
    grown = expand_preserving_previous(parent, 1344)
    sidecar = grown.export_active_fp16()
    save_file(sidecar, str(path))
    record = {
        "stage":"S135_STAGE2_FIVE_CYCLES_THEN_ZERO_GROWTH",
        "next_rank":1344,
        "previous_core_architecture":PREVIOUS_PARAMS,
        "next_core_architecture":S135_PARAMS,
        "new_channels_trained":False,
        "new_architecture_initial_up_exact_zero":True,
        "old_456_exact_EOS_parity":True,
        "new_128_exact_EOS_parity":True,
        "newly_learned_params_from_structural_growth":0,
        "CPU_speed_ratio":1.0,
        "rate_bpp_all_runtime_included":.6400667,
        "physical_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "old_456_full_correct":361,
        "new_128_full_correct":17,
        "five_J7_cycles_verified":[{
            "cycle":i,
            "lost_previously_correct_full":0,
            "lost_previously_correct_first":0,
            "physical_reloaded":True,
            "source_immutable":True,
            "transaction_committed":False,
            "transaction_rollback_verified":True,
        } for i in range(1,6)],
    }
    report=path.with_suffix(".json")
    report.write_text(json.dumps(record))
    return report


class CoreGrowth196Tests(unittest.TestCase):
    def test_real_rank1344_46m_and_16_concept_area_count(self):
        self.assertEqual(S135_PARAMS,46_702_400)
        self.assertEqual(S135_PARAMS-PREVIOUS_PARAMS,3_440_640)
        self.assertEqual(AREA_PARAMS,21_900)
        self.assertEqual(NEXT_TOTAL_PARAMS,46_724_300)
        self.assertEqual(len(initial_active_indices(1344)),72)
        self.assertEqual(len(plan_to_135m()),27)
        self.assertEqual(plan_to_135m()[-1]["new_main_core_params"],134_999_360)

    def test_real_s135_physical_strict_immutability_and_no_fake_training(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/"sidecar.safetensors"
            report=make_test_source(file)
            verified,size,loaded=verify_real_growth_provenance(report,file)
            self.assertEqual(size,file.stat().st_size)
            self.assertEqual(verified["newly_learned_params_from_structural_growth"],0)
            self.assertEqual(len(loaded[IDS_KEY]),72)
            self.assertEqual(int(torch.count_nonzero(loaded[WEIGHTS_KEY][:,:,-24:])),0)
            # An explicit single mutable tensor changes the immutable archive SHA.
            t=load_file(str(file))
            t[WEIGHTS_KEY]=t[WEIGHTS_KEY].clone()
            t[WEIGHTS_KEY][0,1,2]=.125
            save_file(t,str(file))
            with self.assertRaises(AssertionError):
                verify_real_growth_provenance(report,file)

    def test_lossless_46m_growth_and_physical_NRA_FP16(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"grown.safetensors"
            make_test_source(path)
            area=NeuroRepresentationArea()
            r=pack_lossless(
                physical_path=path,area=area,
                physical_out=Path(d)/"grown.z",
                region_out=Path(d)/"region.z")
            self.assertEqual(zlib.decompress((Path(d)/"grown.z").read_bytes()),
                             path.read_bytes())
            self.assertEqual(r["growth_plain_bytes"],path.stat().st_size)
            self.assertGreater(r["region_packed_bytes"],0)
            self.assertEqual(len(zlib.decompress((Path(d)/"region.z").read_bytes())),
                             r["region_plain_bytes"])
            with self.assertRaises(FileExistsError):
                pack_lossless(physical_path=path,area=area,
                    physical_out=Path(d)/"grown.z",region_out=Path(d)/"region.z")

    def test_budget_requires_actual_bytes_never_future_135m_denominator(self):
        # Conservatively account an exact physical replacement, no free zeros.
        sample=physical_budget(parent_bpp=.6400667,
            old_physical_bytes=200_000,packed_growth_bytes=150_000,
            packed_region_bytes=23_000,new_source_bytes=45_000,
            extra_manifest_bytes=7000)
        self.assertLess(sample["architectural_bpp"],.7)
        self.assertGreater(sample["conservative_no_zero_growth_bpp"],
                           sample["architectural_bpp"])
        bigger=physical_budget(parent_bpp=.6400667,
            old_physical_bytes=200_000,packed_growth_bytes=150_000,
            packed_region_bytes=23_000,new_source_bytes=145_000,
            extra_manifest_bytes=7000)
        self.assertGreater(bigger["physical_bits"],sample["physical_bits"])
        with self.assertRaises(ValueError):
            physical_budget(parent_bpp=.8,old_physical_bytes=3,
                            packed_growth_bytes=2,packed_region_bytes=2,
                            new_source_bytes=0)


if __name__=="__main__":
    unittest.main()
