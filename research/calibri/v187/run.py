"""Actual pretrained original 4D cross-row source pilot, not end-to-end QA.

A formal source/rate/operator CPU *screening* step only.
No complete CALIBRI 4D model is serialized here; do NOT call source
error, GEMV equivalence, or isolated throughput real-answer success.
"""
import argparse,hashlib,json,statistics,time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from binary_geometry_v64 import MODEL,MODEL_SHA
from v84.sources import original_teacher
from v174.run import save
from direct_codebook_v59 import compress_matrix,QuantHandle
from direct_runtime_v59 import NativeKernel,compile_kernel
from .codec import quantize,reconstruct
from .native import Kernel187
from . import protocol as p


def to_cb59_handle(name,full):
    packed=compress_matrix(full)
    h=QuantHandle(name,tuple(full.shape),
        np.frombuffer(packed["packed"],dtype=np.uint8),
        np.frombuffer(packed["row_scales"],dtype="<f2").astype(np.float32),
        np.frombuffer(packed["centroids"],dtype="<f2").astype(np.float32).reshape(32,8))
    expected= len(packed["packed"])+len(packed["row_scales"])+len(packed["centroids"])
    return h,packed,expected


def relative(a,b):
    return float(np.linalg.norm((a.astype(np.float64)-b.astype(np.float64)).ravel())/
                 max(np.linalg.norm(a.astype(np.float64).ravel()),1e-30))


def benchmark(before,after,baseline,changed,seed):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(1,baseline.shape[1])).astype(np.float32)
    before.linear(baseline,x)
    after.linear(changed,x)
    observations=[]
    for k in range(p.REPEATS):
        methods=(("baseline","4D") if k%2==0 else ("4D","baseline"))
        recorded={}
        for label in methods:
            started=time.perf_counter()
            if label=="baseline":before.linear(baseline,x)
            else:after.linear(changed,x)
            recorded[label]=time.perf_counter()-started
        observations.append({"repeat":k,"seconds":recorded,
            "relative_speed_4D_to_CB59":recorded["baseline"]/recorded["4D"]})
    return {"paired_runs":observations,
        "median_direct_GEMV_speed_4D_vs_CB59":statistics.median(
            row["relative_speed_4D_to_CB59"] for row in observations),
        "this_is_single_matrix_speed_NOT_full_transformer_throughput":True}


def run(output):
    torch.set_num_threads(2)
    torch.manual_seed(p.SEED)
    output.mkdir(parents=True,exist_ok=True)
    rules={"original_pretrained_model":MODEL,"revision":MODEL_SHA,
        "original_unique_parameters":p.TOTAL_UNIQUE,
        "unmodified_original_CB59_size":p.SOURCE_BYTES,
        "source_sample_matrices":list(p.MODEL_PILOT),
        "preselected_4D_axes":[2,4],"codebook_5bit_entries":p.K,
        "groups_one_per_8_original_weights":True,
        "all_original_per_row_gains_paid":True,
        "predeclared_pair_speed_repeats":p.REPEATS,
        "actual_complete_0p7_model_NOT_constructed":True,
        "source_only_no_answer_quality_claim":True,
        "no_private_v89_TEST_or_Jessica_tests":True,
        "no_main_Render_Supabase_APK_live_code_touch":True}
    save(output/"frozen_4D_protocol.json",rules)
    teacher=original_teacher()
    state=teacher.state_dict()
    if state["model.embed_tokens.weight"].data_ptr()!=state["lm_head.weight"].data_ptr():
        raise AssertionError("Original pretrained source tie absent")
    with TemporaryDirectory(dir=output) as temp:
        library=compile_kernel(Path(__file__).with_name("kernel.c"),
                               Path(temp)/"direct4D.so")
        baseline=NativeKernel(library)
        new=Kernel187(library)
        report=[]
        for name in p.MODEL_PILOT:
            real=state[name].detach().cpu().numpy()
            if real.ndim!=2:
                raise AssertionError("One selected original matrix is not 2D")
            source,physical=quantize(name,real)
            cb59,packed,baseline_bytes=to_cb59_handle(name,real)
            if physical["complete_matrix_payload_bytes"]!=baseline_bytes:
                raise AssertionError("4D tile must use SAME code + scale + codebook byte budget")
            decoded=reconstruct(source)
            if decoded.shape!=real.shape or not np.isfinite(decoded).all():
                raise AssertionError("Physical multi-axis output incorrect")
            rng=np.random.default_rng(p.SEED+len(report))
            vector=rng.normal(size=(4,real.shape[1])).astype(np.float32)
            actual=new.linear(source,vector)
            exact=vector@decoded.T
            np.testing.assert_allclose(actual,exact,rtol=1.e-4,atol=2.e-4)
            ids=np.array([0,min(1,real.shape[0]-1),real.shape[0]-1],dtype=np.int64)
            np.testing.assert_allclose(new.embedding(source,ids),
                decoded[ids],rtol=3.e-5,atol=3.e-5)
            norm4=relative(real,decoded)
            baseline_reconstruction=baseline.linear(cb59,vector)
            baseline_ref=vector@real.T
            source_baseline_rmse=float(packed["relative_RMSE"])
            latency=benchmark(baseline,new,cb59,source,p.SEED+len(report))
            data={"name":name,"shape":list(real.shape),
                "physical_payload_bytes_same":physical["complete_matrix_payload_bytes"],
                "4D_weight_relative_L2":norm4,
                "original_CB59_weight_relative_L2":source_baseline_rmse,
                "4D_relative_L2_divided_by_CB59":norm4/max(source_baseline_rmse,1e-25),
                "same_0p625_group_index_rate":True,
                "4D_pilot_single_operator_direct_CPU":latency,
                "native_4D_vs_decoded_FP32_relative_L2":relative(exact,actual),
                "native_original_cb59_vs_real_FP32_activation_L2":
                    relative(baseline_ref,baseline_reconstruction),
                "source_CRC_or_model_global_archive_NOT_built":True}
            report.append(data)
            save(output/"per_matrix_partial.json",report)
            print(json.dumps({"stage":"real_tensor","index":len(report),
                "name":name,"relative_source_error_4D_div_CB59":
                    data["4D_relative_L2_divided_by_CB59"],
                "physical_equal":True,
                "direct_operator_native_speed":
                    latency["median_direct_GEMV_speed_4D_vs_CB59"]}),flush=True)
        complete={"complete":True,"protocol":rules,"operators":report,
            "total_pilot_matrices":len(report),
            "4D_source_better_on_relative_L2_count":
                sum(x["4D_weight_relative_L2"]<
                    x["original_CB59_weight_relative_L2"] for x in report),
            "4D_source_worse_on_relative_L2_count":
                sum(x["4D_weight_relative_L2"]>
                    x["original_CB59_weight_relative_L2"] for x in report),
            "median_4D_to_CB59_relative_weight_error_ratio":
                statistics.median(x["4D_relative_L2_divided_by_CB59"] for x in report),
            "median_single_matrix_native_speed":
                statistics.median(x["4D_pilot_single_operator_direct_CPU"]
                    ["median_direct_GEMV_speed_4D_vs_CB59"] for x in report),
            "NOT_FULL_MODEL_QA_OR_COMPLETE_SOURCE":True,
            "NOT_PRIVATE_TEST":True}
        save(output/"v187_4D_pilot_complete.json",complete)
        print(json.dumps({"stage":"complete",
            "real_pretrained_matrices":len(report),
            "4D_better_error_count":complete["4D_source_better_on_relative_L2_count"],
            "4D_worse_error_count":complete["4D_source_worse_on_relative_L2_count"],
            "median_weight_error_ratio":
                complete["median_4D_to_CB59_relative_weight_error_ratio"],
            "median_direct_GEMV_speed":
                complete["median_single_matrix_native_speed"]}),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    opts=parser.parse_args()
    run(opts.output)
