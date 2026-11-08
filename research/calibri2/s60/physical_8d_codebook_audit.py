"""CALIBRI S60 read-only 8D codeword geometry and physical-rate audit.

No model training, no QA/TEST/DEV, no new weights and no inference speed
claims. Measures only *actual stored* K32×8 centroid geometry and 5-bit
code usage in immutable accepted S45 physical host + S56 Green Tunnel.
Geometric entropy is NOT a proven compressible file format or model quality.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from direct_codebook_v59 import CB59, unpack5
from calibri2.s8.broad_replay import SOURCE_BYTES, SOURCE_PARAMS

S45_SHA = "12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
S56_SHA = "503926839336ed7161786fa6d8ccb2a4ae6fee876cb566d0e52db80ff128f6d4"
S56_MANIFEST_SHA = "bb15a2638f9beb33318bbd97e156973043b220e8d8cda81874a69b61798d7c2b"
S56_TOTAL = 3181257
LIMIT_BPP = .7

def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def entropy_bits(p: np.ndarray) -> float:
    p=p[p>0]
    return float(-np.dot(p,np.log2(p)))

def analyze(name, handle):
    rows, cols=handle.shape
    if cols%8 or handle.centroids.shape!=(32,8):
        raise AssertionError("Not the physically stored K32x8 ABI")
    codes=unpack5(handle.packed.tobytes(), rows*cols//8)
    count=np.bincount(codes,minlength=32).astype(np.int64)
    if int(count.sum()) != rows*cols//8:
        raise AssertionError("5-bit physical code count drift")
    p=count/count.sum()
    centers=handle.centroids.astype(np.float64)
    mu=np.sum(centers*p[:,None],axis=0)
    centered=centers-mu
    cov=(centered*p[:,None]).T@centered
    eig=np.linalg.eigvalsh(cov)
    if eig.min() < -1e-8: raise AssertionError("invalid covariance")
    eig=np.maximum(eig,0)
    if eig.sum()>0:
        energy=eig/eig.sum()
        effective_dim=float(2**entropy_bits(energy))
        dominant=float(eig[-1]/eig.sum())
        largest2=float(eig[-2:].sum()/eig.sum())
    else:
        effective_dim=0.0
        dominant=0.0
        largest2=0.0
    e=entropy_bits(p)
    return {
       "tensor":name,
       "shape":[rows,cols],
       "paid_neural_params":rows*cols,
       "physical_8D_groups":int(count.sum()),
       "distinct_codes_used":int(np.count_nonzero(count)),
       "most_common_code_probability":round(float(p.max()),7),
       "observed_code_entropy_bits_per_8D_group":round(e,6),
       "actual_fixed_bits_per_8D_group":5,
       "hypothetical_entropy_gap_bits_per_8D_group":round(5-e,6),
       "weighted_8D_centroid_effective_PCA_dim":round(effective_dim,6),
       "weighted_8D_centroid_largest_eigen_fraction":round(dominant,6),
       "weighted_8D_centroid_top2_energy_fraction":round(largest2,6),
       "per_row_FP16_scales_count":rows,
       "code_counts":count.tolist(),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("s56_dual",type=Path)
    p.add_argument("s56_manifest",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    source=a.source.read_bytes()
    tunnel=a.s56_dual.read_bytes()
    mbytes=a.s56_manifest.read_bytes()
    if (len(source)!=SOURCE_BYTES or sha(source)!=S45_SHA or sha(tunnel)!=S56_SHA or
        sha(mbytes)!=S56_MANIFEST_SHA):
        raise AssertionError("S45/S56 physical SHA mismatch")
    manifest=json.loads(mbytes)
    if manifest["all_inclusive_total_bytes"]!=S56_TOTAL:
        raise AssertionError("S56 parent physical manifest mismatch")
    archive=CB59(source)
    names=sorted(k for k,v in archive.meta["tensor_descriptors"].items()
                 if v["dtype"]=="CB59_K32V8")
    result=[analyze(name,archive.handles[name]) for name in names]
    total_groups=sum(r["physical_8D_groups"] for r in result)
    if total_groups<=0 or len(names)!=85: # 12×7 layer projections + tied embed
        raise AssertionError("Unexpected physical matrix geometry or count")
    observed_entropy=sum(r["observed_code_entropy_bits_per_8D_group"]*
                         r["physical_8D_groups"] for r in result)/total_groups
    weighted_dimension=sum(r["weighted_8D_centroid_effective_PCA_dim"]*
                         r["physical_8D_groups"] for r in result)/total_groups
    report={
      "stage":"CALIBRI_S60_FROZEN_PHYSICAL_MULTIDIMENSIONAL_CODEBOOK_AUDIT",
      "source_sha256":sha(source),
      "green_tunnel_sha256":sha(tunnel),
      "number_unique_compressed_matrices":len(names),
      "unique_architecture_parameter_count":SOURCE_PARAMS,
      "physical_bytes":{"packed_source":len(source),"S56_dual_tunnel":len(tunnel),
                        "S56_manifest":len(mbytes),"all_inclusive":S56_TOTAL},
      "all_inclusive_bpp":round(8*S56_TOTAL/SOURCE_PARAMS,12),
      "under_bpp_ceiling":8*S56_TOTAL/SOURCE_PARAMS<=LIMIT_BPP,
      "actual_code_bits":5,
      "actual_vector_dim":8,
      "whole_model_weighted_observed_code_entropy_bits":round(observed_entropy,6),
      "whole_model_weighted_effective_codebook_PCA_dimensions":round(weighted_dimension,6),
      "lowest_effective_dimensions":sorted(
          ({"tensor":r["tensor"],"dim":r["weighted_8D_centroid_effective_PCA_dim"]}
           for r in result),key=lambda x:x["dim"])[:12],
      "largest_5bit_entropy_gaps":sorted(
          ({"tensor":r["tensor"],"gap":r["hypothetical_entropy_gap_bits_per_8D_group"]}
           for r in result),key=lambda x:-x["gap"])[:12],
      "by_tensor":result,
      "important_caveats":{
        "entropy_gap_is_not_a_paid_compression_result":True,
        "codebook_geometry_is_not_validation_or_language_quality":True,
        "actual_file_remains_CB59_5bit":True,
        "no_train_no_original_FP32_teacher":True,
        "no_QA_DEV_TEST_S26_S29_S53_S55_S57":True,
        "no_GPU_required":True,
        "no_LIVE_Jessica_Render_Supabase_APK_mutations":True,
      }
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!="by_tensor"},
                     ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
