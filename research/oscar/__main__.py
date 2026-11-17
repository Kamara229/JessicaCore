"""OSCAR executable: python -m oscar --fp32 checkpoint.safetensors ...

Read a compatible original dense FP32 checkpoint safely (safetensors), use
cold or trained-dictionary mode, and export full authenticated native CB61
source. Optional runtime components produce one physical Oscar ZIP, including
the actual Green Tunnel, router and neural sidecar weights.

Accepted language quality is NEVER assumed from CLI conversion success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from direct_codebook_v59 import CB59
from safetensors.torch import load_file

from oscar.codec import OscarConverter
from oscar.pack import bundle,cb61_pack,cb61_unpack


def sha(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def read_fp32(path:Path, *, prior:CB59|None, aliases:dict):
    if not path.is_file() or path.suffix!=".safetensors":
        raise ValueError("OSCAR accepts only local safetensors FP32 source; no pickle")
    state=load_file(str(path),device="cpu")
    if prior is not None:
        alias_prior=prior.meta["source_alias_keys"]
        for name,key in alias_prior.items():
            if name in aliases and aliases[name]!=key:
                raise ValueError("OSCAR explicit alias manifest conflicts with trained prior")
        aliases={**alias_prior,**aliases}
    for alias,canonical in sorted(aliases.items()):
        if canonical not in state:
            raise ValueError("OSCAR safetensors source misses tied canonical "+canonical)
        if alias==canonical:
            raise ValueError("OSCAR alias cannot reference itself")
        if alias in state and not state[alias].equal(state[canonical]):
            raise ValueError("OSCAR FP32 source has conflicting tensors for alias "+alias)
        state[alias]=state[canonical]
    return state


def main(argv=None):
    p=argparse.ArgumentParser(
        prog="python -m oscar",
        description=("OSCAR: standalone physical FP32-to-quantized-transformer conversion "
                     "(43M and 135M shapes, quality tested separately)"))
    p.add_argument("--fp32",required=True,type=Path,
                   help="original model FP32 .safetensors checkpoint (not pickle)")
    p.add_argument("--mode",choices=("cold","inherited"),required=True)
    p.add_argument("--prior",type=Path,
                   help="required for inherited mode: original trained .cb59 or .s61")
    p.add_argument("--aliases",type=Path,
                   help="optional JSON object {alias_weight_name:canonical_weight_name}")
    p.add_argument("--depth",type=int,required=True,
                   help="original model decoder layer count, e.g. 12 or 30")
    p.add_argument("--nominal-params",type=int,
                   help="optional physically declared architectural growth denominator")
    p.add_argument("--component",action="append",default=[],
                   metavar="filename=local_path",
                   help="all real inference GT/router/copy weights, repeated per component")
    p.add_argument("--output",type=Path,required=True,
                   help="output folder for actual Oscar core / optional complete runtime")
    p.add_argument("--derived-from-calibri",action="store_true",
                   help="true only if FP32 checkpoint comes from decoded CALIBRI")
    p.add_argument("--max-bpp",type=float,default=.7)
    args=p.parse_args(argv)
    if args.depth<=0 or not 0<args.max_bpp<=4:
        p.error("OSCAR original architecture depth and rate cap invalid")
    if args.mode=="inherited" and args.prior is None:
        p.error("OSCAR inherited mode requires --prior trained dictionary")
    if args.mode=="cold" and args.prior is not None:
        p.error("OSCAR cold mode forbids --prior")
    prior=None
    if args.prior:
        prior_raw=args.prior.read_bytes()
        prior=CB59(cb61_unpack(prior_raw) if prior_raw.startswith(b"CB61ZL01")
                   else prior_raw)
    aliases=json.loads(args.aliases.read_text()) if args.aliases else {}
    if (not isinstance(aliases,dict) or any(
        not isinstance(x,str) or not isinstance(y,str)
        for x,y in aliases.items())):
        p.error("OSCAR aliases must be a JSON string-to-string object")
    fp32_sha=sha(args.fp32.read_bytes())
    state=read_fp32(args.fp32,prior=prior,aliases=aliases)
    result=OscarConverter(mode=args.mode,prior=prior).convert(
        state,layer_count=args.depth,
        derived_from_calibri=args.derived_from_calibri)
    nominal=args.nominal_params or result.source_unique_params
    if nominal<result.source_unique_params:
        p.error("OSCAR cannot use a nominal denominator smaller than original base")
    folder=args.output
    folder.mkdir(parents=True,exist_ok=True)
    output_cb61=cb61_pack(result.raw_cb59)
    source=folder/"Oscar_full_converted_CB61.s61"
    source.write_bytes(output_cb61)
    if cb61_unpack(source.read_bytes())!=result.raw_cb59:
        raise AssertionError("OSCAR executable physical CB61 reload failed")
    components={}
    for item in args.component:
        if "=" not in item:
            p.error("OSCAR --component must be filename=local_path")
        name,path=item.split("=",1)
        if (not name or "/" in name or "\\" in name
            or name in components or name in ("core.cb61","manifest.json")):
            p.error("OSCAR component names must be unique plain filenames")
        components[name]=Path(path).read_bytes()
    packaged=None
    if components:
        pack=bundle(folder/"Oscar_complete_runtime_candidate.oscar.zip",
            output_cb61,components,
            source_params=result.source_unique_params,
            nominal_params=nominal,
            metadata={
               "Original_architecture_depth":args.depth,
               "source_FP32_file_SHA256":fp32_sha,
               "mode":args.mode,
               "fp32_derived_from_calibri":args.derived_from_calibri,
               "NEVER_assume_quality_without_live_EVAL":True})
        packaged=pack.as_dict()
    core_only_bpp=8*len(output_cb61)/result.source_unique_params
    acceptance={
       "OSCAR_algorithm_version":"0.1",
       "input_FP32_safetensors_SHA256":fp32_sha,
       "mode":args.mode,
       "conversion":result.summary(),
       "original_trained_prior_used":prior is not None,
       "original_fp32_derived_from_calibri_DECLARED":args.derived_from_calibri,
       "physical_runtime":packaged,
       "source_only_CB61_file":source.name,
       "source_only_CB61_bytes":source.stat().st_size,
       "core_only_bpp":core_only_bpp,
       "source_only_vs_inference_bundle_distinguished":True,
       "max_bpp_requested":args.max_bpp,
       "passes_requested_nominal_bpp":(
           (packaged["nominal_0p7_bpp"] if packaged else core_only_bpp)
           <=args.max_bpp),
       "passes_requested_CONSERVATIVE_bpp":(
           (packaged["conservative_excluding_generated_growth_bpp"]
            if packaged else core_only_bpp)<=args.max_bpp),
       "model_quality_accepted":False,
       "fresh_independent_FP32_teacher_validation_performed":False,
       "native_C_speed_verified":False,
       "NRA_absent":True,
    }
    report=folder/"Oscar_convert_FP32_actual_physical_report.json"
    report.write_text(json.dumps(acceptance,indent=2,
                                 ensure_ascii=False,sort_keys=True)+"\n",
                      encoding="utf-8")
    print(json.dumps({"stage":"OSCAR_v01_CLI_actual_FP32_convert",
       "mode":args.mode,
       "source_CB61_sha256":sha(output_cb61),
       "source_bytes":source.stat().st_size,
       "original_unique_params":result.source_unique_params,
       "actual_nominal_runtime_bpp":(
           packaged["nominal_0p7_bpp"] if packaged else core_only_bpp),
       "actual_conservative_runtime_bpp":(
           packaged["conservative_excluding_generated_growth_bpp"]
           if packaged else core_only_bpp),
       "quality_proven":False,
       "output":str(folder)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
