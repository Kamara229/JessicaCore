"""S61: physically paid, exactly reversible, CPU-native CB59 entropy envelope.

This is NOT a neural-weight change, not re-training, and not an independent
quality improvement. It seeks REAL stored-byte savings to finance a future
larger Green Tunnel. Zlib is standard-library lossless entropy coding of the
actual already-paid 5-bit CB59 file; one-time CPU decode on model startup
restores EXACT identical bytes for the existing native CB59 inference path.
GPU is neither required nor invoked. No QA prompts/data are loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import struct
import time
import zlib
from pathlib import Path

SOURCE_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
SIDE_SHA="503926839336ed7161786fa6d8ccb2a4ae6fee876cb566d0e52db80ff128f6d4"
PARENT_MANIFEST_SHA="bb15a2638f9beb33318bbd97e156973043b220e8d"
# Exact original sha256 is checked below; this variable is not a checksum substitute.
PARENT_MANIFEST_SHA_FULL="bb15a2638f9beb33318bbd97e156973043b220e8d8cda81874a69b61798d7c2b"
SOURCE_BYTES=3141513
SIDE_BYTES=39040
SOURCE_PARAMS=36380480
PARENT_TOTAL=3181257
MAGIC=b"CB61ZL01"
HEADER=struct.Struct("<8sQI32s")
MAX_DECODE_BYTES=20_000_000
ZLIB_LEVEL=6

def sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()

def verify_cb59(raw:bytes)->None:
    if len(raw)!=SOURCE_BYTES or sha(raw)!=SOURCE_SHA:
        raise ValueError("actual immutable S45 source SHA/size mismatch")
    # Standalone CB59 structure+CRC+SHA check, not just comparing a name.
    ch=struct.Struct("<4sBII")
    if len(raw)<ch.size+36:raise ValueError("truncated CB59 archive")
    magic,version,mlen,blen=ch.unpack_from(raw)
    if magic!=b"CB59" or version!=1 or ch.size+mlen+blen+36!=len(raw):
        raise ValueError("bad CB59 complete physical archive")
    if zlib.crc32(raw[:-36])!=struct.unpack_from("<I",raw,len(raw)-36)[0]:
        raise ValueError("CB59 legacy physical CRC mismatch")
    if hashlib.sha256(raw[:-32]).digest()!=raw[-32:]:
        raise ValueError("CB59 legacy physical footer SHA mismatch")

def encode(raw:bytes,level:int=ZLIB_LEVEL)->bytes:
    verify_cb59(raw)
    compressed=zlib.compress(raw,level)
    return HEADER.pack(MAGIC,len(raw),zlib.crc32(raw),hashlib.sha256(raw).digest())+compressed

def decode(packed:bytes)->bytes:
    if len(packed)<HEADER.size+1:
        raise ValueError("CB61 compressed archive too short")
    magic,declared_length,declared_crc,digest=HEADER.unpack_from(packed)
    if magic!=MAGIC or declared_length!=SOURCE_BYTES or declared_length>MAX_DECODE_BYTES:
        raise ValueError("CB61 physical header invalid or overly large")
    o=zlib.decompressobj()
    raw=o.decompress(packed[HEADER.size:],declared_length+1)
    if o.unconsumed_tail or not o.eof or o.unused_data or len(raw)!=declared_length:
        raise ValueError("CB61 decoder overflow, truncated, or trailing data")
    if zlib.crc32(raw)!=declared_crc or hashlib.sha256(raw).digest()!=digest:
        raise ValueError("CB61 decompressed content differs from physical source")
    verify_cb59(raw)
    return raw

def physical_manifest(source_packed:bytes,side:bytes,old_manifest:bytes,manifest_path:Path):
    old=json.loads(old_manifest)
    if old.get("all_inclusive_total_bytes")!=PARENT_TOTAL:
        raise ValueError("S56 physical parent size changed")
    manifest={
       "stage":"CALIBRI_S61_EXACT_ENTROPY_CB59_CPU_STARTUP_RESTORE",
       "codec":"zlib_fixed_level_6_full_CB59_source",
       "format":"CB61ZL01",
       "source_stored_sha256":sha(source_packed),
       "S45_source_original_sha256":SOURCE_SHA,
       "S56_unchanged_controller_sha256":sha(side),
       "source_stored_bytes":len(source_packed),
       "controller_bytes":len(side),
       "baseline_legacy_manifest_sha256":sha(old_manifest),
       "original_architecture_unique_parameters":SOURCE_PARAMS,
       "lossless_binary_exact_restore":True,
       "runtime_requires_GPU":False,
       "CPU_decode_once_before_native_inference":True,
       "no_additional_weights_or_teacher":True,
       "legacy_S56_all_inclusive_bytes":PARENT_TOTAL,
       "not_an_independent_quality_or_speed_claim":True,
    }
    for _ in range(32):
        b=json.dumps(manifest,sort_keys=True,indent=2).encode()
        nxt={
            "manifest_bytes":len(b),
            "all_inclusive_total_bytes":len(source_packed)+len(side)+len(b),
            "all_inclusive_bpp":f"{8*(len(source_packed)+len(side)+len(b))/SOURCE_PARAMS:.12f}",
        }
        if all(manifest.get(k)==v for k,v in nxt.items()): break
        manifest.update(nxt)
    b=json.dumps(manifest,sort_keys=True,indent=2).encode()
    total=len(source_packed)+len(side)+len(b)
    if total!=manifest["all_inclusive_total_bytes"] or total*8/SOURCE_PARAMS>.7:
        raise ValueError("physical codec did not yield an audited <=0.7bpp package")
    manifest_path.write_bytes(b)
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("S45_CB59",type=Path)
    p.add_argument("S56_CONTROLLER",type=Path)
    p.add_argument("S56_MANIFEST",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    source=args.S45_CB59.read_bytes()
    side=args.S56_CONTROLLER.read_bytes()
    previous=args.S56_MANIFEST.read_bytes()
    verify_cb59(source)
    if len(side)!=SIDE_BYTES or sha(side)!=SIDE_SHA:
        raise ValueError("S56 Green Tunnel physically altered")
    if sha(previous)!=PARENT_MANIFEST_SHA_FULL:
        raise ValueError("S56 physically sealed manifest altered")
    code=encode(source)
    restored=decode(code)
    if restored!=source:
        raise AssertionError("CB61 was not byte-for-byte lossless")
    output=args.out/"S61_SOURCE_EXACT_ZLIB_CB59.s61"
    output.write_bytes(code)
    manifest=physical_manifest(code,side,previous,args.out/"S61_PHYSICAL_MANIFEST.json")
    # A reproducible separate CPU consumer can load the original unchanged CB59
    # inference kernel after this one-time expansion. Benchmark repeated cold
    # decompression only; do not confuse it with end-to-end token/s.
    cold_times=[]
    for _ in range(18):
        start=time.perf_counter()
        raw=decode(code)
        cold_times.append(time.perf_counter()-start)
        if raw!=source:raise AssertionError("CPU restore parity regressed")
    size_saved=len(source)-len(code)
    savings_total=PARENT_TOTAL-manifest["all_inclusive_total_bytes"]
    report={
        "stage":"S61_REPRODUCIBLE_PHYSICAL_ENTROPY_CODED_CB59",
        "exact_source_sha256":SOURCE_SHA,
        "exact_restored_sha256":sha(restored),
        "all_CB59_native_weights_bit_identical":True,
        "Green_Tunnel_original_unchanged":True,
        "physical_packed_size_bytes":len(code),
        "CB59_source_size_bytes":len(source),
        "source_bytes_saved_after_physical_envelope":size_saved,
        "source_compression_ratio":len(code)/len(source),
        "all_inclusive_original_bytes":PARENT_TOTAL,
        "all_inclusive_result_bytes":manifest["all_inclusive_total_bytes"],
        "all_inclusive_bytes_freed_for_future_physical_tunnel":savings_total,
        "all_inclusive_bpp":manifest["all_inclusive_bpp"],
        "all_inclusive_bpp_within_0_7":float(manifest["all_inclusive_bpp"])<=.7,
        "cpu_cold_decompression_median_ms":statistics.median(cold_times)*1000,
        "cpu_cold_decompression_p95_ms":sorted(cold_times)[16]*1000,
        "measured_cpu_throughput_ratio":None,
        "measurement_scope":"cold CPU entropy decode only, not neural inference",
        "no_QA_used_or_generalization_claim":True,
        "not_promoted_without_cpu_neural_speed_test":True,
        "no_GPU_used":True,
        "no_Jessica_LIVE_Render_Supabase_APK_changes":True,
    }
    (args.out/"S61_RESULT.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
