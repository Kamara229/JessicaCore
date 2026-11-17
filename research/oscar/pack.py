"""OSCAR complete inference archive: ONE CB61 source plus each real neural sidecar.

The archive is a deterministic ZIP_STORED file with internal SHA manifest.
Charge the literal FINAL ZIP physical size (including file names, headers and
manifest) against the target original architecture denominator. No FP32 source
or duplicate CB59 plaintext can be embedded alongside CB61.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import struct
import zipfile
import zlib

from direct_codebook_v59 import CB59

CB61_HEADER=struct.Struct("<8sQI32s")
TIME=(1980,1,1,0,0,0)
MAX_SOURCE=300_000_000


def sha(x:bytes):
    return hashlib.sha256(x).hexdigest()


def cb61_pack(raw_cb59:bytes)->bytes:
    """Original native-C-compatible 36M CB61; no quality changes."""
    CB59(raw_cb59)
    header=CB61_HEADER.pack(
        b"CB61ZL01",len(raw_cb59),zlib.crc32(raw_cb59),
        hashlib.sha256(raw_cb59).digest())
    blob=header+zlib.compress(raw_cb59,level=9)
    if cb61_unpack(blob)!=raw_cb59:
        raise AssertionError("OSCAR actual CB61 source failed CRC/SHA roundtrip")
    return blob


def cb61_unpack(blob:bytes)->bytes:
    if len(blob)<CB61_HEADER.size+1:
        raise ValueError("OSCAR truncated CB61 file")
    magic,length,crc,digest=CB61_HEADER.unpack_from(blob)
    if magic!=b"CB61ZL01" or length<1 or length>MAX_SOURCE:
        raise ValueError("OSCAR CB61 invalid source format or impossible length")
    stream=zlib.decompressobj()
    raw=stream.decompress(blob[CB61_HEADER.size:],length+1)
    if (len(raw)!=length or not stream.eof or stream.unused_data
        or stream.unconsumed_tail or zlib.crc32(raw)!=crc or
        hashlib.sha256(raw).digest()!=digest):
        raise ValueError("OSCAR CB61 source corruption or unexpected data")
    CB59(raw)
    return raw


@dataclass
class OscarPhysicalResult:
    path:Path
    bytes:int
    source_unique_params:int
    nominal_params:int
    sha256:str
    source_cb61_sha256:str
    auxiliary_bytes:dict

    @property
    def nominal_bpp(self):
        return 8*self.bytes/self.nominal_params

    @property
    def conservative_core_bpp(self):
        return 8*self.bytes/self.source_unique_params

    def as_dict(self):
        return {
          "oscar_bundle":self.path.name,
          "all_inclusive_actual_bundle_bytes":self.bytes,
          "actual_complete_bundle_SHA256":self.sha256,
          "actual_converted_source_CB61_SHA256":self.source_cb61_sha256,
          "source_unique_FP32_core_parameters":self.source_unique_params,
          "nominal_architecture_parameters":self.nominal_params,
          "nominal_0p7_bpp":self.nominal_bpp,
          "conservative_excluding_generated_growth_bpp":self.conservative_core_bpp,
          "passes_nominal_0p7":self.nominal_bpp<=.7,
          "passes_conservative_0p7":self.conservative_core_bpp<=.7,
          "all_runtime_auxiliary_physical_bytes":self.auxiliary_bytes,
          "teacher_FP32_model_not_included":True,
          "all_components_shipped_once":True,
        }


def bundle(path:Path,cb61:bytes,sidecars:dict[str,bytes],
           *,source_params:int,nominal_params:int,
           metadata:dict|None=None)->OscarPhysicalResult:
    """OSCAR's deployable *candidate package*, NOT quality endorsement.

    Allowed sidecar keys are unique filenames (S142, S94, S95, S98 etc).
    The full ZIP is actual runtime bytes, including manifest and metadata.
    """
    if (source_params<=0 or nominal_params<source_params
        or len(sidecars)!=len(set(sidecars))
        or not sidecars
        or any(not name.endswith(".safetensors") and
               not name.endswith(".bin") for name in sidecars)
        or any(not isinstance(data,bytes) or not data
               for data in sidecars.values())):
        raise ValueError("OSCAR physical runtime missing or duplicate sidecars")
    cb=CB59(cb61_unpack(cb61))
    if cb.meta["original_unique_parameter_count"]!=source_params:
        raise ValueError("OSCAR physical denominator is not the source architecture")
    names=["core.cb61"]+sorted(sidecars)
    if len(names)!=len(set(names)):
        raise ValueError("OSCAR duplicate original inference source")
    components={"core.cb61":cb61,**sidecars}
    runtime={
       "format":"Oscar-v0.1-full-inference-runtime-candidate",
       "source":"core.cb61",
       "source_CB59_unique_pretrained_parameter_denominator":source_params,
       "nominal_denominator_including_growth":nominal_params,
       "runtime_components":{name:{"bytes":len(data),"sha256":sha(data)}
                             for name,data in sorted(components.items())},
       "metadata":metadata or {},
       "NRA_or_live_integration":False,
       "quality_promotion_allowed":False,
    }
    manifest=(json.dumps(runtime,sort_keys=True,separators=(",",":"),
                         ensure_ascii=False)+"\n").encode()
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    bio=io.BytesIO()
    with zipfile.ZipFile(bio,"w",allowZip64=False) as z:
        for name,blob in [*sorted(components.items()),("manifest.json",manifest)]:
            member=zipfile.ZipInfo(name, date_time=TIME)
            member.compress_type=zipfile.ZIP_STORED
            member.create_system=3
            member.external_attr=(0o644<<16)
            z.writestr(member,blob)
    packed=bio.getvalue()
    path.write_bytes(packed)
    if path.read_bytes()!=packed:
        raise AssertionError("OSCAR actual bundle changed after physical write")
    with zipfile.ZipFile(io.BytesIO(packed)) as z:
        if (sorted(z.namelist())!=sorted([*names,"manifest.json"]) or
            z.read("core.cb61")!=cb61):
            raise AssertionError("OSCAR real physical runtime ZIP roundtrip failed")
        recorded=json.loads(z.read("manifest.json"))
        for name,details in recorded["runtime_components"].items():
            if (len(z.read(name))!=details["bytes"] or
                sha(z.read(name))!=details["sha256"]):
                raise AssertionError("OSCAR real physical weight corruption: "+name)
    return OscarPhysicalResult(
        path=path,bytes=len(packed),source_unique_params=source_params,
        nominal_params=nominal_params,sha256=sha(packed),
        source_cb61_sha256=sha(cb61),
        auxiliary_bytes={name:len(sidecars[name]) for name in sorted(sidecars)}
    )
