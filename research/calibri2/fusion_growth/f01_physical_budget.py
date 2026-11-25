"""F01: exact source-bundle accounting and lossless core packing, research only.

Not a deployed format, not an all-in certification: the original CPU loader
is checked after byte-exact restoration, the neural child is SHA-pinned but
not attached. No FP32 teacher, no training or protected evaluation.
"""
from pathlib import Path
import argparse
import hashlib
import json
import lzma
import struct
import tempfile
import time
import zlib

N_CONSERVATIVE = 36_380_480
N_HISTORICAL = 43_261_760
NAMES = (
    "S93_UNACCEPTED_SELECTED_CB61.s61",
    "S93_UNACCEPTED_SELECTED_JOINT.safetensors",
    "S142_ACTUAL_43M_ONE_J7_FP16.safetensors",
    "S119Q2I8Z73_HEAD_REVIVE_ZERO_UNITS_FP16_G0.safetensors",
    "S119Q2I8Z73_HEAD_REVIVE_ZERO_UNITS_GT12_FP16_G0.safetensors",
)
PIN = {
    NAMES[2]: "5a706d9012a61f24678929a2a6ee20f77c922e7e0f4bb516c061f32742cc56c7",
    NAMES[3]: "25acb994cfb8f6e425127da6419b0f47fa5c955b6bbf69d888186e985b1c5256",
    NAMES[4]: "79fd6297ebd350140f15d68bea9146c08e89b767ed6404c7ba311793616d44a6",
}
ROLES = ("core_envelope", "core_and_GT_legacy_side", "core_growth_and_GT12",
         "joint_NRA3_NPA1_projector_native_head", "GT12_trained_gates")

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def metrics(n):
    # Integer threshold avoids floating point boundary errors.
    return {"bytes": n, "conservative_bpp": 8*n/N_CONSERVATIVE,
            "historical_bpp": 8*n/N_HISTORICAL,
            "conservative_headroom_bytes": (7*N_CONSERVATIVE)//80-n,
            "within_conservative_cap": 80*n <= 7*N_CONSERVATIVE}

def pack(files, codec):
    # Include names, lengths, checksums, header and codec overhead in charge.
    meta = [{"name": k, "size": len(v), "sha256": digest(v)}
            for k,v in files.items()]
    head = json.dumps(meta, separators=(",",":")).encode()
    plain = struct.pack("<I",len(head)) + head + b"".join(files.values())
    if codec == "xz":
        return b"FG01X" + lzma.compress(plain, preset=6)
    if codec == "zlib":
        return b"FG01Z" + zlib.compress(plain, level=9)
    raise ValueError(codec)

def unpack(blob):
    if blob[:5] == b"FG01X":
        plain = lzma.decompress(blob[5:])
    elif blob[:5] == b"FG01Z":
        plain = zlib.decompress(blob[5:])
    else:
        raise ValueError("unknown codec")
    n, = struct.unpack("<I",plain[:4])
    entries = json.loads(plain[4:4+n])
    cursor = 4+n
    out = {}
    for entry in entries:
        name = entry["name"]
        if name not in NAMES or name in out:
            raise ValueError("unknown or duplicate name")
        value = plain[cursor:cursor+entry["size"]]
        if len(value) != entry["size"] or digest(value) != entry["sha256"]:
            raise ValueError("damaged payload")
        out[name] = value
        cursor += len(value)
    if cursor != len(plain):
        raise ValueError("trailing bytes")
    return out

def tensor_inventory(raw):
    n, = struct.unpack("<Q",raw[:8])
    header = json.loads(raw[8:8+n])
    tensors = {k:v for k,v in header.items() if k != "__metadata__"}
    offsets = sorted(tuple(v["data_offsets"]) for v in tensors.values())
    cursor = 0
    for a,b in offsets:
        if a != cursor or b < a:
            raise ValueError("non-contiguous tensor payload")
        cursor = b
    if cursor != len(raw)-8-n:
        raise ValueError("bad tensor payload length")
    return {"header_bytes":8+n, "payload_bytes":cursor,
            "tensors": {k:{"shape":v["shape"],"dtype":v["dtype"],
                           "bytes":v["data_offsets"][1]-v["data_offsets"][0]}
                        for k,v in tensors.items()}}

def run(root, out):
    files = {}
    inventory = []
    for name,role in zip(NAMES,ROLES):
        paths = list(root.rglob(name))
        if len(paths) != 1:
            raise ValueError(f"expected exactly one {name}")
        raw = paths[0].read_bytes()
        if name in PIN and digest(raw) != PIN[name]:
            raise ValueError(f"source SHA mismatch: {name}")
        files[name] = raw
        row = {"name":name,"role":role,"sha256":digest(raw),**metrics(len(raw))}
        if name.endswith(".safetensors"):
            row["tensor_inventory"] = tensor_inventory(raw)
        inventory.append(row)
    out.mkdir(parents=True,exist_ok=False)
    measurements = {}
    for codec in ("zlib","xz"):
        t = time.perf_counter()
        blob = pack(files,codec)
        encode_seconds = time.perf_counter()-t
        t = time.perf_counter()
        recovered = unpack(blob)
        decode_seconds = time.perf_counter()-t
        if recovered != files:
            raise AssertionError("not byte exact")
        core_blob = pack(dict(list(files.items())[:3]),codec)
        if unpack(core_blob) != dict(list(files.items())[:3]):
            raise AssertionError("core not byte exact")
        measurements[codec] = {
            **metrics(len(blob)), "core_only_packed_bytes":len(core_blob),
            "encode_seconds":encode_seconds,"decode_seconds":decode_seconds,
            "byte_exact":True}
        (out/f"F01.{codec}.bundle").write_bytes(blob)
    best = min(measurements,key=lambda x:measurements[x]["bytes"])
    from research.calibri_nra3.s119k_actual43m_full_native_lm_head_shadow import load_actual_original
    from safetensors.torch import load_file
    import torch
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        restored = unpack((out/f"F01.{best}.bundle").read_bytes())
        for name,raw in restored.items():
            (d/name).write_bytes(raw)
        model,sources,bridge,full,expansion = load_actual_original(*(d/n for n in NAMES[:3]))
        if len(model.model.layers) != 12:
            raise AssertionError("depth changed")
        child = load_file(str(d/NAMES[3]))
        gates = load_file(str(d/NAMES[4]))
        child_count = sum(t.numel() for t in child.values())
        gate_count = sum(t.numel() for t in gates.values())
        if child_count != 421793 or gate_count != 12:
            raise AssertionError(f"unexpected learned tensors {child_count}, {gate_count}")
        del model,sources,bridge,full,expansion,child,gates
    report = {
        "stage":"F01_PHYSICAL_CORE_BUDGET",
        "source_runs":[42681768394,43078895081,45439790187],
        "depth":12,"core_loader_reopened":True,"child_loaded_not_attached":True,
        "raw":metrics(sum(map(len,files.values()))),
        "raw_core_only_bytes":sum(len(files[n]) for n in NAMES[:3]),
        "codec_measurements":measurements,"smallest_codec":best,
        "inventory":inventory,"child_elements":child_count,"GT_gate_elements":gate_count,
        "denominator_policy":"36,380,480 conservative original trained unique core weights; historical 43,261,760 for comparison only; no zero or duplicated growth inflation",
        "limitations":["Five known source model files only; tokenizer, runtime constants and any other required inputs not yet certified.",
                       "S93 side tensors also occur in S142; solid packing may exploit duplicate bytes, but no file or tensor is removed.",
                       "Lossless archive size is not native runtime representation, CPU memory or throughput.",
                       "No forward, full child integration, J7 replay, independent quality or stable plateau claim."],
        "candidate_accepted":False,"output_gate":0,"FP32_teacher_calls":0,
        "optimizer_steps":0,"all_in_certified":False
    }
    (out/"F01_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F01_ASSESS",json.dumps({k:v for k,v in report.items() if k!="inventory"},sort_keys=True),flush=True)

if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.root,a.out)
