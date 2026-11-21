"""NRA3 S119c: physical immutable S115 trained-FP16 neural child entropy audit.

Bounded research-only experiment. Check immutable full source SHA and original
S194 616-J7 report, inspect ALL actual 53,751 neural FP16 weights by tensor.
Reversible container zlib/LZMA compression, with bit-exact roundtrip. This is
a *storage-only* feasibility test; does NOT install an inference decoder, change
original model, train, run 616 on a candidate, benchmark CPU or score GOLD.

A lossless envelope saves NO runtime FP16 memory and does NOT prove inference
bpp; any extra decoder code, startup cost, and complete actual model loader
inventory are still uncharged. Keep acceptance disabled.
"""
from __future__ import annotations
import argparse, hashlib, json, lzma, struct, zlib
from collections import Counter
from pathlib import Path
from research.calibri_nra3.s119b_S194_parent_S115_child_physical_budget import (
    S115_FILE, S115_REPORT, S194_FILE, S115_EXPECTED_SHA256, audit,
    historical_projection, parse_safetensors,
)
from research.calibri_nra3.s119a_hard_four_goals import check

STAGE="NRA3_S119C_REAL_S115_NEURAL_FP16_LOSSLESS_STORAGE_ABLATION_NO_GOLD"
REPORT="NRA3_S119C_REAL_NEURAL_FP16_RECONSTRUCTION_PHYSICAL_BUDGET.json"


def sha256(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def tensor_bytes(raw:bytes):
    length=struct.unpack("<Q",raw[:8])[0]
    hdr=json.loads(raw[8:8+length])
    data=memoryview(raw)[8+length:]
    for name, info in hdr.items():
        if name=="__metadata__":
            continue
        a,b=info["data_offsets"]
        yield name, bytes(data[a:b]), info["shape"]


def envelope(payload:bytes,kind:str)->bytes:
    check(kind in ("zlib1","zlib6","zlib9","lzma6"),"unknown reversible format")
    if kind.startswith("zlib"):
        raw=zlib.compress(payload,level=int(kind[-1]))
    else:
        raw=lzma.compress(payload,format=lzma.FORMAT_XZ,preset=6)
    # all header bytes counted (magic + kind + length + SHA)
    tag=kind.encode("ascii")
    return b"NRA3S119C" + bytes([len(tag)]) + tag + struct.pack("<Q",len(payload)) + bytes.fromhex(sha256(payload)) + raw


def decode_envelope(blob:bytes)->bytes:
    check(blob.startswith(b"NRA3S119C") and len(blob)>60,"missing lossless envelope")
    n=blob[9]
    check(0<n<=16 and len(blob)>9+1+n+8+32,"invalid lossless envelope prefix")
    kind=blob[10:10+n].decode("ascii")
    check(kind in ("zlib1","zlib6","zlib9","lzma6"),"unknown compression kind")
    off=10+n
    length=struct.unpack("<Q",blob[off:off+8])[0]
    check(0<length<2_000_000,"invalid decompressed size")
    expected=blob[off+8:off+40].hex()
    pay=blob[off+40:]
    if kind.startswith("zlib"):
        obj=zlib.decompressobj()
        recovered=obj.decompress(pay,length+1)+obj.flush()
        check(obj.eof and not obj.unused_data and not obj.unconsumed_tail,
              "zlib incomplete or trailing bytes")
    else:
        recovered=lzma.decompress(pay,format=lzma.FORMAT_XZ)
    check(len(recovered)==length and sha256(recovered)==expected,
          "lossless physical roundtrip corrupted")
    return recovered


def tensor_inventory(raw:bytes):
    inventory=[]
    for name, payload, shape in tensor_bytes(raw):
        fp16_units=len(payload)//2
        zeros=sum(payload[i:i+2]==b"\0\0" for i in range(0,len(payload),2))
        z=len(zlib.compress(payload,9))
        inventory.append({
            "name":name,
            "shape":shape,
            "FP16_count":fp16_units,
            "source_tensor_data_bytes":len(payload),
            "literal_positive_zero_FP16_count":zeros,
            "zero_fraction":zeros/fp16_units,
            "isolated_zlib9_data_bytes_EXCLUDING_per_tensor_headers":z,
            "original_tensor_bytes_sha256":sha256(payload),
        })
    return sorted(inventory,key=lambda x:(-x["source_tensor_data_bytes"],x["name"]))


def evaluate(s194:Path, s115:Path, s115_report:Path):
    # S119b performs original source J7 provenance + exact trained child SHA checks.
    previous=audit(s194,s115,s115_report)
    raw=s115.read_bytes()
    check(sha256(raw)==S115_EXPECTED_SHA256,"S115 source changed")
    shape=parse_safetensors(raw)
    rows=tensor_inventory(raw)
    check(sum(x["FP16_count"] for x in rows)==53751,
          "incomplete real child FP16 inventory")
    parent=previous["budget_projection"]["s194_reported_parent_bpp_on_historical_43261760_denominator"]
    options=[]
    for kind in ("zlib1","zlib6","zlib9","lzma6"):
        blob=envelope(raw,kind)
        check(decode_envelope(blob)==raw,"must restore exact S115 original bytes")
        project=historical_projection(parent_bpp=parent,physical_child_bytes=len(blob))
        options.append({
            "codec":kind,
            "compressed_envelope_bytes_INCLUDING_header_and_SHA":len(blob),
            "full_S115_source_byte_identical_after_roundtrip":True,
            "storage_bytes_saved_vs_original_FP16":len(raw)-len(blob),
            "projected_legacy_bpp_UNPAID_decoder_code":project["projected_legacy_bpp"],
            "projected_conservative_bpp_UNPAID_decoder_code":project["projected_conservative_bpp"],
            "headroom_legacy_bytes_UNPAID_decoder_code":project["legacy_projected_headroom_bytes"],
            "headroom_conservative_bytes_UNPAID_decoder_code":project["strict_projected_headroom_bytes"],
            "storage_fit_legacy_UNPAID_decoder_code":project["candidate_passes_historical_rate_projection"],
            "storage_fit_conservative_UNPAID_decoder_code":project["candidate_passes_conservative_rate_projection"],
        })
    best=min(options,key=lambda x:x["compressed_envelope_bytes_INCLUDING_header_and_SHA"])
    return {
        "stage":STAGE,
        "S194_report_sha256":sha256(s194.read_bytes()),
        "S115_trained_full_child_sha256":sha256(raw),
        "S115_old_53743_FP16_FROZEN":True,
        "original_616_J7_historical_baseline_ONLY_candidate_not_replayed":True,
        "original_12layer_Calibri_and_GT_UNTOUCHED":True,
        "SynTagRusB_sealed_GOLD_NEVER_IMPORTED":True,
        "S115_noncommercial_teacher_TRAIN_not_loaded":True,
        "neural_child_original_file_bytes":len(raw),
        "neural_child_original_geometry":shape,
        "neural_FP16_tensors_largest_first":rows,
        "lossless_storage_options":options,
        "best_reversible_storage_option":best,
        "all_inference_runtime_decoder_costs_STILL_UNPAID":True,
        "all_full_loader_inputs_not_inventoried":True,
        "resident_FP16_weights_not_reduced":True,
        "candidate_output_gate":0,
        "candidate_accepted":False,
        "activation_allowed":False,
        "decision":"STORAGE_ONLY_NO_ACTIVATION",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--s194",type=Path,required=True)
    p.add_argument("--s115",type=Path,required=True)
    p.add_argument("--s115-report",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    check(not a.out.exists(),"never overwrite S119c real weight audit")
    report=evaluate(a.s194,a.s115,a.s115_report)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    best=report["best_reversible_storage_option"]
    print("S119C_REAL_FP16_ENTROPY_AUDIT "+json.dumps({
        "source_bytes":report["neural_child_original_file_bytes"],
        "tensors":len(report["neural_FP16_tensors_largest_first"]),
        "best_codec":best["codec"],
        "best_envelope_bytes":best["compressed_envelope_bytes_INCLUDING_header_and_SHA"],
        "legacy_bpp":best["projected_legacy_bpp_UNPAID_decoder_code"],
        "conservative_bpp":best["projected_conservative_bpp_UNPAID_decoder_code"],
        "legacy_headroom_bytes":best["headroom_legacy_bytes_UNPAID_decoder_code"],
        "conservative_headroom_bytes":best["headroom_conservative_bytes_UNPAID_decoder_code"],
        "candidate_accepted":False},sort_keys=True),flush=True)


if __name__=="__main__":
    main()
