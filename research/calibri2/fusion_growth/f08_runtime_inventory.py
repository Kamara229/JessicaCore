"""Physical lower bound for preserving legacy J7 paths alongside Fusion.
Never trains or reads protected prompts; audits only model artifacts/metadata.
A packed storage measurement is not resident RAM or native CPU acceptance.
"""
import argparse
import hashlib
import json
import lzma
import struct
from pathlib import Path
from research.calibri2.fusion_growth.f01_physical_budget import digest,metrics
from research.calibri2.fusion_growth.f03_rate_train_core import decode

PARENT_SHA="9e710de6db41d6bb2a99776080f045c537d4af2f6e971529ba6527bf993828d4"
EXTRAS=(
 ("S94_ALTERNATE_EXACT_CORE_TUNNEL_BANK.bin","alternate profile bank","S94_PROFILE_BANK_FEASIBILITY_REPORT.json","physical_bank_sha256"),
 ("S95_PROMPT_ONLY_HASHED_NEURAL_ROUTER_FP16.safetensors","prompt router","S95_PROMPT_ROUTER_STRICT_TRAIN_AUDIT.json","router_sha256"),
 ("S98_MPS_TRAINED_NUMERIC_OWNER_COPY_FP16.safetensors","owner-copy scorer",None,None),
 ("S160_REAL_PARALLEL_SLOT_FP16.safetensors","parallel binding head","S160_REAL_PARALLEL_456_128_J7_AUDIT.json","physical_FP16_sha256"),
 ("S160_RUNTIME.json","parallel runtime config",None,None),
)
def unique(root,name):
    p=list(root.rglob(name))
    if len(p)!=1:raise ValueError(f"expected exactly one {name}, got {len(p)}")
    return p[0]
def run(root,out):
    packet=unique(root,"K16_embed_K8_body.bundle").read_bytes()
    assert digest(packet)==PARENT_SHA
    archive,side=decode(packet)
    assert archive.meta["original_unique_parameter_count"]==36380480 and len(side)==4
    inventory=[]
    blobs={}
    for name,role,report,key in EXTRAS:
        raw=unique(root,name).read_bytes()
        h=digest(raw)
        if report:
            expected=json.loads(unique(root,report).read_text())[key]
            assert h==expected,(name,h,expected)
        inventory.append({"name":name,"role":role,"bytes":len(raw),"sha256":h,
                          "verification":"historical report digest" if report else "pinned historical artifact; digest recorded"})
        blobs[name]=raw
    manifest=json.dumps(inventory,sort_keys=True,separators=(",",":")).encode()
    extra_raw=sum(map(len,blobs.values()))
    # Self-contained additional archive, charged metadata, exact readback.
    plain=struct.pack("<I",len(manifest))+manifest+b"".join(blobs.values())
    packed=b"FG08"+lzma.compress(plain,preset=6)
    p=lzma.decompress(packed[4:]);n,=struct.unpack("<I",p[:4])
    items=json.loads(p[4:4+n]);cursor=4+n
    for item in items:
        raw=p[cursor:cursor+item["bytes"]];cursor+=len(raw)
        assert raw==blobs[item["name"]] and digest(raw)==item["sha256"]
    assert cursor==len(p)
    # Tokenizer is not trained model weights. Disclose actual footprint
    # separately rather than silently hiding it in the base-runtime assumption.
    from transformers import AutoTokenizer
    from binary_geometry_v64 import MODEL,MODEL_SHA
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_SHA)
    assert tok.vocab_size==49152 and tok.eos_token_id==2
    out.mkdir(parents=True,exist_ok=False)
    td=out/"tokenizer"
    tok.save_pretrained(td)
    tokenizer_files=[{"name":x.name,"bytes":x.stat().st_size,"sha256":digest(x.read_bytes())}
                     for x in sorted(td.iterdir()) if x.is_file()]
    tokenizer_bytes=sum(x["bytes"] for x in tokenizer_files)
    (out/"legacy_extras.bundle").write_bytes(packed)
    report={"stage":"F08_LEGACY_RUNTIME_LOWER_BOUND","parent_sha256":PARENT_SHA,
            "inventory":inventory,"known_F03":metrics(len(packet)),
            "additional_raw_bytes":extra_raw,"additional_packed_bytes":len(packed),
            "raw_extra_lower_bound":metrics(len(packet)+extra_raw),
            "packed_extra_lower_bound":metrics(len(packet)+len(packed)),
            "tokenizer_files":tokenizer_files,"tokenizer_bytes_separate":tokenizer_bytes,
            "with_uncompressed_tokenizer":metrics(len(packet)+len(packed)+tokenizer_bytes),
            "all_in_certified":False,"candidate_accepted":False,"output_gate":0,
            "limitations":["F03 core-only checkpoint, no F06 result assumed.",
                           "Additional legacy model files used by S194 loader; no claim the legacy router/profile bank works with K8/K16 codebooks.",
                           "No complete deployment source/config/native kernel accounting yet; no J7/CPU replay.",
                           "Tokenizer reported separately and included in an alternative total; neither is an acceptance decision.",
                           "Storage archives require restore and do not establish runtime speed/RAM."]}
    (out/"F08_REPORT.json").write_text(json.dumps(report,indent=2))
    print("F08_ASSESS",json.dumps(report,sort_keys=True),flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
