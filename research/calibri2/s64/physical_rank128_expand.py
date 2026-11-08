"""S64 physically paid Green Tunnel rank expansion (12 -> 128), CPU-only.

The 8D codebooks remain lossless S61 CB61; the FIRST-color rank-2 gate
remains bit-exact. A larger AFTER-first neural gate carries every original
rank-12 parameter in an identical subspace, with zero new output columns.
Therefore its *initial* result is mathematically identical to S56. This
does NOT establish better language quality; only subsequent TRAIN,
independent blind QA and CPU speed decide whether enlargement is useful.

No torch, GPU, Groq, QA, Jessica, Render, Supabase or inference weights in LIVE.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, struct
from pathlib import Path

PARAMS=36380480
SOURCE_SHA="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398"
CONTROLLER_SHA="503926839336ed7161786fa6d8ccb2a4ae6fee876cb566d0e52db80ff128f6d4"
BASELINE_TOTAL=2492142
MAX_BPP=.7
OLD_RANK=12
NEW_RANK=128
WIDTH=320
SEED=640128
PREFIX="later."

def sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()

def decode_safetensors(blob):
    if len(blob)<8:raise ValueError("empty safetensors")
    header_len=struct.unpack_from("<Q",blob)[0]
    if header_len>1_000_000 or 8+header_len>len(blob):
        raise ValueError("invalid safetensors header length")
    meta=json.loads(blob[8:8+header_len])
    data=blob[8+header_len:]
    tensors={}
    for key,desc in meta.items():
        if key=="__metadata__":continue
        shape=desc["shape"]
        if desc["dtype"]!="F32":
            raise ValueError(f"Unexpected dtype in paid Green Tunnel: {key}")
        beg,end=desc["data_offsets"]
        count=math.prod(shape)
        if beg<0 or end>len(data) or end-beg!=count*4:
            raise ValueError(f"Unsafe tensor shape/offsets: {key}")
        tensors[key]=(list(shape),data[beg:end])
    return meta.get("__metadata__"),tensors

def encode_safetensors(attributes,tensors):
    header={}
    payload=bytearray()
    if attributes:header["__metadata__"]=attributes
    for name in sorted(tensors):
        shape,raw=tensors[name]
        if math.prod(shape)*4!=len(raw):raise ValueError("bad tensor byte length")
        start=len(payload)
        payload.extend(raw)
        header[name]={"dtype":"F32","shape":shape,"data_offsets":[start,len(payload)]}
    content=json.dumps(header,sort_keys=True,separators=(",",":")).encode()
    # safetensors header may be padded with spaces.
    content+=b" "*((-len(content))%8)
    return struct.pack("<Q",len(content))+content+bytes(payload)

def floats(blob):
    if len(blob)%4:raise ValueError("invalid F32 data")
    return struct.unpack("<"+str(len(blob)//4)+"f",blob)

def tensor_expand(data):
    oldmeta,tensors=decode_safetensors(data)
    need={
        "later.down.weight":[OLD_RANK,WIDTH],
        "later.up.weight":[WIDTH,OLD_RANK],
        "later.gate.weight":[1,WIDTH],
        "later.gate.bias":[1],
        "first.down.weight":[2,WIDTH],
        "first.up.weight":[WIDTH,2],
        "first.gate.weight":[1,WIDTH],
        "first.gate.bias":[1],
    }
    for key,shape in need.items():
        if key not in tensors or tensors[key][0]!=shape:
            raise ValueError(f"bad immutable original tunnel shape: {key}")
    if set(tensors)!=set(need):
        raise ValueError("unrecognized extra paid Green Tunnel tensors")
    old_down=floats(tensors["later.down.weight"][1])
    old_up=floats(tensors["later.up.weight"][1])
    rng=random.Random(SEED)
    down=list(old_down)
    for _ in range((NEW_RANK-OLD_RANK)*WIDTH):
        down.append(rng.gauss(0,.028))
    up=[0.0]*(WIDTH*NEW_RANK)
    for row in range(WIDTH):
        for col in range(OLD_RANK):
            up[row*NEW_RANK+col]=old_up[row*OLD_RANK+col]
    tensors["later.down.weight"]=([NEW_RANK,WIDTH],struct.pack("<"+str(len(down))+"f",*down))
    tensors["later.up.weight"]=([WIDTH,NEW_RANK],struct.pack("<"+str(len(up))+"f",*up))
    encoded=encode_safetensors(oldmeta,tensors)
    _,check=decode_safetensors(encoded)
    # Exact source FP32 bytes from old weights, including the rank-2 FIRST
    # controller, remain bit-identical.
    for key,(shape,raw) in decode_safetensors(data)[1].items():
        if key=="later.down.weight":
            if check[key][1][:len(raw)]!=raw:raise AssertionError("down embedding changed")
        elif key=="later.up.weight":
            old=floats(raw); expanded=floats(check[key][1])
            if any(old[i*OLD_RANK+j]!=expanded[i*NEW_RANK+j]
                   for i in range(WIDTH) for j in range(OLD_RANK)):
                raise AssertionError("up embedding changed")
            if any(expanded[i*NEW_RANK+j]!=0.0
                   for i in range(WIDTH) for j in range(OLD_RANK,NEW_RANK)):
                raise AssertionError("new output columns are not zero")
        elif check[key][0]!=shape or check[key][1]!=raw:
            raise AssertionError("old gate/first controller changed")
    return encoded,check

def calculate_manifest(compressed,controller,path):
    info={
      "stage":"CALIBRI_S64_PHYSICAL_RANK128_GREEN_TUNNEL_INIT_NO_TRAIN",
      "codec":"CB61ZL01: unchanged exact reversible original S45 codebook",
      "compressed_source_bytes":len(compressed),
      "compressed_source_sha256":sha(compressed),
      "original_S45_CB59_sha256":SOURCE_SHA,
      "controller_bytes":len(controller),
      "controller_sha256":sha(controller),
      "original_controller_sha256":CONTROLLER_SHA,
      "first_rank":2,"after_first_old_rank":12,"after_first_rank":128,
      "after_first_original_columns_bit_identical":True,
      "after_first_added_output_columns_are_zero":True,
      "all_added_down_rows_seed":SEED,
      "CPU_runnable_without_GPU":True,
      "no_trained_new_quality_claim":True,
      "full_original_S56_answer_replay_required_before_any_promotion":True,
      "independent_heldout_QA_and_CPU_speed_required_before_promotion":True,
      "unique_neural_weight_parameters":PARAMS,
      "no_LIVE_Jessica_Render_Supabase_APK_changes":True,
    }
    for _ in range(25):
        content=json.dumps(info,sort_keys=True,indent=2).encode()
        updated={"manifest_bytes":len(content),"all_inclusive_bytes":
                  len(compressed)+len(controller)+len(content)}
        updated["all_inclusive_bpp"]=f"{8*updated['all_inclusive_bytes']/PARAMS:.12f}"
        if all(info.get(k)==v for k,v in updated.items()):break
        info.update(updated)
    content=json.dumps(info,sort_keys=True,indent=2).encode()
    if len(content)!=info["manifest_bytes"] or len(compressed)+len(controller)+len(content)!=info["all_inclusive_bytes"]:
        raise AssertionError("physical manifest did not converge")
    if 8*info["all_inclusive_bytes"]/PARAMS>MAX_BPP:
        raise ValueError("S64 increase exceeds strict <=.7bpp cap")
    path.write_bytes(content)
    return info

def main():
    a=argparse.ArgumentParser()
    a.add_argument("S61_SOURCE_CB61",type=Path)
    a.add_argument("S56_DUAL_TUNNEL",type=Path)
    a.add_argument("S61_MANIFEST",type=Path)
    a.add_argument("output_dir",type=Path)
    q=a.parse_args()
    q.output_dir.mkdir(parents=True,exist_ok=True)
    src=q.S61_SOURCE_CB61.read_bytes()
    if not src.startswith(b"CB61ZL01"):raise ValueError("not the real S61 physical envelope")
    controller=q.S56_DUAL_TUNNEL.read_bytes()
    if sha(controller)!=CONTROLLER_SHA or len(controller)!=39040:
        raise ValueError("refusing unpinned S56 physical paid controller")
    manifest=json.loads(q.S61_MANIFEST.read_text())
    if manifest.get("source_stored_sha256")!=sha(src) or manifest.get("source_stored_bytes")!=len(src):
        raise ValueError("S61 physical source hash/size mismatch")
    if manifest.get("all_inclusive_total_bytes")!=BASELINE_TOTAL:
        raise ValueError("S61 paid baseline total changed")
    candidate,expanded=tensor_expand(controller)
    (q.output_dir/"S64_RANK128_GREEN_TUNNEL_INIT.safetensors").write_bytes(candidate)
    budget=calculate_manifest(src,candidate,q.output_dir/"S64_PAID_MANIFEST.json")
    result={
       "stage":"CALIBRI_S64_RANK128_GREEN_TUNNEL_PHYSICALLY_PAID",
       "source_original_S61_bytes":len(src),
       "S56_rank12_sidecar_bytes":len(controller),
       "S64_rank128_sidecar_bytes":len(candidate),
       "S64_new_after_first_rank":NEW_RANK,
       "new_neural_adapter_capacity_multiplier_for_after_first":NEW_RANK/OLD_RANK,
       "first_rank2_S56_FP32_parameters_unchanged":True,
       "later_original_subspace_FP32_values_unchanged":True,
       "new_up_columns_exact_zero":True,
       "all_inclusive_bytes":budget["all_inclusive_bytes"],
       "all_inclusive_bpp":budget["all_inclusive_bpp"],
       "physical_headroom_bytes":int(PARAMS*.7/8)-budget["all_inclusive_bytes"],
       "S56_unchanged_quality_at_initialization_is_mathematical_hypothesis_until_neural_CPU_replay":True,
       "model_not_trained_or_promoted":True,
       "no_GPU_or_QA_used":True,
       "no_LIVE_changes":True,
    }
    (q.output_dir/"S64_PHYSICAL_RESULT.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=="__main__":main()
