#!/usr/bin/env python3
"""CALIBRI CB61: activation-aware error *syndrome* for sub-0.70bpw.

CB61 wraps immutable actual pretrained source CB59 K32V8 codebook weights
and adds small, individually lossless FP16 corrections for SELECTED
source-weight input columns, chosen from the source-only quantization
residual R=W-Q and WikiText TRAIN activation second moments.

This is *analogy* to forward error correction, not a claim that parity
can reconstruct arbitrary discarded lossy information: we explicitly
store selected residual side information, count every physical byte,
and compute y=Qx+sum_j R[:,j]*x_j directly, without dense W recovery.
"""
from __future__ import annotations
import hashlib,json,math,struct,zlib
from pathlib import Path
import numpy as np
import torch
from direct_codebook_v59 import CB59,unpack5

MAGIC=b"CB61"
HEAD=struct.Struct("<4sBIII")  # magic,version,manifest,source,corrections
FOOTER=36
MAX_BYTES_PER_MANIFEST=5_000_000
MAX_COLUMNS_PER_MATRIX=3
RESERVE_MANIFEST_BYTES=16000


def rank_residual_columns(source,base:CB59,activation_second_moments):
    """Return all deterministic candidates; no teacher TEST data used."""
    if set(source)!=set(base.meta["full_original_state_keys"]):
        raise AssertionError("CB61 must see every original source tensor")
    unique_names=set(base.meta["trained_unique_keys"])
    out=[]
    for name in sorted(unique_names):
        if name not in base.handles:
            continue
        h=base.handles[name]
        w=source[name].detach().cpu().numpy().astype(np.float32,copy=False)
        m,n=h.shape
        if tuple(w.shape)!=(m,n):
            raise AssertionError("invalid source matrix shape "+name)
        codes=unpack5(h.packed.tobytes(),m*(n//8)).reshape(m,n//8)
        weight_energy=np.zeros(n,dtype=np.float64)
        # Exact FP16 stored centroids/scales: reconstruct only one
        # source group at a time, not a full Q matrix.
        for g in range(n//8):
            center=h.centroids[codes[:,g]]
            decoded=center*h.row_scales[:,None]
            r=(w[:,g*8:g*8+8]-decoded).astype(np.float64)
            weight_energy[g*8:g*8+8]=np.sum(r*r,axis=0,dtype=np.float64)
        d=activation_second_moments.get(name)
        if d is None:
            d=np.ones(n,dtype=np.float64)
        if len(d)!=n:
            raise AssertionError("TRAIN activation mean-square wrong input shape")
        d=np.asarray(d,dtype=np.float64)
        if np.any(~np.isfinite(d)) or np.min(d)<0:
            raise ValueError("bad source TRAIN activation second moment")
        # Keep ranking invariant if entire layer had extreme scale.
        scores=weight_energy*np.clip(d,0,1e6)
        top=sorted(range(n),key=lambda j:(-float(scores[j]),j))[
            :MAX_COLUMNS_PER_MATRIX]
        for j in top:
            # A signed 2-byte source FP16 residual per output row,
            # PLUS matrix/col index in encoded manifest.
            g,k=divmod(j,8)
            current=h.centroids[codes[:,g],k]*h.row_scales
            exact=w[:,j]-current
            raw=exact.astype("<f2").tobytes()
            if not np.isfinite(np.frombuffer(raw,dtype="<f2")).all():
                raise AssertionError("FP16 source syndrome is not finite")
            measured=float(scores[j])
            out.append({"name":name,"col":j,"rows":m,"data":raw,
                        "gross_train_activation_weighted_error":measured,
                        "rate_cost_bytes":len(raw)+45,
                        "score_per_byte":measured/max(1,len(raw)+45)})
    return sorted(out,key=lambda q:(-q["score_per_byte"],q["name"],q["col"]))


def pack(base_blob,source,activation_second_moments,max_total_bytes):
    """Physically spend free source bits, no uncounted source sidecar."""
    cb=CB59(base_blob,expected_keys=source.keys())
    candidates=rank_residual_columns(source,cb,activation_second_moments)
    allowed=max_total_bytes-len(base_blob)-RESERVE_MANIFEST_BYTES-HEAD.size-FOOTER
    if allowed<0:
        raise AssertionError("full-model 0.70bpw has no free syndrome bytes")
    chosen=[]
    occupied=0
    for item in candidates:
        if occupied+item["rate_cost_bytes"]<=allowed:
            chosen.append(item)
            occupied+=item["rate_cost_bytes"]
    # Actual physical header/manifest/source/syndrome counts are checked.
    while True:
        data=bytearray()
        record=[]
        for item in sorted(chosen,key=lambda t:(t["name"],t["col"])):
            offset=len(data)
            data.extend(item["data"])
            record.append({
                "tensor":item["name"],"input_column":item["col"],
                "rows":item["rows"],"offset":offset,"length":len(item["data"]),
                "format":"f16_exact_source_minus_direct_CB59_row_column",
                "TRAIN_activation_weighted_residual_gain_proxy":
                    item["gross_train_activation_weighted_error"]})
        manifest={
            "format":"CB61 source-only 5bit model plus exact selected residual syndromes",
            "schema":1,
            "parent_CB59_SHA256":hashlib.sha256(base_blob).hexdigest(),
            "parent_full_source_unique_parameter_count":
                cb.meta["original_unique_parameter_count"],
            "selected_residual_columns":record,
            "unique_source_tensors":cb.meta["trained_unique_keys"],
            "all_original_state_tensors":cb.meta["full_original_state_keys"],
            "original_tied_weight_aliases":cb.meta["source_alias_keys"],
            "syndrome_information_not_true_lossless_FEC":True,
            "original_candidate_count":len(candidates),
            "corrected_input_column_count":len(record),
            "no_original_dense_W_reconstruction_at_inference":True,
        }
        meta=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()
        if len(meta)>MAX_BYTES_PER_MANIFEST:
            raise AssertionError("CB61 manifest too large")
        payload=HEAD.pack(MAGIC,1,len(meta),len(base_blob),len(data))+(
            meta+base_blob+bytes(data))
        with_crc=payload+struct.pack("<I",zlib.crc32(payload))
        actual=with_crc+hashlib.sha256(with_crc).digest()
        if len(actual)<=max_total_bytes:
            return actual,manifest,{
                "physical_bytes":len(actual),
                "parent_CB59_bytes":len(base_blob),
                "residual_FP16_column_payload_bytes":len(data),
                "actual_manifest_bytes":len(meta),
                "selected_columns":len(chosen),
                "largest_proxy_scores":[{
                    "tensor":i["name"],"col":i["col"],
                    "proxy":i["gross_train_activation_weighted_error"]}
                    for i in chosen[:8]],
            }
        if not chosen:
            raise AssertionError("CB61 header/metadata alone exceeds rate cap")
        chosen.pop()


class CB61:
    def __init__(self,blob,expected_keys=None):
        if len(blob)<HEAD.size+FOOTER:raise ValueError("CB61 truncated file")
        magic,version,m,b,r=HEAD.unpack_from(blob)
        if (magic!=MAGIC or version!=1 or
            not 0<m<MAX_BYTES_PER_MANIFEST or
            not 0<b<300_000_000 or not 0<=r<20_000_000 or
            len(blob)!=HEAD.size+m+b+r+FOOTER):
            raise ValueError("CB61 invalid physical length/header")
        if hashlib.sha256(blob[:-32]).digest()!=blob[-32:]:
            raise ValueError("CB61 entire physical source SHA mismatch")
        if zlib.crc32(blob[:-36])!=struct.unpack_from("<I",blob,len(blob)-36)[0]:
            raise ValueError("CB61 CRC mismatch")
        self.blob=blob
        self.meta=json.loads(blob[HEAD.size:HEAD.size+m].decode())
        source=blob[HEAD.size+m:HEAD.size+m+b]
        self.base=CB59(source,expected_keys=expected_keys)
        if hashlib.sha256(source).hexdigest()!=self.meta["parent_CB59_SHA256"]:
            raise ValueError("CB61 references different pretrained parent")
        delta=memoryview(blob)[HEAD.size+m+b:-FOOTER]
        self.correct={}
        seen=set()
        for row in self.meta["selected_residual_columns"]:
            name=row["tensor"];col=int(row["input_column"])
            if name not in self.base.handles:
                raise ValueError("syndrome references untrained matrix")
            h=self.base.handles[name]
            if not 0<=col<h.shape[1] or row["rows"]!=h.shape[0]:
                raise ValueError("source-syndrome input column misaligned")
            offset=int(row["offset"]);length=int(row["length"])
            if not 0<=offset<=len(delta) or length!=2*h.shape[0] or (
                offset+length>len(delta)):
                raise ValueError("syndrome source FP16 bounds mismatch")
            if (name,col) in seen:
                raise ValueError("duplicate correction column")
            seen.add((name,col))
            arr=np.frombuffer(delta[offset:offset+length],dtype="<f2"
                              ).astype(np.float32)
            if not np.isfinite(arr).all():
                raise ValueError("FP16 source syndrome contains nonfinite values")
            self.correct.setdefault(name,[]).append((col,arr))
        aliases=self.base.meta["source_alias_keys"]
        for alias,canonical in aliases.items():
            if canonical in self.correct:
                self.correct[alias]=self.correct[canonical]
        if len(seen)!=self.meta["corrected_input_column_count"]:
            raise ValueError("correction count inconsistent with manifest")

    def hash(self):return hashlib.sha256(self.blob).hexdigest()
    def physical_bytes(self):return len(self.blob)


def smoke():
    torch.manual_seed(61)
    toy=torch.nn.Linear(16,24)
    src={"weight":toy.weight.detach().contiguous(),
         "bias":toy.bias.detach().contiguous()}
    from direct_codebook_v59 import encode
    baseline,_,_=encode(src)
    budget=len(baseline)+4096
    packed,meta,info=pack(baseline,src,
                          {"weight":np.linspace(0.1,2,16)},budget)
    q=CB61(packed,expected_keys=src.keys())
    assert q.physical_bytes()<=budget
    assert q.base.physical_bytes()==len(baseline)
    assert q.correct.get("weight")
    c=bytearray(packed);c[len(c)//2]^=1
    try:CB61(bytes(c))
    except ValueError:pass
    else:raise AssertionError("CB61 must reject corrupted byte")
    print("CALIBRI v61 CB61 ACTUAL sub0.7 source syndrome byte / CRC / direct hot residual smoke PASS")


if __name__=="__main__":
    smoke()
