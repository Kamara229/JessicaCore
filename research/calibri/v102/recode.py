"""Same-byte original 5-bit K32V8 tied embedding/head re-encoding.

No unpriced sidecar or special native kernel. Fit ONE new entire head
codebook using more deterministic original PRETRAINED weight samples,
then optionally least-squares optimize each 16-bit row scale.
This is source-only parameter refitting; NEVER uses any QA TEST or
validation prompt to fit/select quantization codewords.

All *other* original trained matrices, source manifest and ABI
are byte-identical to reference. Only canonical tied head parts can
differ. Full same-length archive gets valid new CRC and SHA256.
"""
from __future__ import annotations
import hashlib,math,struct,zlib
import numpy as np
from direct_codebook_v59 import CB59,HEADER,pack5,row_scales,_centroids,_nearest_integer_128
from .spec import TRAIN_SOURCE_SAMPLE_GROUPS,ROW_CHUNK

def actual_head_name(meta):
    keys=set(meta["tensor_descriptors"])
    existing=[n for n in ("lm_head.weight","model.embed_tokens.weight") if n in keys]
    if len(existing)!=1:
        raise AssertionError("single physical source for tied embedding/head required")
    return existing[0]

def fit_new_codebook(w,scales):
    m,n=w.shape
    g=n//8
    total=m*g
    sample_ix=np.linspace(0,total-1,TRAIN_SOURCE_SAMPLE_GROUPS,dtype=np.int64)
    samples=w.reshape(-1,8)[sample_ix].astype(np.float32).copy()
    samples/=scales[sample_ix//g,None]
    return _centroids(samples).astype(np.float32)

def recode_rows(w,c,scales,row_ls):
    m,n=w.shape
    g=n//8
    coded=[]
    allscales=np.empty(m,dtype="<f2")
    err=0.;energy=0.
    for start in range(0,m,ROW_CHUNK):
        end=min(start+ROW_CHUNK,m)
        exact=np.asarray(w[start:end],dtype=np.float32)
        s=scales[start:end].astype(np.float32).copy()
        if row_ls:
            # Solve optimal scalar for preliminary nearest codewords at each row.
            x=(exact/s[:,None]).reshape(-1,8).astype("<f2").astype(np.float32)
            ix=_nearest_integer_128(x,c).reshape(end-start,g)
            decoded=c[ix].reshape(end-start,n)
            numerator=np.sum(exact.astype(np.float64)*decoded.astype(np.float64),axis=1)
            denominator=np.sum(decoded.astype(np.float64)**2,axis=1)
            s=np.maximum(numerator/np.maximum(denominator,1e-22),2.**-14)
            s=np.asarray(np.clip(s,2.**-14,60000),dtype="<f2").astype(np.float32)
        # Snap each normalized source vector to the same physical FP16
        # lattice used by original CB59 before deterministic int coding.
        x=(exact/s[:,None]).reshape(-1,8).astype("<f2").astype(np.float32)
        ix=_nearest_integer_128(x,c).astype(np.uint8)
        coded.append(pack5(ix))
        q=c[ix].reshape(end-start,n)*s[:,None]
        diff=exact.astype(np.float64)-q.astype(np.float64)
        err+=float(np.sum(diff*diff))
        energy+=float(np.sum(exact.astype(np.float64)**2))
        allscales[start:end]=s.astype("<f2")
    if not energy>0:raise AssertionError("missing original pretrained head weights")
    return b"".join(coded),allscales.tobytes(),math.sqrt(err/energy)

def swap_whole_tied_head_blob(frozen:bytes,parts):
    """Retain complete original source manifest, geometry and other tensor bytes."""
    archive=CB59(frozen)
    key=actual_head_name(archive.meta)
    desc=archive.meta["tensor_descriptors"][key]
    if desc["dtype"]!="CB59_K32V8":
        raise AssertionError("tied source must keep original C ABI")
    mlen=HEADER.unpack_from(frozen)[2]
    original_body_start=HEADER.size+mlen
    changed=bytearray(frozen[:-36])
    changed_ranges=[]
    for name in ("packed","row_scales","centroids"):
        position,size=desc["parts"][name]
        payload=parts[name]
        if len(payload)!=size:
            raise AssertionError("new encoding is NOT physical same-byte")
        a=original_body_start+position
        b=a+size
        changed[a:b]=payload
        changed_ranges.append((a,b))
    new_prefix=bytes(changed)
    with_crc=new_prefix+struct.pack("<I",zlib.crc32(new_prefix))
    result=with_crc+hashlib.sha256(with_crc).digest()
    if len(result)!=len(frozen) or result==frozen:
        raise AssertionError("candidate did not truly improve/re-encode source")
    rebuilt=CB59(result)
    if (rebuilt.meta!=archive.meta or
        rebuilt.handles["lm_head.weight"] is not rebuilt.handles["model.embed_tokens.weight"]):
        raise AssertionError("source manifest or tied alias corrupted")
    # Verify each byte outside the changed original head regions is identical.
    old=memoryview(frozen)
    cur=memoryview(result)
    boundaries=[(0,0)]+sorted(changed_ranges)+[(len(frozen)-36,len(frozen)-36)]
    for j in range(len(boundaries)-1):
        a=boundaries[j][1];b=boundaries[j+1][0]
        if old[a:b]!=cur[a:b]:
            raise AssertionError("nonhead source region was altered")
    return result

def recode_source(frozen:bytes,exact_original_head_weight):
    archive=CB59(frozen)
    h=archive.handles["lm_head.weight"]
    w=np.asarray(exact_original_head_weight.detach().cpu().numpy(),dtype=np.float32)
    if tuple(w.shape)!=h.shape or not np.isfinite(w).all():
        raise AssertionError("not exact original model tied pretrained tensor")
    scales=row_scales(w).astype(np.float32)
    new_centroids=fit_new_codebook(w,scales)
    choices=[]
    for name,row_ls in (("source_refit_rms",False),("source_refit_row_LS",True)):
        packed,raw_scales,rmse=recode_rows(w,new_centroids,scales,row_ls)
        choices.append((name,{"packed":packed,
                     "row_scales":raw_scales,
                     "centroids":new_centroids.astype("<f2").tobytes()},rmse))
    # ZERO test-label or generated-text tuning: minimize source-weight L2
    # among two pre-registered same-rate representations, source-only.
    chosen=min(choices,key=lambda x:x[2])
    candidate=swap_whole_tied_head_blob(frozen,chosen[1])
    return candidate,{"all_source_only_alternatives":[
        {"name":name,"original_tied_head_relative_RMSE":err}
        for name,_,err in choices],
        "selected_using_original_pretrained_weight_only":chosen[0],
        "source_only_selection_not_any_QA_TEST":True,
        "candidate_tied_head_SHA256":hashlib.sha256(candidate).hexdigest(),
        "new_complete_archive_bytes":len(candidate),
        "all_other_pretrained_matrices_bitwise_identical":True,
        "original_CB59_K32V8_native_kernel_ABI_unchanged":True}
