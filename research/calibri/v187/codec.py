"""True four-axis weight grouping: (output_row_pair, input_col_block, 2,4).

Each 2x4 block's EIGHT original pretrained values share one 5-bit
K32 code. This is not a stylized concept: each index is physically
packed into precisely 0.625 bits/original scalar, same rate as CB59's
1x8 groups. The codebook is still 32x8 FP16, and source row gains are
exactly the same original FP16 per-output-row normalization.

Only a selected set of authentic pretrained tensors is compressed in
this diagnostic. This is NOT yet a full 134.5M model source and
cannot be described as a valid 0.7bpp pretrained runnable model.
"""
from dataclasses import dataclass
import math
import numpy as np
from direct_codebook_v59 import pack5,unpack5,row_scales,_centroids,_nearest_integer_128
from . import protocol as p


@dataclass
class Tile4D:
    name: str
    shape: tuple
    packed: bytes
    row_scales: np.ndarray
    centroids: np.ndarray

    def validate(self):
        m,n=self.shape
        if m%2 or n%4 or m<=0 or n<=0:
            raise ValueError("4D source needs even rows and ncols divisible by 4")
        ng=(m//2)*(n//4)
        if len(self.packed)!=(5*ng+7)//8 or (ng%8):
            raise ValueError("5-bit tile stream not correctly byte-aligned")
        if (self.row_scales.shape!=(m,) or
                self.centroids.shape!=(p.K,p.VALUES_PER_TILE)):
            raise ValueError("Wrong complete original row-gain or centroid geometry")
        if not np.isfinite(self.row_scales).all() or np.any(self.row_scales<=0):
            raise ValueError("4D row-gain invalid")
        if not np.isfinite(self.centroids).all():
            raise ValueError("4D codebook invalid")
        return self

    def payload_bytes(self):
        self.validate()
        return len(self.packed)+len(self.row_scales.astype("<f2").tobytes())+len(self.centroids.astype("<f2").tobytes())


def _vectors(w,scales):
    m,n=w.shape
    # [m//2, 2, n//4, 4] => [m//2, n//4, 2, 4]
    blocks=w.reshape(m//2,2,n//4,4).transpose(0,2,1,3)
    scale=scales.astype(np.float32).reshape(m//2,2)
    normalized=blocks/scale[:,None,:,None]
    return normalized.reshape(-1,8).astype(np.float32)


def _rebuild(groups,shape):
    m,n=shape
    blocks=np.asarray(groups,dtype=np.float32).reshape(m//2,n//4,2,4)
    return blocks.transpose(0,2,1,3).reshape(m,n)


def quantize(name,full,blockpairs=p.SOURCE_BATCH):
    w=np.asarray(full,dtype=np.float32)
    if w.ndim!=2 or w.shape[0]%2 or w.shape[1]%4 or not np.isfinite(w).all():
        raise ValueError("Unexpected original multi-axis pretrained tensor")
    m,n=w.shape
    scales=row_scales(w)
    pairs=m//2
    ng=n//4
    if pairs*ng%8:
        raise ValueError("Pretrained source 5-bit groups must end on 5-byte boundary")
    chosen=np.linspace(0,pairs*ng-1,min(4096,pairs*ng),dtype=np.int64)
    pair_index=chosen//ng
    group_index=chosen%ng
    base=np.empty((len(chosen),8),dtype=np.float32)
    for j in range(2):
        src=w[2*pair_index+j]
        base[:,j*4:(j+1)*4]=np.take_along_axis(
            src,group_index[:,None]*4+np.arange(4)[None,:],axis=1
        )/scales[2*pair_index+j,None].astype(np.float32)
    centers=_centroids(base).astype("<f2")
    if centers.shape!=(32,8):
        raise AssertionError("4D codebook rows invalid")
    parts=[]
    diff_sum=source_sum=0.0
    for start in range(0,pairs,blockpairs):
        end=min(start+blockpairs,pairs)
        current=w[start*2:end*2]
        current_scale=scales[start*2:end*2]
        group=_vectors(current,current_scale).astype("<f2").astype(np.float32)
        ids=_nearest_integer_128(group,centers.astype(np.float32))
        # Individual encoded chunks must fall on index-group byte boundary.
        if ids.size%8:
            raise AssertionError("4D source code count needs 8-index alignment")
        parts.append(pack5(ids.astype(np.uint8)))
        decoded=_rebuild(centers.astype(np.float32)[ids],current.shape)
        decoded*=current_scale.astype(np.float32)[:,None]
        delta=current.astype(np.float64)-decoded.astype(np.float64)
        diff_sum+=float(np.sum(delta**2))
        source_sum+=float(np.sum(current.astype(np.float64)**2))
    physical=Tile4D(name,(m,n),b"".join(parts),
            scales.astype("<f2").astype(np.float32),
            centers.astype("<f2").astype(np.float32)).validate()
    if not source_sum>0:raise AssertionError("Zero-only source")
    return physical,{"source_relative_weight_L2":math.sqrt(diff_sum/source_sum),
        "source_squared_error":diff_sum,
        "all_original_source_weight_energy":source_sum,
        "tile_count":pairs*ng,
        "complete_matrix_payload_bytes":physical.payload_bytes(),
        "physical_5bit_indices_bytes":len(physical.packed),
        "centroids_fp16_bytes":len(centers.tobytes()),
        "scales_fp16_bytes":len(scales.tobytes()),
        "tensor_full_checkpoint_NOT_yet_built":True}


def reconstruct(handle):
    handle.validate()
    ng=(handle.shape[0]//2)*(handle.shape[1]//4)
    ids=unpack5(handle.packed,ng)
    vals=handle.centroids[ids]
    out=_rebuild(vals,handle.shape)
    out*=handle.row_scales[:,None].astype(np.float32)
    return np.ascontiguousarray(out,dtype=np.float32)
