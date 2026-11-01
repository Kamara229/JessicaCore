"""CALIBRI v13 research: serialized INT4/INT8 plus low-rank and sparse corrections.

Research-only. Exact accounted byte-stream, held-out activation / LM checks; no fused
inference or claimed novel primitive. Independent of existing production paths.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import struct
import zlib
import numpy as np
from scipy.linalg import svd

HEADER = struct.Struct('<4sHHBBII')  # 18 bytes: magic, rows, cols, bits, rank, k, crc32


def _quant(a, bits):
    a = np.asarray(a, dtype=np.float32)
    bound = 7 if bits == 4 else 127
    scale = (np.max(np.abs(a), axis=1) / bound).astype('<f2')
    scale[scale == 0] = np.float16(1.0)
    q = np.clip(np.rint(a / scale.astype(np.float32)[:, None]), -bound, bound).astype(np.int8)
    return q, scale


def _pack_q(q, bits):
    vals = np.asarray(q, dtype=np.int8).ravel()
    if bits == 8:
        return vals.tobytes()
    nib = (vals.astype(np.int16) & 15).astype(np.uint8)
    if len(nib) % 2:
        nib = np.r_[nib, np.uint8(0)]
    return (nib[::2] | (nib[1::2] << 4)).tobytes()


def _unpack_q(data, count, bits):
    if bits == 8:
        if len(data) != count:
            raise ValueError('bad INT8 length')
        return np.frombuffer(data, np.int8).astype(np.float32)
    if len(data) != (count + 1) // 2:
        raise ValueError('bad INT4 length')
    inp = np.frombuffer(data, np.uint8)
    nib = np.empty(len(inp) * 2, dtype=np.uint8)
    nib[::2] = inp & 15
    nib[1::2] = inp >> 4
    nib = nib[:count].astype(np.int16)
    return np.where(nib >= 8, nib - 16, nib).astype(np.float32)


def _lr(a, rank):
    if rank == 0:
        return (np.empty((a.shape[0], 0), np.float32),
                np.empty((0, a.shape[1]), np.float32))
    u, s, vh = svd(a, full_matrices=False, lapack_driver='gesdd')
    root = np.sqrt(s[:rank])
    return ((u[:, :rank] * root).astype('<f2').astype(np.float32),
            (root[:, None] * vh[:rank, :]).astype('<f2').astype(np.float32))


def encode(w, x_cal, bits=4, rank=0, sparse_fraction=0., order='post'):
    """Return physical immutable bytes; sparse corrections computed after FP16 rounding."""
    w = np.asarray(w, np.float32)
    x_cal = np.asarray(x_cal, np.float32)
    if (w.ndim != 2 or x_cal.ndim != 2 or x_cal.shape[1] != w.shape[1]
            or not 0 <= rank <= min(w.shape) or rank > 255
            or bits not in (4, 8) or order not in ('post', 'pre')
            or not (0 <= sparse_fraction <= 0.2) or w.size >= 2**32
            or max(w.shape) > 65535 or not np.isfinite(w).all()
            or not np.isfinite(x_cal).all() or len(x_cal) == 0):
        raise ValueError('invalid codec configuration')
    m, n = w.shape
    if order == 'pre' and rank:
        u, v = _lr(w, rank)
        base = w - u @ v
        q, s = _quant(base, bits)
    else:
        q, s = _quant(w, bits)
        base = q.astype(np.float32) * s.astype(np.float32)[:, None]
        u, v = _lr(w - base, rank)
    core = q.astype(np.float32) * s.astype(np.float32)[:, None] + u @ v
    k = int(w.size * sparse_fraction)
    if k:
        rms = np.sqrt(np.mean(x_cal.astype(np.float64)**2, axis=0))
        score = np.abs(w - core) * rms[None, :]
        ix = np.sort(np.argpartition(score.ravel(), -k)[-k:]).astype('<u4')
        values = (w - core).ravel()[ix].astype('<f2')
    else:
        ix = np.empty((0,), '<u4')
        values = np.empty((0,), '<f2')
    payload = (_pack_q(q, bits) + s.astype('<f2').tobytes()
               + u.astype('<f2').tobytes() + v.astype('<f2').tobytes()
               + ix.tobytes() + values.tobytes())
    return HEADER.pack(b'CF13', m, n, bits, rank, k, zlib.crc32(payload)) + payload


def decode(blob):
    if len(blob) < HEADER.size:
        raise ValueError('truncated')
    magic, m, n, bits, rank, k, checksum = HEADER.unpack_from(blob)
    if magic != b'CF13' or bits not in (4, 8) or not m or not n or rank > min(m, n) or k > m*n:
        raise ValueError('invalid header')
    qsize = (m*n + 1)//2 if bits == 4 else m*n
    expected = qsize + m*2 + 2*rank*(m+n) + 6*k
    data = blob[HEADER.size:]
    if len(data) != expected or zlib.crc32(data) != checksum:
        raise ValueError('wrong length or checksum')
    q = _unpack_q(data[:qsize], m*n, bits).reshape(m, n)
    p = qsize
    scales = np.frombuffer(data, '<f2', count=m, offset=p).astype(np.float32); p += 2*m
    u = np.frombuffer(data, '<f2', count=m*rank, offset=p).astype(np.float32).reshape(m, rank); p += 2*m*rank
    v = np.frombuffer(data, '<f2', count=n*rank, offset=p).astype(np.float32).reshape(rank, n); p += 2*n*rank
    ix = np.frombuffer(data, '<u4', count=k, offset=p); p += 4*k
    values = np.frombuffer(data, '<f2', count=k, offset=p).astype(np.float32)
    if (not np.isfinite(scales).all() or not np.isfinite(u).all() or
            not np.isfinite(v).all() or not np.isfinite(values).all() or
            (k and (ix[-1] >= m*n or np.any(np.diff(ix.astype(np.int64)) <= 0)))):
        raise ValueError('invalid factors or sparse index')
    result = q * scales[:, None] + u @ v
    result.ravel()[ix] += values
    return result


def candidates(w, x_cal):
    configs = [('q4',4,0,0.,'post'),('q8',8,0,0.,'post'),
               ('q4_sparse1',4,0,.01,'post'),('q4_sparse3',4,0,.03,'post'),
               ('q4_rank8',4,8,0.,'post'),('q4_rank16',4,16,0.,'post'),
               ('q4_rank8_sparse1',4,8,.01,'post'),
               ('q4_rank16_sparse1',4,16,.01,'post'),
               ('rank8_q4',4,8,0.,'pre'),
               ('rank8_q4_sparse1',4,8,.01,'pre')]
    output = []
    reference = x_cal @ w.T
    denom = np.sum(reference.astype(np.float64)**2)
    for name,bits,r,sparse,order in configs:
        blob=encode(w,x_cal,bits,r,sparse,order)
        recon=decode(blob)
        error=float(np.sum((x_cal @ (w-recon).T).astype(np.float64)**2) / max(denom,1e-30))
        output.append({'name':name, 'blob':blob,'recon':recon,
                       'bytes':len(blob),'cal_rel_mse':error})
    return output


def pareto(items):
    return [a['name'] for a in items if not any(
        b is not a and b['bytes'] <= a['bytes'] and b['cal_rel_mse'] <= a['cal_rel_mse']
        and (b['bytes'] < a['bytes'] or b['cal_rel_mse'] < a['cal_rel_mse']) for b in items)]


def run(args):
    # Genuine public pretrained weights and captured activations; no synthetic fallback.
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import HfApi
    from quality import load_token_ids, ppl_eval, calibrate, module_selection, MODEL, DATASET
    torch.set_num_threads(2)
    torch.manual_seed(13)
    revision=HfApi().model_info(MODEL).sha
    d_rev=HfApi().dataset_info(DATASET).sha
    tok=AutoTokenizer.from_pretrained(MODEL, revision=revision)
    model=AutoModelForCausalLM.from_pretrained(MODEL, revision=revision, torch_dtype=torch.float32)
    model.eval()
    modules=module_selection(model)
    target_name=sorted(modules)[0]
    module=modules[target_name]
    w=module.weight.detach().cpu().numpy().copy().astype(np.float32)
    if min(w.shape) < 16: raise ValueError('insufficient shape for rank16')
    ids_train=load_token_ids(tok,'train',d_rev, args.calibration_tokens)
    ids_test=load_token_ids(tok,'test',d_rev, args.test_tokens)
    x=calibrate(model,{target_name:module},ids_train,args.sequence_length)[target_name].numpy()
    all_items=candidates(w,x)
    q4=next(c for c in all_items if c['name']=='q4')
    byte_budget=int(q4['bytes']*args.relative_budget)
    eligible=[c for c in all_items if c['bytes'] <= byte_budget]
    selected=min(eligible,key=lambda c:(c['cal_rel_mse'],c['bytes']))
    orig=module.weight.detach().clone()
    original_ppl=ppl_eval(model,ids_test,args.sequence_length)
    chosen=list(dict.fromkeys(['q4','q8',selected['name']]))
    ppl={}
    for name in chosen:
        r=next(c for c in all_items if c['name']==name)
        with torch.no_grad():module.weight.copy_(torch.from_numpy(r['recon']))
        ppl[name]=ppl_eval(model,ids_test,args.sequence_length)
    with torch.no_grad():module.weight.copy_(orig)
    xhash=hashlib.sha256(ids_train.numpy().tobytes()).hexdigest()
    thash=hashlib.sha256(ids_test.numpy().tobytes()).hexdigest()
    for c in all_items:
        del c['blob'],c['recon']
    report={'experiment':'CALIBRI_v13_fusion_pilot_real_Pythia14m',
            'model':MODEL,'model_revision':revision,'dataset':DATASET,
            'dataset_revision':d_rev,'cal_train_token_sha256':xhash,
            'heldout_test_token_sha256':thash,'target_module':target_name,
            'shape':list(w.shape),'weights':int(w.size),'calibration_activations':int(len(x)),
            'baseline_fp16_weight_bytes':int(2*w.size),
            'full_model_test_ppl_original':original_ppl,'variants':all_items,
            'calibration_pareto':pareto(all_items),
            'relative_byte_budget_to_q4':args.relative_budget,
            'byte_budget':byte_budget,'selected_using_cal_only':selected['name'],
            'full_model_heldout_test_ppl':ppl,
            'limitations':['One real pretrained Pythia-14m MLP matrix, not MoE/QMoE',
              'No full-model byte budget or materialized fused inference',
              'Dense restored matmul used for quality; speed not measured',
              'Calibrate on train, test on independent WikiText test split',
              'Low-rank factors and sparse patch bytes included; no training of model',
              'Candidate search small, not globally optimal','Quality is PPL after replacing one matrix only']}
    output=pathlib.Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False),flush=True)


def selftest():
    rng=np.random.default_rng(13)
    w=rng.normal(size=(33, 17)).astype(np.float32)
    x=rng.normal(size=(19, 17)).astype(np.float32)
    for b in (4,8):
        for r in (0,8):
            for order in ('post','pre'):
                for frac in (0.,.03):
                    data=encode(w,x,b,r,frac,order)
                    rec=decode(data)
                    assert rec.shape==w.shape and np.isfinite(rec).all()
                    assert len(data)==HEADER.size+(w.size+1)//2*(b==4)+w.size*(b==8)+2*len(w)+2*r*sum(w.shape)+6*int(w.size*frac)
                    broken=bytearray(data);broken[-1]^=1
                    try:decode(broken);raise AssertionError('corruption accepted')
                    except ValueError:pass
    options=candidates(w,x)
    assert 'q4' in pareto(options)
    print('v13 selftest: 16 physical roundtrips, CRC corruption checks, lengths and Pareto PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--output',default='results/calibri_v13_real_fusion.json')
    parser.add_argument('--calibration-tokens',type=int,default=256)
    parser.add_argument('--test-tokens',type=int,default=1024)
    parser.add_argument('--sequence-length',type=int,default=128)
    parser.add_argument('--relative-budget',type=float,default=1.5)
    a=parser.parse_args()
    if a.selftest:selftest()
    else:run(a)
