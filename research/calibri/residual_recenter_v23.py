"""CALIBRI v23: controlled physical INT5 residual re-centering after TT training.

Ablation on a real pretrained Pythia-14m 128x512 projection. We reuse the
v22 frozen-teacher trainer, not a new optimizer, and change ONLY whether the
residual is repacked around trained physical FP16 TT cores. No app changes.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import pathlib
import struct
import zlib

import numpy as np
from fixedbit_v15 import decode as decode_int, encode as encode_int
from geometric_training_v22 import make_teacher_windows, train_method
from geometric_v16 import reconstruct, unpack as unpack_tt
from geometric_v17 import encode as geo_encode, native_layer
from packed_native_v19 import split_geometry
from rowcache_v20 import row_cached_geometry

BUNDLE = struct.Struct('<4sIII')


def recenter_physical(blob: bytes, original_w: np.ndarray) -> bytes:
    """Keep FP16 TT cores; recompute a *physical*, packed INT5 quantized residual.

    Re-centering is a weight-aware encoder operation, not a training gradient.
    Output is independently page-readable just like v22; no full W is needed
    at inference. Weight-sized temporary arrays are encoder-only.
    """
    tt_bytes, old_int5 = split_geometry(blob)
    a, b, _ = unpack_tt(tt_bytes)
    new_int5 = encode_int(original_w - reconstruct(a, b), 5)
    if len(old_int5) != len(new_int5):
        raise AssertionError('recenter changed fixed-bit byte budget')
    payload = tt_bytes + new_int5
    result = BUNDLE.pack(b'CG17', len(tt_bytes), len(new_int5), zlib.crc32(payload)) + payload
    if len(result) != len(blob):
        raise AssertionError('recenter changed total serialized bytes')
    return result


def weight_relative_rmse(blob: bytes, original_w: np.ndarray) -> float:
    tt_bytes, residual_bytes = split_geometry(blob)
    a, b, _ = unpack_tt(tt_bytes)
    recovered = reconstruct(a, b) + decode_int(residual_bytes)
    return float(np.linalg.norm(original_w - recovered) /
                 max(np.linalg.norm(original_w), 1e-20))


def shadow_kl(model, windows) -> float:
    import torch
    import torch.nn.functional as F
    values = []
    with torch.no_grad():
        for item in windows:
            output = model(input_ids=item['ids'], use_cache=False).logits
            target = item['logits'].float()
            temperature = 2.0
            kl = F.kl_div(F.log_softmax(output / temperature, dim=-1),
                          F.softmax(target / temperature, dim=-1),
                          reduction='sum') * temperature**2 / output.shape[1]
            values.append(float(kl))
    return float(np.mean(values))


def score(model, outer, leaf, orig, blob, library, w, dev, tiny, shadow, context):
    import torch
    from quality import ppl_eval
    layer = orig if blob is None else row_cached_geometry(blob, orig.bias, library)
    setattr(outer, leaf, layer)
    try:
        with torch.no_grad():
            dev_ppl = ppl_eval(model, dev, context)['ppl']
            tiny_ppl = ppl_eval(model, tiny, context)['ppl']
            kl = shadow_kl(model, shadow)
            x = shadow[0]['input'].float()
            expected = shadow[0]['output'].float()
            local_rmse = float(torch.linalg.vector_norm(layer(x) - expected) /
                               torch.linalg.vector_norm(expected).clamp_min(1e-20))
            if blob is not None:
                unpacked = native_layer(blob, orig.bias)
                packed_vs_dense_max = float((layer(x) - unpacked(x)).abs().max())
                if packed_vs_dense_max > 0.001:
                    raise AssertionError('packed INT5 kernel diverged from dense decoder')
            else:
                packed_vs_dense_max = None
    finally:
        setattr(outer, leaf, orig)
    return {
        'physical_weight_bytes': len(blob) if blob is not None else 2*w.size,
        'wiki_dev_ppl': dev_ppl,
        'tinystories_dev_ppl': tiny_ppl,
        'unused_train_shadow_kl': kl,
        'unused_train_shadow_local_rel_rmse': local_rmse,
        'packed_vs_dense_max_abs': packed_vs_dense_max,
        'weight_rel_rmse': weight_relative_rmse(blob, w) if blob is not None else 0.,
    }


def run(args):
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from quality import DATASET, MODEL, load_token_ids, module_selection
    from geometric_v18 import load_new_corpus

    if args.train_tokens % args.context or args.shadow_tokens % args.context:
        raise ValueError('token counts must divide context exactly')
    torch.manual_seed(23)
    torch.set_num_threads(2)
    library = ct.CDLL(str(pathlib.Path(args.library).resolve()))
    api = HfApi()
    model_rev = api.model_info(MODEL).sha
    wiki_rev = api.dataset_info(DATASET).sha
    tiny_rev = api.dataset_info('roneneldan/TinyStories').sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=model_rev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=model_rev,
                                                torch_dtype=torch.float32)
    model.eval()
    name = sorted(module_selection(model))[0]
    original = model.get_submodule(name)
    parent_name, leaf = name.rsplit('.', 1)
    outer = model.get_submodule(parent_name)
    w = original.weight.detach().cpu().numpy().copy().astype(np.float32)
    base_blob = geo_encode(w, 2, 16, 32, 5)
    if recenter_physical(base_blob, w) != base_blob:
        raise AssertionError('no-op physical re-centering changed original weights')
    whole_train = load_token_ids(tokenizer, 'train', wiki_rev,
                                 args.train_tokens + args.shadow_tokens)
    train_ids, shadow_ids = whole_train[:args.train_tokens], whole_train[args.train_tokens:]
    dev_ids = load_token_ids(tokenizer, 'validation', wiki_rev, args.dev_tokens)
    tiny_ids, _ = load_new_corpus(tokenizer, args.tiny_tokens, tiny_rev)
    train_windows = make_teacher_windows(model, original, train_ids, args.context)
    shadow_windows = make_teacher_windows(model, original, shadow_ids, args.context)

    candidates = {'original': None, 'untrained_TT_INT5': base_blob}
    traces = {}
    for method in ('kl_only', 'kl_plus_local'):
        trained, traces[method] = train_method(model, outer, leaf, original, base_blob,
                                               library, train_windows, args.steps, args.lr, method)
        candidates[f'trained_fixed_residual_{method}'] = trained
        candidates[f'trained_recentered_residual_{method}'] = recenter_physical(trained, w)
    scores = {}
    for label, blob in candidates.items():
        scores[label] = score(model, outer, leaf, original, blob, library,
                              w, dev_ids, tiny_ids, shadow_windows, args.context)
        print(json.dumps({'candidate': label, **scores[label]}), flush=True)
    report = {
        'experiment': 'CALIBRI_v23_fixed_vs_recentered_INT5_after_TT_teacher_training',
        'model': MODEL, 'model_revision': model_rev, 'module': name,
        'matrix_shape': list(w.shape), 'original_fp16_bytes': 2*w.size,
        'wikitext_revision': wiki_rev, 'tinystories_revision': tiny_rev,
        'train_tokens': len(train_ids), 'shadow_tokens': len(shadow_ids),
        'dev_tokens': len(dev_ids), 'tinystories_tokens': len(tiny_ids),
        'tokens_sha256': {key: hashlib.sha256(ids.numpy().tobytes()).hexdigest()
                          for key, ids in (('train', train_ids), ('shadow', shadow_ids),
                                           ('wiki_dev', dev_ids), ('tiny_dev', tiny_ids))},
        'steps': args.steps, 'lr': args.lr, 'teacher_training': traces,
        'candidates': scores,
        'constraints': [
            'One REAL pretrained 128x512 Pythia-14m dense projection; not Switch MoE or QMoE',
            'Trainable weights are only 1280 physical TT core numbers; core size unchanged',
            'v23 changes the RESIDUAL AFTER training, using original pretrained W and NO eval data',
            'Residual is still a genuine packed INT5 byte stream executed by C, not materialized at inference',
            'WikiText validation and TinyStories validation were already inspected in earlier sessions',
            'Train-shadow tokens are disjoint from optimization but are NOT a fresh corpus',
            'No isolated independent test, generalization, speed or architecture novelty claim',
            'No production branch, Render, Supabase or Android changes',
        ],
    }
    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('CALIBRI v23 physically checked full-model diagnostic COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--library', default='research/calibri/libpacked5_v23.so')
    p.add_argument('--output', default='results/calibri_v23_recenter.json')
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.0005)
    p.add_argument('--train-tokens', type=int, default=2048)
    p.add_argument('--shadow-tokens', type=int, default=512)
    p.add_argument('--dev-tokens', type=int, default=1024)
    p.add_argument('--tiny-tokens', type=int, default=2048)
    p.add_argument('--context', type=int, default=128)
    run(p.parse_args())
