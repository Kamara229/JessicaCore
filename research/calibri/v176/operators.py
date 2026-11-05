"""Differentiable TRAIN surrogate, rounded to the real CB59 storage format.

Dense matrices exist during optimization only. Released inference uses the
unchanged packed C kernel. No latent dense matrix or optimizer is exported.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from direct_codebook_v59 import unpack5
from binary_geometry_v64 import MODEL, MODEL_SHA


def fp16_ste(x):
    rounded = x.to(torch.float16).to(torch.float32)
    return x + (rounded - x).detach()


class TrainWeight(nn.Module):
    def __init__(self, handle):
        super().__init__()
        self.shape = handle.shape
        n = handle.shape[0] * handle.shape[1] // 8
        ids = unpack5(handle.packed.tobytes(), n).astype(np.int64)
        self.register_buffer('codes', torch.from_numpy(ids))
        self.register_buffer('base_scales', torch.from_numpy(handle.row_scales.copy()))
        self.centroids = nn.Parameter(torch.from_numpy(handle.centroids.copy()))
        self.log_scales = nn.Parameter(torch.zeros(handle.shape[0]))
        self.cached = None

    def fields(self):
        return fp16_ste(self.centroids), fp16_ste(self.base_scales * self.log_scales.exp())

    def forward(self):
        if self.cached is None:
            centers, scales = self.fields()
            self.cached = centers[self.codes].reshape(self.shape) * scales[:, None]
        return self.cached

    def clear(self):
        self.cached = None


class TrainLinear(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.out_features, self.in_features = source.shape
        self.bias = None

    def forward(self, x):
        return F.linear(x, self.source())


class TrainEmbedding(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source
        self.num_embeddings, self.embedding_dim = source.shape

    def forward(self, ids):
        return F.embedding(ids, self.source())


def build(archive):
    cfg = AutoConfig.from_pretrained(MODEL, revision=MODEL_SHA)
    cfg._attn_implementation = 'eager'
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32).eval()
    if set(model.state_dict()) != set(archive.meta['full_original_state_keys']):
        raise ValueError('Original model state/config mismatch')
    sources = {}
    descriptors = archive.meta['tensor_descriptors']
    for name, d in descriptors.items():
        if d['dtype'] == 'CB59_K32V8':
            sources[name] = TrainWeight(archive.handles[name])
    aliases = archive.meta['source_alias_keys']
    for path in archive.handles:
        canonical = aliases.get(path, path)
        source = sources[canonical]
        module_path = path.removesuffix('.weight')
        parent, _, leaf = module_path.rpartition('.')
        owner = model.get_submodule(parent) if parent else model
        previous = getattr(owner, leaf)
        if isinstance(previous, nn.Embedding):
            replacement = TrainEmbedding(source)
        elif isinstance(previous, nn.Linear) and previous.bias is None:
            replacement = TrainLinear(source)
        else:
            raise ValueError('Unsupported original operator ' + module_path)
        setattr(owner, leaf, replacement)
    for name, a in archive.raw.items():
        parent, _, leaf = name.rpartition('.')
        parameter = getattr(model.get_submodule(parent), leaf)
        with torch.no_grad():
            parameter.copy_(torch.from_numpy(np.array(a, copy=True)))
    if model.model.embed_tokens.source is not model.lm_head.source:
        raise AssertionError('Tied TRAIN embedding/head no longer shared')
    # No optimizer-visible dense original matrix may survive replacement.
    if any(p.ndim >= 2 and tuple(p.shape) != (32, 8) for p in model.parameters()):
        raise AssertionError('Unexpected trainable dense weight')
    return model, sources


def clear(sources):
    for source in sources.values():
        source.clear()

