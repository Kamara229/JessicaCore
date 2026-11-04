"""Bind native residual corrections for one experiment and fully restore them on exit."""
from contextlib import contextmanager, ExitStack
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v118.runtime import Head
from v119.learn import RankHead
from v122.steer import SteeringBlock
from .protocol import layout


class TiedInput(nn.Module):
    def __init__(self, original, head):
        super().__init__()
        self.original, self.correction = original, head

    def forward(self, ids):
        return self.original(ids) + F.embedding(ids, self.correction.u) @ self.correction.v


class Bridge:
    def __init__(self, blocks, head):
        self.blocks, self.head = blocks, head

    def control(self, policy, question):
        with torch.no_grad():
            coeff = policy(question) if policy is not None else None
            for name, layer in self.blocks.items():
                group = 0 if ".self_attn." in name else 1
                depth = (int(name.split(".")[2]) + 0.5) / 30
                layer.gain = 1.0 if coeff is None else policy.gates(coeff, depth, group)

    def factors(self):
        def fp16(p):
            return p.detach().cpu().numpy().astype("<f2")
        h = self.head
        return ({n: (fp16(b.u), fp16(b.v)) for n, b in self.blocks.items()},
                (fp16(h.u), fp16(h.v), fp16(h.b)))


@contextmanager
def installed(model, source, moved, *, training=False, tie_input=True):
    blocks = {}
    with ExitStack() as stack:
        def replace(name, current, new):
            parent, _, field = name.rpartition(".")
            owner = model.get_submodule(parent) if parent else model
            stack.callback(setattr, owner, field, current)
            setattr(owner, field, new)

        old_head = model.lm_head
        head = (RankHead if training else Head)(old_head, *source.head())
        replace("lm_head", old_head, head)
        if moved and tie_input:
            old_input = model.get_input_embeddings()
            replace("model.embed_tokens", old_input, TiedInput(old_input, head))
        for name, rows, cols, rank in layout(moved):
            layer = model.get_submodule(name)
            native = getattr(layer, "original", layer)
            if (native.out_features, native.in_features) != (rows, cols):
                raise ValueError(f"native geometry mismatch at {name}")
            if training:
                # Both placements start with zero body correction and the same
                # frozen v119 head initialization. Nonzero V enables the first U update.
                u = np.zeros((rows, rank), np.float32)
                v = (torch.randn(rank, cols) * .01).numpy()
            else:
                u, v = source.role(name)
            block = SteeringBlock(layer, u, v)
            if training:
                block.u, block.v = nn.Parameter(block.u), nn.Parameter(block.v)
                block.zero = False
            replace(name, layer, block)
            blocks[name] = block
        yield Bridge(blocks, head)
