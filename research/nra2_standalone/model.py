"""Independent NRA 2.0 v0: neural relational composition, NOT a Calibri hook.

A deliberately narrow first specialization: decide whether node A precedes node B
in an acyclic directed chain represented as edges. GRU message passing is fully
trainable. It consumes only graph tensors and endpoint indicators. It never sees
answer tokens, labels, GitHub QA, or Calibri activations during inference.
"""

import torch
from torch import nn


class RelationalNRA(nn.Module):
    def __init__(self, width: int = 16, hops: int = 5):
        super().__init__()
        if width < 4 or hops < 1:
            raise ValueError("Invalid standalone NRA width or propagation depth")
        self.width = width
        self.hops = hops
        self.target_encoder = nn.Linear(1, width)
        self.message = nn.Linear(width, width, bias=False)
        self.update = nn.GRUCell(width, width)
        self.answer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width),
            nn.Tanh(), nn.Linear(width, 1),
        )

    def forward(self, edges: torch.Tensor, source: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """Return binary logits; no ground truth or answer lookup is accepted."""
        if (edges.ndim != 3 or edges.shape[1] != edges.shape[2]
                or source.shape != edges.shape[:2]
                or target.shape != source.shape):
            raise ValueError("Expected edges [B,N,N] and source,target [B,N]")
        if not (torch.isfinite(edges).all() and torch.isfinite(source).all()
                and torch.isfinite(target).all()):
            raise ValueError("Non-finite relational input")
        batch, count = source.shape
        state = self.target_encoder(target.unsqueeze(-1))
        for _ in range(self.hops):
            # A[i,j]=1 means i is left of j. Information flows j -> i.
            incoming = torch.bmm(edges, self.message(state))
            state = self.update(
                incoming.reshape(-1, self.width),
                state.reshape(-1, self.width),
            ).reshape(batch, count, self.width)
        selected = (state * source.unsqueeze(-1)).sum(dim=1)
        return self.answer(selected).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
