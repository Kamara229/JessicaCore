"""Trainable 4-layer byte-based independent discriminator.

Output one scalar: appropriateness of candidate for question. It is NOT
a fact generator or guarantee of factual judgment. Permits abstention if
all candidates are incoherent, instead of claiming false correctness.
"""
import torch
from torch import nn
from .protocol import BYTE_VOCAB,MAX_TEXT_BYTES,WIDTH,LAYERS,HEADS,FEED_FORWARD
from .bytes_adapter import PAD

class GreenTunnel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token=nn.Embedding(BYTE_VOCAB,WIDTH,padding_idx=PAD)
        self.position=nn.Embedding(MAX_TEXT_BYTES,WIDTH)
        layer=nn.TransformerEncoderLayer(
            d_model=WIDTH,nhead=HEADS,dim_feedforward=FEED_FORWARD,
            dropout=0.0,activation="gelu",batch_first=True,norm_first=True)
        self.blocks=nn.TransformerEncoder(
            layer,num_layers=LAYERS,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(WIDTH)
        self.score=nn.Linear(WIDTH,1)
    def forward(self,tokens):
        mask=tokens!=PAD
        pos=torch.arange(tokens.shape[1],device=tokens.device)
        z=self.token(tokens)+self.position(pos)[None,:,:]
        z=self.blocks(z,src_key_padding_mask=~mask)
        w=mask.unsqueeze(-1).to(z.dtype)
        pooled=(z*w).sum(1)/w.sum(1).clamp(min=1)
        return self.score(self.norm(pooled)).flatten()
    def disk_parameters(self):
        return sum(p.numel() for p in self.parameters())
