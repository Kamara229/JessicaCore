"""Model-agnostic 6-coefficient gate policy with shared <=10MB neural weights.

Same output shape for a host of 30, 300 or 3000 layers. Layer-depth
normalization (depth/L) is the transferable actuator interface; trained
source-specific U/V residuals are NOT assumed universal.
"""
import hashlib
import math
import torch
from torch import nn
from safetensors.torch import load_file,save_file
from v120.critic import GreenTunnel
from v120.bytes_adapter import pairs
from .protocol import MAX_DELTA,GREEN_SOURCE_SHA,MAX_CHECKPOINT_BYTES

class SharedGreenPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network=GreenTunnel()
        self.network.score=nn.Linear(224,6)
        with torch.no_grad():
            nn.init.zeros_(self.network.score.weight)
            nn.init.zeros_(self.network.score.bias)
    def forward(self,question):
        # Prompt only: GOLD answer is NEVER fed into the controller.
        data=pairs([(question,"")])
        return self.network(data).reshape(6)
    def gates(self,coeff,relative_depth,role):
        if role not in (0,1):
            raise ValueError("role must be attention(0) or MLP(1)")
        if not 0<=relative_depth<=1:
            raise ValueError("depth must be normalized and model-independent")
        pos=2*relative_depth-1
        z=coeff[role*3:role*3+3]
        return 1.0+MAX_DELTA*torch.tanh(z[0]+z[1]*pos+z[2]*pos*pos)

def initialize(previous_source):
    raw=previous_source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=GREEN_SOURCE_SHA:
        raise ValueError("original 8.22MB Green Tunnel not SHA-identical")
    state=load_file(str(previous_source))
    original=GreenTunnel()
    original.load_state_dict(state,strict=True)
    policy=SharedGreenPolicy()
    # Only final task head is replaced; all 8MB pretrained critic parameters
    # are retained as a starting point, not counted as a free 10MB.
    weight=dict(original.state_dict())
    weight.pop("score.weight");weight.pop("score.bias")
    current=policy.network.state_dict()
    current.update(weight)
    policy.network.load_state_dict(current,strict=True)
    # Freeze lower representation to reduce overfitting on 16 tasks.
    for p in policy.parameters():p.requires_grad_(False)
    for p in policy.network.blocks.layers[-1].parameters():p.requires_grad_(True)
    for p in policy.network.score.parameters():p.requires_grad_(True)
    return policy

def serialize(policy,path):
    state={k:v.detach().cpu().contiguous()
           for k,v in policy.state_dict().items()}
    save_file(state,str(path))
    size=path.stat().st_size
    if size>MAX_CHECKPOINT_BYTES:
        raise AssertionError("shared neural weights exceeded standalone 10MB")
    return {"green_tunnel_full_neural_weight_bytes":size,
            "neural_model_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "fixed_parameter_count":sum(p.numel() for p in policy.parameters()),
            "neural_head_outputs":6,
            "independent_of_host_layer_count":True}
