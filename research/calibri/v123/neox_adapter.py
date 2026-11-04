"""Read-only no-parameter GPT-NeoX interface for frozen 6D Green policy.

Different underlying model architecture, n_layers and hidden dimensions:
same Green weights. This scaling operator is a transfer diagnostic, NOT a
learned Pythia-specific quality correction or source compression proof.
"""
import re,torch
from torch import nn
from v122.policy import SharedGreenPolicy
from .protocol import STEERING_MAGNITUDE

PATTERN=re.compile(r"^gpt_neox\.layers\.(\d+)\.(attention\.dense|mlp\.dense_4h_to_h)$")

class OutputScale(nn.Module):
    def __init__(self,source,role,depth):
        super().__init__()
        self.source=source
        self.role=role
        self.depth=depth
        self.gain=1.0
    def forward(self,x):
        return self.source(x)*self.gain

class NeoXAdapter:
    def __init__(self,model):
        self.model=model
        self.nodes=[]
        self.originals={}
        count=int(model.config.num_hidden_layers)
        if count<2:raise AssertionError("invalid real NeoX depth")
        try:
            for name,layer in list(model.named_modules()):
                match=PATTERN.fullmatch(name)
                if not match:continue
                idx=int(match.group(1))
                if idx<0 or idx>=count:raise AssertionError("bad layer geometry")
                role=0 if match.group(2)=="attention.dense" else 1
                wrapper=OutputScale(layer,role,(idx+.5)/count)
                parent,_,leaf=name.rpartition(".")
                self.originals[name]=layer
                setattr(model.get_submodule(parent),leaf,wrapper)
                self.nodes.append(wrapper)
            if len(self.nodes)!=count*2:
                raise AssertionError("not a complete known GPTNeoX/attention role map")
        except Exception:
            self.restore();raise
    def set_question(self,policy,question):
        coeff=policy(question)
        gains=[]
        for node in self.nodes:
            alpha=policy.gates(coeff,node.depth,node.role)
            node.gain=1.0+STEERING_MAGNITUDE*(alpha-1.0)
            gains.append(float(node.gain))
        return {"number_of_host_operator_roles":len(self.nodes),
                "minimum_scale":min(gains),"maximum_scale":max(gains)}
    def neutral(self):
        for node in self.nodes:node.gain=1.
    def restore(self):
        for name,layer in self.originals.items():
            parent,_,leaf=name.rpartition(".")
            setattr(self.model.get_submodule(parent),leaf,layer)
