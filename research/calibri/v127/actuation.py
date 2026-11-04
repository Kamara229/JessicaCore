"""Fixed-bound role controls; nothing saved back into packed trained weights."""
import torch
from .protocol import MAX_DELTA
from v122.steer import Installed

class Actuator:
    def __init__(self,bridge):
        if not isinstance(bridge,Installed) or len(bridge.maps)!=60:
            raise AssertionError("expected exactly sixty previously paid structural hooks")
        self.bridge=bridge
    def neutral(self):
        for mod,role,depth in self.bridge.maps:
            mod.gain=torch.tensor(1.)
    def oracle(self,parameters,mode):
        if mode=="individual_sixty_oracle":
            if tuple(parameters.shape)!=(60,):raise ValueError("sixty-operator oracle required")
            gains=1.+MAX_DELTA*torch.tanh(parameters)
        elif mode=="shared_six_oracle":
            if tuple(parameters.shape)!=(6,):raise ValueError("size-invariant six-dimensional oracle required")
            vals=[]
            for module,role,depth in self.bridge.maps:
                p=2.*depth-1.
                z=parameters[3*role:3*role+3]
                vals.append(1.+MAX_DELTA*torch.tanh(z[0]+p*z[1]+(p*p)*z[2]))
            gains=torch.stack(vals)
        else:
            raise ValueError("unknown prespecified action mode")
        for (module,role,depth),g in zip(self.bridge.maps,gains):
            module.gain=g
        if (gains.min().item()<.3-1e-5 or gains.max().item()>1.7+1e-5):
            raise ValueError("unsafe control outside preregistered bounds")
        return gains
    def frozen_shared(self,policy,question):
        return self.bridge.set_question(policy,question)
