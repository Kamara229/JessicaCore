"""V006 GPU table path: active codebook only, versioned gather indices.
No dense weight matrix, no new learned parameters, same FP16 STE fields.
"""
import torch

class ActiveTable:
    def __init__(self,max_gather_elements=4*1024*1024):
        self.limit=max_gather_elements
        self.plans={}
        self.stats={"calls":0,"plan_builds":0,"plan_hits":0,"gather_calls":0,"max_gather_elements":0,"active_widths":{}}
    def plan(self,source):
        codes=source.codes
        stamp=(id(codes),codes.data_ptr(),codes._version,str(codes.device),tuple(source.shape))
        key=id(source)
        old=self.plans.get(key)
        if old is not None and old[0]==stamp:
            self.stats["plan_hits"]+=1
            return old[1:]
        o,i=source.shape;g=i//8
        assert i%8==0 and codes.numel()==o*g
        lo,hi=int(codes.min()),int(codes.max())
        assert lo>=0 and hi<source.centroids.shape[0]
        k=next(k for k in (8,16,32) if k>hi)
        assert k<=source.centroids.shape[0]
        picks=(codes.reshape(o,g)+torch.arange(g,device=codes.device)*k).contiguous()
        self.plans[key]=(stamp,k,picks)
        self.stats["plan_builds"]+=1
        self.stats["active_widths"][str(key)]=k
        return k,picks
    def __call__(self,x,source):
        c,s=source.fields();o,i=source.shape;g=i//8
        k,picks=self.plan(source)
        assert x.shape[-1]==i and c.shape[1]==8
        flat=x.reshape(-1,g,8)
        table=(flat@c[:k].T).reshape(-1,g*k)
        rows=max(1,min(o,self.limit//max(1,flat.shape[0]*g)))
        result=[]
        self.stats["calls"]+=1
        for start in range(0,o,rows):
            pick=picks[start:start+rows].reshape(-1)
            self.stats["gather_calls"]+=1
            self.stats["max_gather_elements"]=max(self.stats["max_gather_elements"],flat.shape[0]*pick.numel())
            part=table.index_select(1,pick).reshape(flat.shape[0],-1,g).sum(-1)
            result.append(part*s[start:start+rows])
        return torch.cat(result,dim=-1).reshape(*x.shape[:-1],o)
