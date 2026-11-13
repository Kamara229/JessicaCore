"""NRA2 S3K: trainable node/span-bound neural graph bridge on real CPU tokens.

Node mention identities and consecutive mention windows originate ONLY
from literal user prompt text and authentic tokenizer offsets (S3J).
The learned neural net predicts adjacency and query endpoints from
frozen 43M contextual hidden tokens, never gold at inference.
This is a research *shadow* adapter with output gate 0, not a decoder.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

NODES=6
WIDTH=320
H=48


@dataclass
class EntityBinding:
    mention_mask: torch.Tensor   # [B,T,6] bool
    event_mask: torch.Tensor     # [B,E,T] bool; contextual tokens BETWEEN adjacent mentions
    event_pairs: torch.Tensor    # [B,E,2] long; textual ordered mention identities
    event_valid: torch.Tensor    # [B,E] bool
    query_order: torch.Tensor    # [B,2] long; last two user-supplied mention identities


def bind_mentions_to_context(aligned, tokens_length):
    """Use S3J offset indices only. No graph labels, edges or source gold.

    Consecutive text-mention pairs are *candidates*, NOT graph edges.
    Crucially, query pairs also enter these candidates; the neural
    scorer must distinguish 'R1 -> R2' from 'can R1 reach R2?' itself.
    """
    if not aligned or not isinstance(tokens_length,int) or tokens_length<2:
        raise AssertionError("S3K requires actual tokenizer alignment and sequence length")
    b=len(aligned)
    events=[]
    for a in aligned:
        if a["tokens"]>tokens_length or len(a["mention_mask"])!=a["tokens"]:
            raise AssertionError("S3K offset-vs-real-hidden token length mismatch")
        ms=a["mentions"]
        if len(ms)<2 or ms[-1].node==ms[-2].node:
            raise AssertionError("S3K query last two mentions missing/distinctness invalid")
        chosen=[]
        for lhs,rhs in zip(ms,ms[1:]):
            if lhs.node==rhs.node:
                continue
            start=max(lhs.token_positions)+1
            stop=min(rhs.token_positions)
            if stop<start or stop>tokens_length or start<0:
                raise AssertionError("S3K invalid source prompt consecutive mention ordering")
            chosen.append((lhs.node,rhs.node,start,stop))
        if not chosen:
            raise AssertionError("S3K has no adjacent textual mention pairs")
        events.append(chosen)
    e=max(len(z) for z in events)
    mention=torch.zeros((b,tokens_length,NODES),dtype=torch.bool)
    segments=torch.zeros((b,e,tokens_length),dtype=torch.bool)
    owners=torch.full((b,e,2),-1,dtype=torch.long)
    valid=torch.zeros((b,e),dtype=torch.bool)
    query=torch.zeros((b,2),dtype=torch.long)
    for i,(a,chosen) in enumerate(zip(aligned,events)):
        mention[i,:a["tokens"]]=torch.tensor(a["mention_mask"],dtype=torch.bool)
        query[i]=torch.tensor([a["mentions"][-2].node,a["mentions"][-1].node])
        for j,(src,dst,start,stop) in enumerate(chosen):
            owners[i,j]=torch.tensor([src,dst])
            valid[i,j]=True
            if stop>start:
                segments[i,j,start:stop]=True
    if (not bool(mention.any(dim=1).sum(dim=1).ge(2).all())
        or not bool(valid.any(dim=1).all())
        or bool((mention.sum(dim=-1)>1).any())
        or bool((segments & mention.any(dim=-1).unsqueeze(1)).any())):
        raise AssertionError("S3K ambiguous node token masks or malformed candidate spans")
    return EntityBinding(mention,segments,owners,valid,query)


class NodeSpanBoundGraph(nn.Module):
    """Entity-specific pooling and learned TEXTUAL edge witness scoring.

    No learned global node queries: each node vector comes strictly from
    its named-entity occurrences in the full frozen token sequence.
    """
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.LayerNorm(WIDTH),nn.Linear(WIDTH,H),nn.GELU())
        self.node_norm=nn.LayerNorm(H)
        self.span_norm=nn.LayerNorm(H)
        self.event_edge=nn.Sequential(
            nn.Linear(H*3+1,H*2),nn.GELU(),nn.Linear(H*2,1))
        self.endpoint=nn.Sequential(nn.Linear(H+2,H),nn.GELU(),nn.Linear(H,2))
        self.no_witness_logit=nn.Parameter(torch.tensor(-3.5))

    def forward(self,tokens,token_mask,binding:EntityBinding):
        if (tokens.ndim!=3 or tokens.shape[-1]!=WIDTH
            or token_mask.shape!=tokens.shape[:2] or token_mask.dtype!=torch.bool
            or not torch.isfinite(tokens).all()):
            raise AssertionError("S3K requires valid real [B,T,320] frozen contextual tokens")
        b,t,_=tokens.shape
        m=binding.mention_mask
        e=binding.event_mask
        p=binding.event_pairs
        v=binding.event_valid
        q=binding.query_order
        if (m.shape!=(b,t,NODES) or m.dtype!=torch.bool
            or e.ndim!=3 or e.shape[0]!=b or e.shape[2]!=t
            or e.dtype!=torch.bool or p.shape!=(b,e.shape[1],2)
            or v.shape!=(b,e.shape[1]) or v.dtype!=torch.bool
            or q.shape!=(b,2) or p.dtype!=torch.long or q.dtype!=torch.long
            or not bool(token_mask.any(dim=1).all())
            or bool((m & ~token_mask.unsqueeze(-1)).any())
            or bool((e & ~token_mask.unsqueeze(1)).any())
            or bool(((q<0)|(q>=NODES)).any())
            or bool(((p[v]<0)|(p[v]>=NODES)).any())):
            raise AssertionError("S3K invalid or padded entity-bound feature ABI")
        x=self.project(tokens.float())
        # Aggregate ALL literal mention occurrences of the exact node.
        node_sum=torch.einsum("btn,btd->bnd",m.float(),x)
        node_count=m.sum(dim=1)
        node=self.node_norm(node_sum/node_count.clamp(min=1).unsqueeze(-1))
        node=node*node_count.gt(0).unsqueeze(-1)
        # Text between neighboring mentions contains cues such as "->",
        # "to", "from", and query wording. No arrow is hardcoded as a gold edge.
        segment=torch.einsum("bet,btd->bed",e.float(),x)
        segment=segment/e.sum(dim=-1).clamp(min=1).unsqueeze(-1)
        segment=self.span_norm(segment)
        source=torch.gather(node,1,p[:,:,0].clamp(min=0).unsqueeze(-1).expand(-1,-1,H))
        target=torch.gather(node,1,p[:,:,1].clamp(min=0).unsqueeze(-1).expand(-1,-1,H))
        length=e.sum(dim=-1).float().clamp(max=24).unsqueeze(-1)/24.
        evidence=self.event_edge(torch.cat([source,target,segment,length],dim=-1)).squeeze(-1)
        evidence=evidence.masked_fill(~v,-50.)
        # Multiple textual witnesses for one directed pair are aggregated
        # differentiably; a pair with no mention-to-mention witness is negative.
        matching=(
            (p[:,:,0].unsqueeze(1).unsqueeze(1)
                 ==torch.arange(NODES,device=p.device)[None,:,None,None])
            &(p[:,:,1].unsqueeze(1).unsqueeze(1)
                 ==torch.arange(NODES,device=p.device)[None,None,:,None])
            &v.unsqueeze(1).unsqueeze(1)
        )
        per_pair=evidence[:,None,None,:].expand(-1,NODES,NODES,-1)
        per_pair=per_pair.masked_fill(~matching,-50.)
        null=self.no_witness_logit.expand(b,NODES,NODES,1)
        edge=torch.logsumexp(torch.cat([per_pair,null],dim=-1),dim=-1)
        present=node_count.gt(0)
        permissible=present[:,:,None]&present[:,None,:]
        diagonal=torch.eye(NODES,dtype=torch.bool,device=edge.device)
        edge=edge.masked_fill(~permissible|diagonal[None,:,:],-14.)

        # Query entity identities from FINAL TWO textual mentions, never
        # labeled (src,dst). The network learns their semantic roles.
        ordinal=F.one_hot(q,num_classes=NODES).float() # [B,2,N]
        role=ordinal.transpose(1,2)                  # [B,N,2]
        endpoints=self.endpoint(torch.cat([node,role],dim=-1))
        src=endpoints[:,:,0].masked_fill(~present,-14.)
        dst=endpoints[:,:,1].masked_fill(~present,-14.)
        return edge,src,dst


def graph_forward(model,nra,tokens,mask,binding):
    edge,src,dst=model(tokens,mask,binding)
    reach=nra(torch.sigmoid(edge),F.softmax(src,dim=-1),F.softmax(dst,dim=-1))
    return edge,src,dst,reach
