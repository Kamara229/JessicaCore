"""S3O: trainable syntax-conditioned QUERY roles over frozen real CPU tokens.

This augments S3K's previously working node-attached graph edge
extractor with a neural query role classifier. Neither endpoint gold nor
graph adjacency/answer enters inference. Literal token spans are from
the S160 native tokenizer; local context around the two query mentions
disambiguates "from A to B" vs "can B be reached from A".
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from research.nra2_bridge.stage3j_token_entity_alignment import (
    content_in_chat,
)
from research.nra2_bridge.stage3k_neural_entity_bridge import (
    bind_mentions_to_context,NodeSpanBoundGraph,NODES,WIDTH,H,EntityBinding,
)


@dataclass
class QueryTextMask:
    query_mask: torch.Tensor            # [B,T] bool
    anchor_mask: torch.Tensor           # [B,2,T] bool
    left_context_mask: torch.Tensor     # [B,2,T] bool
    right_context_mask: torch.Tensor    # [B,2,T] bool
    node_order: torch.Tensor            # [B,2] (from text, not gold)
    relative_positions: torch.Tensor   # [B,2], dimensionless text positions


def build_query_text_mask(tokenizer,questions,alignment,sequence_length):
    """Align actual question phrase and both entity mentions to tokenizer.

    Only literal question text is used. Find the final '?' and the
    clause following the final dot+space separator. A prompt with a
    different syntax fails closed rather than pretending that the
    last two entities are the true source and destination.
    """
    if (len(questions)!=len(alignment) or sequence_length<2
        or len(questions)==0):
        raise AssertionError("S3O query-text alignment arguments invalid")
    n=len(questions)
    result=QueryTextMask(
        query_mask=torch.zeros((n,sequence_length),dtype=torch.bool),
        anchor_mask=torch.zeros((n,2,sequence_length),dtype=torch.bool),
        left_context_mask=torch.zeros((n,2,sequence_length),dtype=torch.bool),
        right_context_mask=torch.zeros((n,2,sequence_length),dtype=torch.bool),
        node_order=torch.zeros((n,2),dtype=torch.long),
        relative_positions=torch.zeros((n,2),dtype=torch.float32))
    for i,(question,a) in enumerate(zip(questions,alignment)):
        chat,content_start=content_in_chat(tokenizer,question)
        q_end=question.find("?")
        delim=question.rfind(". ",0,q_end)
        if q_end<0 or question.find("?",q_end+1)>=0 or delim<0:
            raise AssertionError("S3O cannot isolate unique final query clause")
        q_start=delim+2
        if q_start>=q_end:
            raise AssertionError("S3O query clause empty")
        original=list(tokenizer.encode(chat,add_special_tokens=False))
        mapped=tokenizer(chat,return_offsets_mapping=True,add_special_tokens=False)
        offsets=[tuple(x) for x in mapped["offset_mapping"]]
        if (original!=list(mapped["input_ids"]) or original!=a["input_ids"]
            or a["tokens"]!=len(original) or len(original)>sequence_length):
            raise AssertionError("S3O genuine frozen S160 token alignment drift")
        query_char_begin=content_start+q_start
        query_char_end=content_start+q_end+1
        for j,(left,right) in enumerate(offsets):
            if left<query_char_end and right>query_char_begin:
                result.query_mask[i,j]=True
        matches=[m for m in a["mentions"]
                 if q_start<=m.char_start and m.char_end<=q_end]
        if len(matches)!=2 or matches[0].node==matches[1].node:
            raise AssertionError("S3O query phrase needs exactly 2 different entity mentions")
        for k,m in enumerate(matches):
            if any(t>=sequence_length for t in m.token_positions):
                raise AssertionError("S3O entity token outside read-only CPU tensor")
            if any(not result.query_mask[i,t] for t in m.token_positions):
                raise AssertionError("S3O query entity escaped query token span")
            result.node_order[i,k]=m.node
            for t in m.token_positions:
                result.anchor_mask[i,k,t]=True
            first=min(m.token_positions)
            last=max(m.token_positions)
            result.left_context_mask[i,k,max(0,first-9):first]=(
                result.query_mask[i,max(0,first-9):first]
            )
            result.right_context_mask[i,k,last+1:last+10]=(
                result.query_mask[i,last+1:last+10]
            )
            result.relative_positions[i,k]=(m.char_start-q_start)/max(1,q_end-q_start)
    if (not bool(result.query_mask.any(dim=1).all())
        or not bool(result.anchor_mask.any(dim=2).all())
        or bool((result.anchor_mask & ~result.query_mask[:,None,:]).any())
        or bool((result.node_order[:,0]==result.node_order[:,1]).any())):
        raise AssertionError("S3O query span mask/cardinality invalid")
    return result


class LearnedQueryRoleHead(nn.Module):
    """Neural binary source/destination assignment for query mentions.

    Treats two LITERAL query mention identities as unordered candidates.
    Actual source/destination roles are LEARNED from contextual spans.
    No fixed last-two textual order is equated with semantic source.
    """
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.LayerNorm(WIDTH),nn.Linear(WIDTH,H),nn.GELU())
        self.score=nn.Sequential(
            nn.Linear(4*H+1,2*H),nn.GELU(),
            nn.Linear(2*H,H),nn.GELU(),
            nn.Linear(H,1))

    @staticmethod
    def pool(x,mask):
        return torch.einsum("bkt,btd->bkd",mask.float(),x)/(
            mask.float().sum(dim=-1).clamp(min=1).unsqueeze(-1)
        )

    def forward(self,tokens,token_mask,q:QueryTextMask):
        b,t,d=tokens.shape
        if (d!=WIDTH or token_mask.shape!=(b,t) or token_mask.dtype!=torch.bool
            or q.query_mask.shape!=(b,t) or q.anchor_mask.shape!=(b,2,t)
            or q.left_context_mask.shape!=(b,2,t)
            or q.right_context_mask.shape!=(b,2,t)
            or q.node_order.shape!=(b,2)
            or q.relative_positions.shape!=(b,2)
            or bool(((q.node_order<0)|(q.node_order>=NODES)).any())
            or bool((q.node_order[:,0]==q.node_order[:,1]).any())
            or bool((q.query_mask & ~token_mask).any())
            or bool((q.anchor_mask & ~q.query_mask[:,None,:]).any())
            or bool((q.left_context_mask & ~q.query_mask[:,None,:]).any())
            or bool((q.right_context_mask & ~q.query_mask[:,None,:]).any())
            or not bool(q.anchor_mask.any(dim=-1).all())
            or not torch.isfinite(tokens).all()):
            raise AssertionError("S3O query-role tokens/offsets not valid")
        x=self.encoder(tokens.float())
        anchor=self.pool(x,q.anchor_mask)
        left=self.pool(x,q.left_context_mask)
        right=self.pool(x,q.right_context_mask)
        question=self.pool(x,q.query_mask[:,None,:]).expand(-1,2,-1)
        role_features=torch.cat(
            [anchor,left,right,question,q.relative_positions[:,:,None]],dim=-1
        )
        role_scores=self.score(role_features).squeeze(-1)
        # Differentiable restricted softmax: exactly two entities from
        # query text can be src/dst. The order is not assumed correct.
        src=torch.full((b,NODES),-14.,device=tokens.device,dtype=torch.float32)
        dst=torch.full_like(src,-14.)
        src=src.scatter(1,q.node_order,role_scores)
        dst=dst.scatter(1,q.node_order,-role_scores)
        return src,dst


class S3KGraphWithLearnedRoles(nn.Module):
    """Keep S3K's explicit entity-bound edge path, replace fragile role head."""
    def __init__(self):
        super().__init__()
        self.graph=NodeSpanBoundGraph()
        self.query=LearnedQueryRoleHead()

    def forward(self,tokens,token_mask,binding:EntityBinding,query:QueryTextMask):
        edges,_old_src,_old_dst=self.graph(tokens,token_mask,binding)
        src,dst=self.query(tokens,token_mask,query)
        if not torch.isfinite(edges).all():
            raise AssertionError("S3O nonfinite graph branch")
        return edges,src,dst


def graph_forward(model,nra,tokens,mask,binding,query):
    edges,src,dst=model(tokens,mask,binding,query)
    reach=nra(torch.sigmoid(edges),F.softmax(src,dim=-1),F.softmax(dst,dim=-1))
    return edges,src,dst,reach
