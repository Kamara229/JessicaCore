"""S3J: tokenizer-offset node mention masks, research-only and LABEL-FREE.

This file maps literal R0..R5 mentions from user prompt text to the
**same frozen tokenizer** prefill positions used by Calibri 43M S160.
No graph edges, source-target gold, reachable labels, model logits or
answers are accepted or returned. Do not use as an oracle for edges.
Works with a fast tokenizer with exact character offsets; fails closed
rather than guess when tokenization or offset alignment is ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

MENTION = re.compile(r"(?<![A-Za-z0-9_])R([0-5])(?![A-Za-z0-9_])")
NODES = 6


@dataclass(frozen=True)
class Mention:
    node: int
    char_start: int
    char_end: int
    token_positions: tuple[int, ...]


def content_in_chat(tok, question):
    if not isinstance(question,str) or not question:
        raise AssertionError("S3J empty question")
    full=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    # Chat template must contain the user's literal text exactly once;
    # don't use a prompt-wide regular expression (it could hit roles).
    if not isinstance(full,str) or full.count(question)!=1:
        raise AssertionError("S3J ambiguous tokenizer chat template")
    return full,full.index(question)


def node_mention_alignment(tok, question, expected_ids=None):
    """Return [T,6] binary mention mask *and* occurrence spans.

    The tokenizer handles chat prefixes/suffixes. A byte pair may
    include punctuation, so use actual offset mappings, not single-token
    'R1' lookups, approximate substring index, or encode('R1').
    """
    full,user_offset=content_in_chat(tok,question)
    original_ids=list(tok.encode(full,add_special_tokens=False))
    if expected_ids is not None and original_ids!=list(expected_ids):
        raise AssertionError("S3J frozen S160 prefill ids drift")
    try:
        encoded=tok(full,add_special_tokens=False,
                    return_offsets_mapping=True)
    except Exception as exc:
        raise AssertionError("S3J requires true tokenizer offset mappings") from exc
    ids=list(encoded["input_ids"])
    offsets=[tuple(p) for p in encoded["offset_mapping"]]
    if ids!=original_ids or len(offsets)!=len(ids) or len(ids)<2:
        raise AssertionError("S3J tokenizer IDs or offset ABI drift")
    if any(len(o)!=2 or not all(isinstance(v,int) for v in o)
           or o[0]<0 or o[1]<o[0] or o[1]>len(full) for o in offsets):
        raise AssertionError("S3J invalid tokenizer offsets")

    mentions=[]
    positions_used={}
    mask=[[0]*NODES for _ in ids]
    for hit in MENTION.finditer(question):
        node=int(hit.group(1))
        begin=user_offset+hit.start()
        end=user_offset+hit.end()
        positions=tuple(i for i,(left,right) in enumerate(offsets)
                        if left<end and right>begin)
        if not positions:
            raise AssertionError("S3J mention has no offset-aligned tokens")
        # Reject multi-entity tokens to avoid treating one semantic token
        # as two independently identifiable entity representations.
        if any(i in positions_used and positions_used[i]!=node
               for i in positions):
            raise AssertionError("S3J token merges distinct node identities")
        if any(not (offsets[i][1]>offsets[i][0]) for i in positions):
            raise AssertionError("S3J mention overlaps special/empty token")
        for i in positions:
            positions_used[i]=node
            mask[i][node]=1
        mentions.append(Mention(node,hit.start(),hit.end(),positions))
    if not mentions or len({m.node for m in mentions})<2:
        raise AssertionError("S3J need two or more named graph entities")
    # No label, semantic adjacency, query answer or graph target.
    return {
        "prompt_sha256":hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "input_ids":ids,
        "mention_mask":mask,
        "mentions":mentions,
        "tokens":len(ids),
        "node_identities_present":sorted({m.node for m in mentions}),
    }


def validate_without_gold(tok, questions, prefill_id_sequences=None):
    if not questions or len(questions)!=len(set(questions)):
        raise AssertionError("S3J requires distinct research prompts")
    if prefill_id_sequences is not None and len(questions)!=len(prefill_id_sequences):
        raise AssertionError("S3J prefill ID sequence count mismatch")
    ret=[]
    for n,question in enumerate(questions):
        ids=None if prefill_id_sequences is None else prefill_id_sequences[n]
        ret.append(node_mention_alignment(tok,question,ids))
    if len({r["prompt_sha256"] for r in ret})!=len(questions):
        raise AssertionError("S3J duplicate prompt digest")
    return ret
